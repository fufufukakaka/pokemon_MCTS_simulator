"""
Decision Transformer Guided MCTS

Decision Transformerを使ってMCTSの探索をガイドする。
- Policy: 展開時の事前確率として使用（UCBで優先度付け）
- Value: 終端評価として使用（rolloutの代わり）

数ターン先を読むことで、短絡的でない安定択を見つける。
"""

from __future__ import annotations

import logging
import math
import random
from copy import deepcopy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

import torch
import torch.nn.functional as F

from src.pokemon_battle_sim.battle import Battle
from src.pokemon_battle_sim.pokemon import Pokemon

from .config import PokemonBattleTransformerConfig
from .dataset import ObservationTracker, TurnState

if TYPE_CHECKING:
    from .model import PokemonBattleTransformer
    from .tokenizer import BattleSequenceTokenizer

logger = logging.getLogger(__name__)


@dataclass
class DTGuidedMCTSConfig:
    """DT誘導型MCTSの設定"""

    # MCTS設定
    n_simulations: int = 200  # シミュレーション回数
    c_puct: float = 1.5  # 探索のバランスパラメータ（高いほど探索重視）

    # 先読み深さ
    max_depth: int = 10  # 最大探索深度（ターン数）

    # 温度パラメータ
    temperature: float = 1.0  # 1.0=確率的, 0=貪欲

    # NN使用設定
    use_nn_value: bool = True  # NNのValueを使うか
    use_nn_policy: bool = True  # NNのPolicyを使うか
    nn_value_weight: float = 0.8  # NN Value vs Rollout Valueの重み

    # Rollout設定
    rollout_max_turns: int = 30  # ランダムプレイアウトの最大ターン数

    # 枝刈り
    prune_threshold: float = 0.01  # この確率以下の行動は探索しない

    # デバイス
    device: str = "cpu"


@dataclass
class BattleContext:
    """バトル履歴コンテキスト"""

    my_team: list[str] = field(default_factory=list)
    opp_team: list[str] = field(default_factory=list)
    selection: list[int] = field(default_factory=list)
    lead_idx: int = 0
    turns: list[TurnState] = field(default_factory=list)
    actions: list[int] = field(default_factory=list)
    opp_observation: ObservationTracker = field(default_factory=ObservationTracker)
    prev_opp_active_name: str = ""


class DTGuidedMCTSNode:
    """DT誘導型MCTSのノード"""

    def __init__(
        self,
        state: Battle,
        player: int,
        parent: Optional["DTGuidedMCTSNode"] = None,
        action: Optional[int] = None,
        prior: float = 0.0,
        depth: int = 0,
    ):
        self.state = state
        self.player = player
        self.parent = parent
        self.action = action  # このノードに至った行動
        self.prior = prior  # NNが予測した事前確率
        self.depth = depth  # ルートからの深さ

        self.children: dict[int, DTGuidedMCTSNode] = {}
        self.visit_count = 0
        self.total_value = 0.0
        self.is_expanded = False

    @property
    def q_value(self) -> float:
        """平均価値"""
        if self.visit_count == 0:
            return 0.0
        return self.total_value / self.visit_count

    def ucb_score(self, c_puct: float) -> float:
        """UCB (PUCT) スコア - AlphaZeroスタイル"""
        if self.parent is None:
            return 0.0

        # Q + c_puct * P * sqrt(N_parent) / (1 + N)
        exploration = (
            c_puct
            * self.prior
            * math.sqrt(self.parent.visit_count)
            / (1 + self.visit_count)
        )
        return self.q_value + exploration

    def select_child(self, c_puct: float) -> "DTGuidedMCTSNode":
        """UCBスコアが最大の子ノードを選択"""
        return max(self.children.values(), key=lambda c: c.ucb_score(c_puct))

    def best_action(self, temperature: float = 1.0) -> int:
        """訪問回数に基づいて最良の行動を選択"""
        if not self.children:
            return Battle.SKIP

        if temperature == 0:
            # 貪欲選択
            return max(self.children.items(), key=lambda x: x[1].visit_count)[0]
        else:
            # 確率的選択
            visits = [child.visit_count for child in self.children.values()]
            total = sum(visits)
            if total == 0:
                return random.choice(list(self.children.keys()))

            # 温度でスケール
            probs = [(v / total) ** (1 / temperature) for v in visits]
            prob_sum = sum(probs)
            probs = [p / prob_sum for p in probs]

            actions = list(self.children.keys())
            return random.choices(actions, weights=probs)[0]

    def get_policy(self) -> dict[int, float]:
        """訪問回数に基づくPolicy分布を取得"""
        if not self.children:
            return {}

        total_visits = sum(child.visit_count for child in self.children.values())
        if total_visits == 0:
            n = len(self.children)
            return {action: 1.0 / n for action in self.children.keys()}

        return {
            action: child.visit_count / total_visits
            for action, child in self.children.items()
        }

    def get_analysis(self) -> dict[int, dict]:
        """各行動の詳細分析を取得"""
        analysis = {}
        for action, child in self.children.items():
            analysis[action] = {
                "visits": child.visit_count,
                "q_value": child.q_value,
                "prior": child.prior,
                "ucb": child.ucb_score(1.5),
            }
        return analysis


class DTGuidedMCTS:
    """
    Decision Transformer 誘導型MCTS

    DTのPolicyとValueを使ってMCTS探索をガイドする。
    数ターン先を読むことで、目先の利益だけでなく長期的に有利な手を選ぶ。
    """

    def __init__(
        self,
        model: "PokemonBattleTransformer",
        tokenizer: "BattleSequenceTokenizer",
        config: Optional[DTGuidedMCTSConfig] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or DTGuidedMCTSConfig()

        self.model.eval()
        self.device = torch.device(self.config.device)
        self.model.to(self.device)

        # コンテキストキャッシュ
        self._context_cache: dict[int, BattleContext] = {}

    def search(
        self,
        battle: Battle,
        player: int,
        context: Optional[BattleContext] = None,
        target_return: float = 1.0,
    ) -> tuple[dict[int, float], float, dict[int, dict]]:
        """
        MCTS探索を実行

        Args:
            battle: 現在の対戦状態
            player: 行動するプレイヤー
            context: バトル履歴コンテキスト
            target_return: 目標リターン（1.0=勝利）

        Returns:
            (policy, value, analysis):
                - policy: 行動確率分布
                - value: 推定勝率
                - analysis: 各行動の詳細分析
        """
        # コンテキストを取得または作成
        if context is None:
            context = self._get_or_create_context(battle, player)

        # ルートノード作成
        root = DTGuidedMCTSNode(state=deepcopy(battle), player=player, depth=0)

        # ルートノードを展開
        self._expand(root, context, target_return)

        # シミュレーション
        for _ in range(self.config.n_simulations):
            node = root
            search_path = [node]

            # Selection: 葉ノードまで降りる
            while node.is_expanded and node.children:
                node = node.select_child(self.config.c_puct)
                search_path.append(node)

            # 最大深度に達していなければ展開
            if (
                node.state.winner() is None
                and not node.is_expanded
                and node.depth < self.config.max_depth
            ):
                self._expand(node, context, target_return)

            # Evaluation
            value = self._evaluate(node, player, context, target_return)

            # Backpropagation
            self._backup(search_path, value, player)

        # 結果を返す
        policy = root.get_policy()
        value = root.q_value
        analysis = root.get_analysis()

        return policy, value, analysis

    def _expand(
        self,
        node: DTGuidedMCTSNode,
        context: BattleContext,
        target_return: float,
    ):
        """ノードを展開"""
        if node.state.winner() is not None:
            node.is_expanded = True
            return

        # pokemon が None の場合は交代フェーズ
        if node.state.pokemon[node.player] is None:
            phase = "change"
        else:
            phase = "battle"

        available = node.state.available_commands(node.player, phase=phase)
        if not available:
            node.is_expanded = True
            return

        # NNでPolicyを取得
        if self.config.use_nn_policy:
            nn_policy = self._get_nn_policy(
                node.state, node.player, context, target_return
            )
        else:
            # 均等分布
            nn_policy = {cmd: 1.0 / len(available) for cmd in available}

        # 子ノード作成
        for action in available:
            prior = nn_policy.get(action, 1.0 / len(available))

            # 枝刈り: 確率が低すぎる行動はスキップ
            if prior < self.config.prune_threshold and len(available) > 1:
                continue

            # 次状態を作成
            child_state = self._create_child_state(node.state, node.player, action)

            child = DTGuidedMCTSNode(
                state=child_state,
                player=node.player,
                parent=node,
                action=action,
                prior=prior,
                depth=node.depth + 1,
            )
            node.children[action] = child

        node.is_expanded = True

    def _create_child_state(
        self,
        state: Battle,
        player: int,
        action: int,
    ) -> Battle:
        """子状態を作成（相手の行動はランダム）"""
        child_state = deepcopy(state)

        # 相手の行動をランダムに選択
        opp = 1 - player

        # 相手の行動を決定
        # available_commands は両方の pokemon が必要な場合があるため、
        # 安全にフォールバックを用意
        try:
            if child_state.pokemon[opp] is None:
                opp_phase = "change"
            else:
                opp_phase = "battle"
            opp_actions = child_state.available_commands(opp, phase=opp_phase)
            opp_action = random.choice(opp_actions) if opp_actions else Battle.SKIP
        except (AttributeError, TypeError):
            # pokemon が None で available_commands が失敗した場合
            opp_action = Battle.SKIP

        # コマンド実行
        if player == 0:
            commands = [action, opp_action]
        else:
            commands = [opp_action, action]

        child_state.proceed(commands=commands)

        return child_state

    def _get_nn_policy(
        self,
        state: Battle,
        player: int,
        context: BattleContext,
        target_return: float,
    ) -> dict[int, float]:
        """NNからPolicy分布を取得"""
        # pokemon が None の場合は交代フェーズ
        if state.pokemon[player] is None:
            phase = "change"
        else:
            phase = "battle"
        available = state.available_commands(player, phase=phase)
        if not available:
            return {}

        # 観測情報を更新
        self._update_observation(state, player, context)

        # エンコード（Battle オブジェクトを直接使用）
        turn_number = len(context.actions)
        encoded = self._encode_context_from_battle(
            battle=state,
            player=player,
            context=context,
            turn=turn_number,
            target_return=target_return,
        )

        # 行動マスクを作成
        action_mask = torch.zeros(self.model.config.num_action_outputs)
        for cmd in available:
            if 0 <= cmd < self.model.config.num_action_outputs:
                action_mask[cmd] = 1.0
        action_mask = action_mask.to(self.device)

        # NNで推論
        with torch.no_grad():
            # バッチ次元を追加
            input_ids = encoded["input_ids"].unsqueeze(0).to(self.device)
            position_ids = encoded["position_ids"].unsqueeze(0).to(self.device)
            timestep_ids = encoded["timestep_ids"].unsqueeze(0).to(self.device)
            segment_ids = encoded["segment_ids"].unsqueeze(0).to(self.device)
            rtg_values = encoded["rtg_values"].unsqueeze(0).to(self.device)
            attention_mask = encoded["attention_mask"].unsqueeze(0).to(self.device)
            action_mask_batch = action_mask.unsqueeze(0)

            state_features = None
            if "state_features" in encoded:
                state_features = encoded["state_features"].unsqueeze(0).to(self.device)

            outputs = self.model.forward(
                input_ids=input_ids,
                position_ids=position_ids,
                timestep_ids=timestep_ids,
                segment_ids=segment_ids,
                rtg_values=rtg_values,
                attention_mask=attention_mask,
                state_features=state_features,
                action_mask=action_mask_batch,
                use_causal_mask=True,
            )

            # Softmaxで確率に変換
            logits = outputs["action_logits"].squeeze(0)
            probs = F.softmax(logits, dim=-1)

        # コマンドIDに変換
        result = {}
        for cmd in available:
            if 0 <= cmd < self.model.config.num_action_outputs:
                result[cmd] = probs[cmd].item()
            else:
                result[cmd] = 1.0 / len(available)

        # 正規化
        total = sum(result.values())
        if total > 0:
            result = {k: v / total for k, v in result.items()}

        return result

    def _get_nn_value(
        self,
        state: Battle,
        player: int,
        context: BattleContext,
        target_return: float,
    ) -> float:
        """NNからValue（勝率）を取得"""
        # 観測情報を更新
        self._update_observation(state, player, context)

        # エンコード（Battle オブジェクトを直接使用）
        turn_number = len(context.actions)
        encoded = self._encode_context_from_battle(
            battle=state,
            player=player,
            context=context,
            turn=turn_number,
            target_return=target_return,
        )

        with torch.no_grad():
            # バッチ次元を追加
            input_ids = encoded["input_ids"].unsqueeze(0).to(self.device)
            position_ids = encoded["position_ids"].unsqueeze(0).to(self.device)
            timestep_ids = encoded["timestep_ids"].unsqueeze(0).to(self.device)
            segment_ids = encoded["segment_ids"].unsqueeze(0).to(self.device)
            rtg_values = encoded["rtg_values"].unsqueeze(0).to(self.device)
            attention_mask = encoded["attention_mask"].unsqueeze(0).to(self.device)

            state_features = None
            if "state_features" in encoded:
                state_features = encoded["state_features"].unsqueeze(0).to(self.device)

            # 行動マスクはダミー
            action_mask = torch.ones(
                1, self.model.config.num_action_outputs, device=self.device
            )

            outputs = self.model.forward(
                input_ids=input_ids,
                position_ids=position_ids,
                timestep_ids=timestep_ids,
                segment_ids=segment_ids,
                rtg_values=rtg_values,
                attention_mask=attention_mask,
                state_features=state_features,
                action_mask=action_mask,
                use_causal_mask=True,
            )

            value = outputs["value"].squeeze().item()

        return value

    def _evaluate(
        self,
        node: DTGuidedMCTSNode,
        root_player: int,
        context: BattleContext,
        target_return: float,
    ) -> float:
        """ノードを評価"""
        # 勝敗が決まっていれば確定値
        winner = node.state.winner()
        if winner is not None:
            return 1.0 if winner == root_player else 0.0

        if self.config.use_nn_value:
            # NNで評価
            nn_value = self._get_nn_value(
                node.state, root_player, context, target_return
            )

            if self.config.nn_value_weight >= 1.0:
                return nn_value
            else:
                # Rolloutと組み合わせ
                rollout_value = self._rollout(node.state, root_player)
                return (
                    self.config.nn_value_weight * nn_value
                    + (1 - self.config.nn_value_weight) * rollout_value
                )
        else:
            return self._rollout(node.state, root_player)

    def _rollout(self, state: Battle, player: int) -> float:
        """ランダムプレイアウト"""
        sim_state = deepcopy(state)

        for _ in range(self.config.rollout_max_turns):
            if sim_state.winner() is not None:
                break

            # pokemon が None の場合は交代フェーズ
            phase0 = "change" if sim_state.pokemon[0] is None else "battle"
            phase1 = "change" if sim_state.pokemon[1] is None else "battle"
            moves0 = sim_state.available_commands(0, phase=phase0)
            moves1 = sim_state.available_commands(1, phase=phase1)
            cmd0 = random.choice(moves0) if moves0 else Battle.SKIP
            cmd1 = random.choice(moves1) if moves1 else Battle.SKIP
            sim_state.proceed(commands=[cmd0, cmd1])

        winner = sim_state.winner()
        if winner is not None:
            return 1.0 if winner == player else 0.0

        # 決着がつかなかった場合はHPベースの評価
        return self._calculate_hp_score(sim_state, player)

    def _calculate_hp_score(self, state: Battle, player: int) -> float:
        """HPベースのスコアを計算"""
        my_total = 0.0
        my_max = 0.0
        opp_total = 0.0
        opp_max = 0.0

        for p in state.selected[player]:
            if p is not None:
                my_total += p.hp
                my_max += p.status[0] if p.status[0] > 0 else 1

        for p in state.selected[1 - player]:
            if p is not None:
                opp_total += p.hp
                opp_max += p.status[0] if p.status[0] > 0 else 1

        my_ratio = my_total / my_max if my_max > 0 else 0
        opp_ratio = opp_total / opp_max if opp_max > 0 else 0

        # 0.5を中心にスケール
        return 0.5 + (my_ratio - opp_ratio) / 2

    def _backup(
        self,
        search_path: list[DTGuidedMCTSNode],
        value: float,
        root_player: int,
    ):
        """バックプロパゲーション"""
        for node in reversed(search_path):
            node.visit_count += 1
            # ノードのプレイヤー視点の価値に変換
            node_value = value if node.player == root_player else 1 - value
            node.total_value += node_value

    def _get_or_create_context(self, battle: Battle, player: int) -> BattleContext:
        """コンテキストを取得または作成"""
        opponent = 1 - player

        if player not in self._context_cache:
            my_team = [p.name for p in battle.selected[player] if p]
            opp_team = [p.name for p in battle.selected[opponent] if p]

            opp_observation = ObservationTracker()

            opp_active = battle.pokemon[opponent]
            prev_opp_active_name = ""
            if opp_active:
                opp_observation.reveal_pokemon(opp_active.name)
                self._detect_initial_ability(opp_active, opp_observation)
                prev_opp_active_name = opp_active.name

            self._context_cache[player] = BattleContext(
                my_team=my_team,
                opp_team=opp_team,
                opp_observation=opp_observation,
                prev_opp_active_name=prev_opp_active_name,
            )

        return self._context_cache[player]

    def _update_observation(self, state: Battle, player: int, context: BattleContext):
        """観測情報を更新"""
        opponent = 1 - player
        opp_active = state.pokemon[opponent]

        if opp_active and opp_active.name != context.prev_opp_active_name:
            context.opp_observation.reveal_pokemon(opp_active.name)
            self._detect_initial_ability(opp_active, context.opp_observation)
            context.prev_opp_active_name = opp_active.name

    def _detect_initial_ability(
        self, pokemon: Pokemon, tracker: ObservationTracker
    ) -> None:
        """場に出た時に発動する特性を検出"""
        instant_abilities = {
            "いかく",
            "ひでり",
            "あめふらし",
            "すなおこし",
            "ゆきふらし",
            "エレキメイカー",
            "グラスメイカー",
            "ミストメイカー",
            "サイコメイカー",
            "おみとおし",
            "かたやぶり",
            "ダウンロード",
            "トレース",
            "よちむ",
            "こだいかっせい",
            "クォークチャージ",
            "ひひいろのこどう",
            "わざわいのうつわ",
            "わざわいのつるぎ",
            "わざわいのおふだ",
            "わざわいのたま",
        }
        ability = pokemon.ability or ""
        if ability in instant_abilities:
            tracker.reveal_ability(pokemon.name, ability)

    def _encode_context_from_battle(
        self,
        battle: Battle,
        player: int,
        context: BattleContext,
        turn: int,
        target_return: float,
    ) -> dict[str, torch.Tensor]:
        """
        Battle オブジェクトから直接コンテキストをエンコード

        tokenizer の正しいインターフェースを使用:
        - encode_team_preview(my_team, opp_team, rtg)
        - encode_selection(selected_indices, lead_index, context, rtg)
        - encode_turn_state(battle, player, turn, rtg, context)
        """
        # 1. チームプレビュー
        encoded = self.tokenizer.encode_team_preview(
            my_team=context.my_team,
            opp_team=context.opp_team,
            rtg=target_return,
        )

        # 2. 選出があれば追加
        if context.selection:
            encoded = self.tokenizer.encode_selection(
                selected_indices=context.selection,
                lead_index=context.lead_idx,
                context=encoded,
                rtg=target_return,
            )

        # 3. 現在のターン状態を追加
        # 相手の観測情報を取得
        opp_pokemon = battle.pokemon[1 - player]
        opp_revealed_moves = None
        opp_revealed_item = None
        opp_revealed_ability = None

        if opp_pokemon and context.opp_observation:
            opp_name = opp_pokemon.name
            if opp_name in context.opp_observation.pokemon_observations:
                obs = context.opp_observation.pokemon_observations[opp_name]
                opp_revealed_moves = obs.revealed_moves if obs.revealed_moves else None
                opp_revealed_item = obs.revealed_item if obs.revealed_item else None
                opp_revealed_ability = (
                    obs.revealed_ability if obs.revealed_ability else None
                )

        encoded = self.tokenizer.encode_turn_state(
            battle=battle,
            player=player,
            turn=turn,
            rtg=target_return,
            context=encoded,
            opp_revealed_moves=opp_revealed_moves,
            opp_revealed_item=opp_revealed_item,
            opp_revealed_ability=opp_revealed_ability,
        )

        return encoded

    def _concat_encoded(
        self,
        encoded1: dict[str, torch.Tensor],
        encoded2: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """エンコード結果を連結"""
        result = {}
        for key in encoded1.keys():
            if key in encoded2:
                result[key] = torch.cat([encoded1[key], encoded2[key]], dim=0)
            else:
                result[key] = encoded1[key]
        return result

    def get_best_action(
        self,
        battle: Battle,
        player: int,
        context: Optional[BattleContext] = None,
        target_return: float = 1.0,
    ) -> int:
        """最良の行動を取得"""
        policy, _, _ = self.search(battle, player, context, target_return)

        if not policy:
            phase = "change" if battle.pokemon[player] is None else "battle"
            available = battle.available_commands(player, phase=phase)
            return random.choice(available) if available else Battle.SKIP

        return max(policy.items(), key=lambda x: x[1])[0]

    def reset(self):
        """コンテキストキャッシュをリセット"""
        self._context_cache.clear()
