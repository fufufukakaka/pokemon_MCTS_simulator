"""
NN誘導型MCTS

学習済みPolicy-Value Networkを使ってMCTSの探索を誘導する。
AlphaZeroスタイルの探索を実現する。
"""

from __future__ import annotations

import math
import random
from copy import deepcopy
from dataclasses import dataclass
from typing import Optional

import torch

from src.hypothesis.hypothesis_mcts import _calculate_battle_score
from src.hypothesis.item_belief_state import ItemBeliefState
from src.hypothesis.item_prior_database import ItemPriorDatabase
from src.hypothesis.selfplay import TurnRecord
from src.pokemon_battle_sim.battle import Battle

from .network import PolicyValueNetwork
from .observation_encoder import ObservationEncoder


@dataclass
class NNGuidedMCTSConfig:
    """NN誘導型MCTSの設定"""

    # MCTS設定
    n_simulations: int = 100  # シミュレーション回数
    c_puct: float = 1.5  # 探索のバランスパラメータ

    # 仮説設定
    n_hypotheses: int = 10  # 仮説サンプリング数

    # 温度パラメータ（行動選択時）
    temperature: float = 1.0  # 1.0=確率的, 0=貪欲

    # NN使用設定
    use_nn_value: bool = True  # NNのValueを使うか
    use_nn_policy: bool = True  # NNのPolicyを使うか
    nn_value_weight: float = 0.5  # NN Value vs Rollout Valueの重み

    # デバイス
    device: str = "cpu"


class NNGuidedMCTSNode:
    """NN誘導型MCTSのノード"""

    def __init__(
        self,
        state: Battle,
        player: int,
        parent: Optional["NNGuidedMCTSNode"] = None,
        action: Optional[int] = None,
        prior: float = 0.0,
    ):
        self.state = state
        self.player = player
        self.parent = parent
        self.action = action  # このノードに至った行動
        self.prior = prior  # NNが予測した事前確率

        self.children: dict[int, NNGuidedMCTSNode] = {}
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
        """UCB (Upper Confidence Bound) スコア"""
        if self.parent is None:
            return 0.0

        # Q + c_puct * P * sqrt(N_parent) / (1 + N)
        exploration = c_puct * self.prior * math.sqrt(self.parent.visit_count) / (1 + self.visit_count)
        return self.q_value + exploration

    def select_child(self, c_puct: float) -> "NNGuidedMCTSNode":
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


class NNGuidedMCTS:
    """
    NN誘導型MCTS

    学習済みNNを使ってMCTSの探索を誘導する。
    - Policy: 展開時の事前確率として使用
    - Value: 終端評価として使用（rolloutの代わり or 補助）
    """

    def __init__(
        self,
        model: PolicyValueNetwork,
        encoder: ObservationEncoder,
        prior_db: ItemPriorDatabase,
        action_vocab: dict[str, int],
        config: Optional[NNGuidedMCTSConfig] = None,
    ):
        self.model = model
        self.encoder = encoder
        self.prior_db = prior_db
        self.action_vocab = action_vocab
        self.id_to_action = {v: k for k, v in action_vocab.items()}
        self.config = config or NNGuidedMCTSConfig()

        self.model.eval()
        self.device = torch.device(self.config.device)
        self.model.to(self.device)

    def search(
        self,
        battle: Battle,
        player: int,
        belief_state: ItemBeliefState,
    ) -> tuple[dict[int, float], float]:
        """
        MCTS探索を実行

        Args:
            battle: 現在の対戦状態
            player: 行動するプレイヤー
            belief_state: 相手持ち物の信念状態

        Returns:
            (policy, value): 行動確率分布と勝率
        """
        # ルートノード作成
        root = NNGuidedMCTSNode(state=deepcopy(battle), player=player)

        # ルートノードを展開
        self._expand(root, belief_state)

        # シミュレーション
        for _ in range(self.config.n_simulations):
            node = root
            search_path = [node]

            # Selection: 葉ノードまで降りる
            while node.is_expanded and node.children:
                node = node.select_child(self.config.c_puct)
                search_path.append(node)

            # 勝敗が決まっていなければ展開
            if node.state.winner() is None and not node.is_expanded:
                self._expand(node, belief_state)

            # Evaluation
            value = self._evaluate(node, player, belief_state)

            # Backpropagation
            self._backup(search_path, value, player)

        # 結果を返す
        policy = root.get_policy()
        value = root.q_value

        return policy, value

    def _expand(self, node: NNGuidedMCTSNode, belief_state: ItemBeliefState):
        """ノードを展開"""
        if node.state.winner() is not None:
            node.is_expanded = True
            return

        available = node.state.available_commands(node.player)
        if not available:
            node.is_expanded = True
            return

        # NNでPolicyを取得
        if self.config.use_nn_policy:
            nn_policy = self._get_nn_policy(node.state, node.player, belief_state)
        else:
            # 均等分布
            nn_policy = {cmd: 1.0 / len(available) for cmd in available}

        # 子ノード作成
        for action in available:
            prior = nn_policy.get(action, 1.0 / len(available))

            # 仮説をサンプリングして次状態を作成
            child_state = self._create_child_state(node.state, node.player, action, belief_state)

            child = NNGuidedMCTSNode(
                state=child_state,
                player=node.player,
                parent=node,
                action=action,
                prior=prior,
            )
            node.children[action] = child

        node.is_expanded = True

    def _create_child_state(
        self,
        state: Battle,
        player: int,
        action: int,
        belief_state: ItemBeliefState,
    ) -> Battle:
        """子状態を作成（相手の行動はランダム）"""
        child_state = deepcopy(state)

        # 相手の行動をランダムに選択
        opp = 1 - player
        opp_actions = child_state.available_commands(opp)
        opp_action = random.choice(opp_actions) if opp_actions else Battle.SKIP

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
        belief_state: ItemBeliefState,
    ) -> dict[int, float]:
        """NNからPolicy分布を取得"""
        # TurnRecordを作成してエンコード
        turn_record = self._create_turn_record(state, player, belief_state)
        features = self.encoder.encode_flat(turn_record).unsqueeze(0).to(self.device)

        # 有効な行動のマスク
        available = state.available_commands(player)
        action_mask = torch.zeros(1, self.model.num_actions, device=self.device)

        for cmd in available:
            action_str = self._cmd_to_action_str(state, player, cmd)
            if action_str in self.action_vocab:
                action_id = self.action_vocab[action_str]
                if action_id < self.model.num_actions:
                    action_mask[0, action_id] = 1.0

        # NNで推論
        with torch.no_grad():
            policy, _ = self.model(features, action_mask)

        # コマンドIDに変換
        result = {}
        for cmd in available:
            action_str = self._cmd_to_action_str(state, player, cmd)
            if action_str in self.action_vocab:
                action_id = self.action_vocab[action_str]
                if action_id < self.model.num_actions:
                    result[cmd] = policy[0, action_id].item()
                else:
                    result[cmd] = 1.0 / len(available)
            else:
                result[cmd] = 1.0 / len(available)

        # 正規化
        total = sum(result.values())
        if total > 0:
            result = {k: v / total for k, v in result.items()}

        return result

    def _evaluate(
        self,
        node: NNGuidedMCTSNode,
        root_player: int,
        belief_state: ItemBeliefState,
    ) -> float:
        """ノードを評価"""
        # 勝敗が決まっていれば確定値
        winner = node.state.winner()
        if winner is not None:
            return 1.0 if winner == root_player else 0.0

        if self.config.use_nn_value:
            # NNで評価
            nn_value = self._get_nn_value(node.state, root_player, belief_state)

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

    def _get_nn_value(
        self,
        state: Battle,
        player: int,
        belief_state: ItemBeliefState,
    ) -> float:
        """NNからValue（勝率）を取得"""
        turn_record = self._create_turn_record(state, player, belief_state)
        features = self.encoder.encode_flat(turn_record).unsqueeze(0).to(self.device)

        with torch.no_grad():
            _, value = self.model(features)

        return value[0, 0].item()

    def _rollout(self, state: Battle, player: int) -> float:
        """ランダムプレイアウト"""
        sim_state = deepcopy(state)

        # 最大ターン数を制限
        max_turns = 50
        for _ in range(max_turns):
            if sim_state.winner() is not None:
                break

            moves0 = sim_state.available_commands(0)
            moves1 = sim_state.available_commands(1)
            cmd0 = random.choice(moves0) if moves0 else Battle.SKIP
            cmd1 = random.choice(moves1) if moves1 else Battle.SKIP
            sim_state.proceed(commands=[cmd0, cmd1])

        return _calculate_battle_score(sim_state, player)

    def _backup(
        self,
        search_path: list[NNGuidedMCTSNode],
        value: float,
        root_player: int,
    ):
        """バックプロパゲーション"""
        for node in reversed(search_path):
            node.visit_count += 1
            # ノードのプレイヤー視点の価値に変換
            node_value = value if node.player == root_player else 1 - value
            node.total_value += node_value

    def _create_turn_record(
        self,
        state: Battle,
        player: int,
        belief_state: ItemBeliefState,
    ) -> TurnRecord:
        """Battle状態からTurnRecordを作成"""
        from src.hypothesis.selfplay import (
            FieldCondition,
            PokemonState,
            TurnRecord,
        )

        def create_pokemon_state(pokemon) -> PokemonState:
            if pokemon is None:
                return PokemonState(
                    name="", hp=0, max_hp=0, hp_ratio=0.0, ailment="",
                    rank=[0] * 8, types=[], ability="", item="", moves=[],
                    terastallized=False, tera_type=""
                )
            max_hp = pokemon.status[0] if pokemon.status[0] > 0 else 1
            return PokemonState(
                name=pokemon.name,
                hp=pokemon.hp,
                max_hp=max_hp,
                hp_ratio=pokemon.hp / max_hp,
                ailment=getattr(pokemon, "ailment", ""),
                rank=list(getattr(pokemon, "rank", [0] * 8)),
                types=list(getattr(pokemon, "types", [])),
                ability=getattr(pokemon, "ability", ""),
                item=getattr(pokemon, "item", ""),
                moves=list(getattr(pokemon, "moves", [])),
                terastallized=getattr(pokemon, "terastal", False),
                tera_type=getattr(pokemon, "Ttype", ""),
            )

        opp = 1 - player
        cond = state.condition

        field = FieldCondition(
            sunny=cond.get("sunny", 0),
            rainy=cond.get("rainy", 0),
            snow=cond.get("snow", 0),
            sandstorm=cond.get("sandstorm", 0),
            electric_field=cond.get("elecfield", 0),
            grass_field=cond.get("glassfield", 0),
            psychic_field=cond.get("psycofield", 0),
            mist_field=cond.get("mistfield", 0),
            gravity=cond.get("gravity", 0),
            trick_room=cond.get("trickroom", 0),
            reflector=list(cond.get("reflector", [0, 0])),
            light_screen=list(cond.get("lightwall", [0, 0])),
            tailwind=list(cond.get("oikaze", [0, 0])),
            safeguard=list(cond.get("safeguard", [0, 0])),
            mist=list(cond.get("whitemist", [0, 0])),
            spikes=list(cond.get("makibishi", [0, 0])),
            toxic_spikes=list(cond.get("dokubishi", [0, 0])),
            stealth_rock=list(cond.get("stealthrock", [0, 0])),
            sticky_web=list(cond.get("nebanet", [0, 0])),
        )

        # 持ち物信念
        opp_team = [p.name for p in state.selected[opp]]
        item_beliefs = {name: belief_state.get_belief(name) for name in opp_team}

        my_bench = [
            create_pokemon_state(p)
            for p in state.selected[player]
            if p != state.pokemon[player]
        ]
        opp_bench = [
            create_pokemon_state(p)
            for p in state.selected[opp]
            if p != state.pokemon[opp]
        ]

        return TurnRecord(
            turn=0,
            player=player,
            my_pokemon=create_pokemon_state(state.pokemon[player]),
            my_bench=my_bench,
            opp_pokemon=create_pokemon_state(state.pokemon[opp]),
            opp_bench=opp_bench,
            field=field,
            item_beliefs=item_beliefs,
            policy={},
            value=0.0,
            action="",
            action_id=-1,
        )

    def _cmd_to_action_str(self, state: Battle, player: int, cmd: int) -> str:
        """コマンドIDを行動文字列に変換"""
        if cmd < 0:
            return "SKIP"
        elif cmd < 4:
            pokemon = state.pokemon[player]
            if pokemon and cmd < len(pokemon.moves):
                return f"MOVE:{pokemon.moves[cmd]}"
            return f"MOVE:{cmd}"
        elif cmd >= 20 and cmd < 30:
            idx = cmd - 20
            if idx < len(state.selected[player]):
                return f"SWITCH:{state.selected[player][idx].name}"
            return f"SWITCH:{idx}"
        elif cmd == 30:
            return "STRUGGLE"
        else:
            return f"CMD:{cmd}"

    def get_best_action(
        self,
        battle: Battle,
        player: int,
        belief_state: ItemBeliefState,
    ) -> int:
        """最良の行動を取得"""
        policy, _ = self.search(battle, player, belief_state)

        if not policy:
            available = battle.available_commands(player)
            return random.choice(available) if available else Battle.SKIP

        return max(policy.items(), key=lambda x: x[1])[0]
