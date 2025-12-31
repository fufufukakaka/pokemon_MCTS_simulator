"""
Decision Transformer AI サービス

RebelAI と同じインターフェースを提供する UI 統合用ラッパー。
"""

from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from src.pokemon_battle_sim.battle import Battle

from .config import PokemonBattleTransformerConfig
from .data_generator import _battle_to_field_state, _get_turn_state, _pokemon_to_state
from .dataset import FieldState, PokemonState, TurnState
from .model import PokemonBattleTransformer, load_model
from .tokenizer import BattleSequenceTokenizer

logger = logging.getLogger(__name__)


@dataclass
class DecisionTransformerAIConfig:
    """Decision Transformer AI 設定"""

    checkpoint_path: str
    device: str = "cpu"
    target_return: float = 1.0  # 勝ちを目指す
    temperature: float = 0.5  # サンプリング温度
    deterministic: bool = True  # 決定的行動選択
    surrender_threshold: float = 0.05  # この勝率以下で降参を検討


@dataclass
class BattleContext:
    """
    バトル履歴を追跡するコンテキスト

    Decision Transformerはシーケンスモデルなので、
    過去のターン履歴を保持する必要がある。
    """

    my_team: list[str] = field(default_factory=list)
    opp_team: list[str] = field(default_factory=list)
    selection: list[int] = field(default_factory=list)
    lead_idx: int = 0
    turns: list[TurnState] = field(default_factory=list)
    actions: list[int] = field(default_factory=list)


class DecisionTransformerAI:
    """
    Decision Transformer AIインスタンス

    RebelAI と同じインターフェースを提供する。
    """

    def __init__(self, config: DecisionTransformerAIConfig):
        self.config = config
        self.device = torch.device(config.device)
        self.checkpoint_path = Path(config.checkpoint_path)

        # モデルとトークナイザーをロード
        logger.info(f"Loading Decision Transformer from {self.checkpoint_path}")
        self.model = self._load_model()
        self.tokenizer = self._load_tokenizer()

        # バトルコンテキスト（プレイヤーごと）
        self.contexts: dict[int, BattleContext] = {}

        logger.info("Decision Transformer AI initialized successfully")

    def _load_model(self) -> PokemonBattleTransformer:
        """モデルをロード"""
        return load_model(
            checkpoint_path=str(self.checkpoint_path),
            device=self.config.device,
        )

    def _load_tokenizer(self) -> BattleSequenceTokenizer:
        """トークナイザーをロード"""
        tokenizer_path = self.checkpoint_path / "tokenizer"
        if tokenizer_path.exists():
            return BattleSequenceTokenizer.load(tokenizer_path, self.model.config)

        # フォールバック：新規作成
        return BattleSequenceTokenizer(self.model.config)

    def reset(self) -> None:
        """状態をリセット"""
        self.contexts.clear()

    def get_selection(
        self,
        my_team_names: list[str],
        opponent_team_names: list[str],
        deterministic: bool | None = None,
    ) -> list[int]:
        """
        チーム選出を決定

        Args:
            my_team_names: 自チームのポケモン名リスト (6匹)
            opponent_team_names: 相手チームのポケモン名リスト (6匹)
            deterministic: 決定的選択にするか

        Returns:
            選出順序のリスト（先頭が先発、3匹）
        """
        if deterministic is None:
            deterministic = self.config.deterministic

        try:
            selected, lead_idx = self.model.get_selection(
                my_team=my_team_names,
                opp_team=opponent_team_names,
                tokenizer=self.tokenizer,
                target_return=self.config.target_return,
                deterministic=deterministic,
                temperature=self.config.temperature,
            )

            # 先発を先頭にして返す
            result = [selected[lead_idx]] + [
                s for i, s in enumerate(selected) if i != lead_idx
            ]

            # コンテキストを初期化
            for player in [0, 1]:
                self.contexts[player] = BattleContext(
                    my_team=my_team_names if player == 0 else opponent_team_names,
                    opp_team=opponent_team_names if player == 0 else my_team_names,
                    selection=selected,
                    lead_idx=lead_idx,
                )

            return result

        except Exception as e:
            logger.error(f"Selection failed: {e}")
            # フォールバック：ランダム
            indices = list(range(len(my_team_names)))
            random.shuffle(indices)
            return indices[:3]

    def should_surrender(self, battle: Battle, player: int) -> bool:
        """
        AIが降参すべきか判定

        Args:
            battle: Battleインスタンス
            player: AIのプレイヤー番号

        Returns:
            True: 降参すべき, False: 続行
        """
        try:
            # 簡易チェック：残りポケモン
            my_remaining = sum(
                1
                for p in battle.selected[player]
                if p is not None and p.hp > 0
            )
            opp_remaining = sum(
                1
                for p in battle.selected[1 - player]
                if p is not None and p.hp > 0
            )

            if my_remaining == 0:
                return True

            if my_remaining == 1 and opp_remaining >= 3:
                # 1匹 vs 3匹以上、勝率推定してみる
                analysis = self.get_analysis(battle, player)
                if analysis["available"] and analysis["value"] < self.config.surrender_threshold:
                    return True

            return False

        except Exception as e:
            logger.error(f"Surrender check failed: {e}")
            return False

    def get_battle_command(self, battle: Battle, player: int) -> int:
        """
        バトルフェーズで行動を選択

        Args:
            battle: Battleインスタンス
            player: プレイヤー番号（0 or 1）

        Returns:
            選択したコマンド
        """
        # 利用可能なコマンドを取得
        available = battle.available_commands(player, phase="battle")

        if not available or available == [Battle.NO_COMMAND]:
            return Battle.SKIP

        if len(available) == 1:
            return available[0]

        try:
            # コンテキストを更新
            context = self._get_or_create_context(battle, player)

            # 行動マスクを作成
            action_mask = self._create_action_mask(available)

            # バトル状態をエンコード
            turn_state = self._extract_turn_state(battle, player)
            encoded = self._encode_context(context, turn_state)

            # 行動選択
            action_id, win_prob = self.model.get_action(
                context=encoded,
                action_mask=action_mask,
                deterministic=self.config.deterministic,
                temperature=self.config.temperature,
            )

            # 有効な行動かチェック
            if action_id in available:
                # コンテキストを更新
                context.turns.append(turn_state)
                context.actions.append(action_id)
                return action_id
            else:
                # フォールバック：利用可能な行動から選択
                logger.warning(
                    f"Model predicted unavailable action {action_id}, "
                    f"available: {available}"
                )
                return random.choice(available)

        except Exception as e:
            logger.error(f"Decision Transformer action selection failed: {e}")
            return random.choice(available)

    def get_change_command(self, battle: Battle, player: int) -> int:
        """
        交代フェーズで行動を選択

        Args:
            battle: Battleインスタンス
            player: プレイヤー番号

        Returns:
            選択したコマンド
        """
        available = battle.available_commands(player, phase="change")

        if not available or available == [Battle.NO_COMMAND]:
            return Battle.SKIP

        if len(available) == 1:
            return available[0]

        try:
            # 交代先をモデルで選択
            context = self._get_or_create_context(battle, player)
            action_mask = self._create_action_mask(available)
            turn_state = self._extract_turn_state(battle, player)
            encoded = self._encode_context(context, turn_state)

            action_id, _ = self.model.get_action(
                context=encoded,
                action_mask=action_mask,
                deterministic=True,  # 交代は常に決定的
                temperature=self.config.temperature,
            )

            if action_id in available:
                return action_id
            else:
                # フォールバック：HP比率が高いポケモンを優先
                return self._select_best_switch(battle, player, available)

        except Exception as e:
            logger.error(f"Change command selection failed: {e}")
            return self._select_best_switch(battle, player, available)

    def get_analysis(self, battle: Battle, player: int) -> dict[str, Any]:
        """
        現在の戦況分析を取得

        Args:
            battle: Battleインスタンス
            player: プレイヤー番号（0 or 1）

        Returns:
            分析結果（戦略分布、推定勝率など）
        """
        analysis: dict[str, Any] = {
            "available": False,
            "policy": {},
            "value": 0.5,
            "action_names": {},
        }

        try:
            # 利用可能なコマンドを取得
            available = battle.available_commands(player, phase="battle")
            if not available or available == [Battle.NO_COMMAND]:
                return analysis

            # コンテキストを取得
            context = self._get_or_create_context(battle, player)

            # 行動マスクを作成
            action_mask = self._create_action_mask(available)

            # バトル状態をエンコード
            turn_state = self._extract_turn_state(battle, player)
            encoded = self._encode_context(context, turn_state)

            # ポリシーと価値を取得
            policy, value = self.model.get_policy_and_value(
                context=encoded,
                action_mask=action_mask,
            )

            # 行動名のマッピング
            action_names = self._get_action_names(battle, player, policy.keys())

            analysis = {
                "available": True,
                "policy": policy,
                "value": value,
                "action_names": action_names,
            }

        except Exception as e:
            logger.error(f"Analysis failed: {e}")

        return analysis

    def _get_or_create_context(
        self, battle: Battle, player: int
    ) -> BattleContext:
        """コンテキストを取得または作成"""
        if player not in self.contexts:
            # 新規作成
            my_team = [p.name for p in battle.selected[player] if p]
            opp_team = [p.name for p in battle.selected[1 - player] if p]
            self.contexts[player] = BattleContext(
                my_team=my_team,
                opp_team=opp_team,
            )
        return self.contexts[player]

    def _create_action_mask(self, available: list[int]) -> torch.Tensor:
        """行動マスクを作成"""
        mask = torch.zeros(self.model.config.num_action_outputs)
        for cmd in available:
            if 0 <= cmd < self.model.config.num_action_outputs:
                mask[cmd] = 1.0
        return mask

    def _extract_turn_state(self, battle: Battle, player: int) -> TurnState:
        """バトルからTurnStateを抽出（data_generatorの関数を再利用）"""
        return _get_turn_state(battle, player)

    def _encode_context(
        self, context: BattleContext, current_turn: TurnState
    ) -> dict[str, torch.Tensor]:
        """コンテキストをトークン化"""
        # チームプレビュー
        encoded = self.tokenizer.encode_team_preview(
            my_team=context.my_team,
            opp_team=context.opp_team,
            rtg=self.config.target_return,
        )

        # 選出があれば追加
        if context.selection:
            selection_encoded = self.tokenizer.encode_selection(
                selection=context.selection,
                lead_idx=context.lead_idx,
            )
            encoded = self._concat_encoded(encoded, selection_encoded)

        # 過去のターン履歴を追加
        for turn_state, action in zip(context.turns, context.actions):
            turn_encoded = self.tokenizer.encode_turn_state(turn_state)
            encoded = self._concat_encoded(encoded, turn_encoded)

            action_encoded = self.tokenizer.encode_action(action)
            encoded = self._concat_encoded(encoded, action_encoded)

        # 現在のターン状態を追加
        current_encoded = self.tokenizer.encode_turn_state(current_turn)
        encoded = self._concat_encoded(encoded, current_encoded)

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

    def _select_best_switch(
        self, battle: Battle, player: int, available: list[int]
    ) -> int:
        """HP比率が高いポケモンへの交代を選択"""
        best_switch = available[0]
        best_hp = 0.0

        for cmd in available:
            if cmd >= 20:
                idx = cmd - 20
                if idx < len(battle.selected[player]):
                    pokemon = battle.selected[player][idx]
                    if pokemon and pokemon.status[0] > 0:
                        hp_ratio = pokemon.hp / pokemon.status[0]
                        if hp_ratio > best_hp:
                            best_hp = hp_ratio
                            best_switch = cmd

        return best_switch

    def _get_action_names(
        self, battle: Battle, player: int, actions: list[int] | set[int]
    ) -> dict[int, str]:
        """行動IDから行動名へのマッピングを作成"""
        action_names = {}
        pokemon = battle.pokemon[player]

        if pokemon is None:
            return action_names

        for cmd in actions:
            if 0 <= cmd <= 3:
                # 通常技
                if cmd < len(pokemon.moves):
                    action_names[cmd] = pokemon.moves[cmd]
                else:
                    action_names[cmd] = f"技{cmd + 1}"
            elif 10 <= cmd <= 13:
                # テラスタル技
                move_idx = cmd - 10
                if move_idx < len(pokemon.moves):
                    action_names[cmd] = f"テラス+{pokemon.moves[move_idx]}"
                else:
                    action_names[cmd] = f"テラス+技{move_idx + 1}"
            elif 20 <= cmd <= 25:
                # 交代
                bench_idx = cmd - 20
                if bench_idx < len(battle.selected[player]):
                    bench_pokemon = battle.selected[player][bench_idx]
                    if bench_pokemon:
                        action_names[cmd] = f"交代→{bench_pokemon.name}"
                    else:
                        action_names[cmd] = f"交代→{bench_idx + 1}番"
                else:
                    action_names[cmd] = f"交代{cmd}"
            elif cmd == Battle.STRUGGLE:
                action_names[cmd] = "わるあがき"
            else:
                action_names[cmd] = f"行動{cmd}"

        return action_names


# グローバルキャッシュ
_dt_ai_cache: dict[str, DecisionTransformerAI] = {}


def load_decision_transformer_ai(
    checkpoint_path: str,
    device: str = "cpu",
    target_return: float = 1.0,
    temperature: float = 0.5,
    deterministic: bool = True,
) -> DecisionTransformerAI:
    """
    Decision Transformer AIをロード（キャッシュ付き）

    Args:
        checkpoint_path: チェックポイントのパス
        device: デバイス
        target_return: 目標リターン（1.0 = 勝利を目指す）
        temperature: サンプリング温度
        deterministic: 決定的行動選択

    Returns:
        DecisionTransformerAI インスタンス
    """
    cache_key = f"{checkpoint_path}:{device}:{target_return}:{temperature}:{deterministic}"

    if cache_key not in _dt_ai_cache:
        config = DecisionTransformerAIConfig(
            checkpoint_path=checkpoint_path,
            device=device,
            target_return=target_return,
            temperature=temperature,
            deterministic=deterministic,
        )
        _dt_ai_cache[cache_key] = DecisionTransformerAI(config)

    return _dt_ai_cache[cache_key]


def clear_dt_ai_cache() -> None:
    """キャッシュをクリア"""
    _dt_ai_cache.clear()
