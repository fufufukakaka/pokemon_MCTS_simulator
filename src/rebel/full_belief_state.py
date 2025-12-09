"""
完全な信念状態の管理

チーム構成（選出・先発）の不確実性と、
各ポケモンの型（技・持ち物・テラス等）の不確実性を統合した信念システム。

レベル1: TeamCompositionBelief - どの3匹が選出され、誰が先発か
レベル2: PokemonBeliefState - 選出された各ポケモンの型仮説
"""

from __future__ import annotations

import random
from copy import deepcopy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

from src.hypothesis.pokemon_usage_database import PokemonUsageDatabase

from .belief_state import (
    Observation,
    ObservationType,
    PokemonBeliefState,
    PokemonTypeHypothesis,
)
from .team_composition_belief import TeamCompositionBelief, TeamCompositionHypothesis

if TYPE_CHECKING:
    from src.policy_value_network.team_selector import NNTeamSelector
    from src.selection_bert.selection_belief import SelectionBeliefPredictor


@dataclass
class SampledWorld:
    """
    サンプリングされた1つの「世界」

    チーム構成と各ポケモンの型を含む完全な仮説。
    """

    composition: TeamCompositionHypothesis  # 選出・先発
    types: dict[str, PokemonTypeHypothesis]  # {pokemon_name: type_hypothesis}


class FullBeliefState:
    """
    完全な信念状態

    チーム構成の不確実性と型の不確実性を統合管理する。

    使用フロー:
    1. チームプレビューで初期化（6匹の情報）
    2. TeamSelectionNetworkで選出・先発の事前分布を設定
    3. バトル開始時に先発が判明 → 更新
    4. 交代で新しいポケモンが判明 → 更新
    5. 技使用、持ち物判明等 → 型信念を更新
    """

    def __init__(
        self,
        team_preview_names: list[str],
        team_preview_data: list[dict],
        usage_db: PokemonUsageDatabase,
        selector: Optional["NNTeamSelector"] = None,
        my_team_data: Optional[list[dict]] = None,
        my_team_names: Optional[list[str]] = None,
        selection_bert_predictor: Optional["SelectionBeliefPredictor"] = None,
        max_hypotheses_per_pokemon: int = 50,
    ):
        """
        Args:
            team_preview_names: 相手の6匹の名前
            team_preview_data: 相手の6匹のデータ
            usage_db: 使用率データベース
            selector: TeamSelectionNetwork（Noneなら一様分布）
            my_team_data: 自分の6匹のデータ（selector使用時に必要）
            my_team_names: 自分の6匹の名前（selection_bert_predictor使用時に必要）
            selection_bert_predictor: SelectionBeliefPredictor（selection_bertを使う場合）
            max_hypotheses_per_pokemon: ポケモンあたりの最大型仮説数
        """
        self.team_preview_names = team_preview_names
        self.team_preview_data = team_preview_data
        self.usage_db = usage_db
        self.max_hypotheses = max_hypotheses_per_pokemon

        # インデックスと名前の対応
        self.name_to_index = {name: i for i, name in enumerate(team_preview_names)}

        # チーム構成の信念
        # Selection BERTを優先、次にNNTeamSelector、どちらもなければ一様分布
        if selection_bert_predictor is not None and my_team_names is not None:
            self.composition_belief = TeamCompositionBelief.from_selection_bert(
                team_preview_names=team_preview_names,
                my_team_names=my_team_names,
                predictor=selection_bert_predictor,
            )
        elif selector is not None and my_team_data is not None:
            self.composition_belief = TeamCompositionBelief.from_selection_network(
                team_preview_names=team_preview_names,
                team_preview_data=team_preview_data,
                my_team_data=my_team_data,
                selector=selector,
            )
        else:
            self.composition_belief = TeamCompositionBelief(
                team_preview_names=team_preview_names
            )

        # 各ポケモンの型信念（6匹全員分、遅延初期化）
        self._type_beliefs: dict[str, PokemonBeliefState] = {}

        # 観測履歴
        self.observation_history: list[Observation] = []

    def _get_or_create_type_belief(self, pokemon_name: str) -> PokemonBeliefState:
        """型信念を取得（なければ作成）"""
        if pokemon_name not in self._type_beliefs:
            # 単一ポケモンの信念を作成
            self._type_beliefs[pokemon_name] = PokemonBeliefState(
                opponent_pokemon_names=[pokemon_name],
                usage_db=self.usage_db,
                max_hypotheses_per_pokemon=self.max_hypotheses,
            )
        return self._type_beliefs[pokemon_name]

    def update_lead_revealed(self, lead_name: str) -> None:
        """
        先発が判明したときの更新

        Args:
            lead_name: 先発ポケモンの名前
        """
        if lead_name not in self.name_to_index:
            return

        lead_index = self.name_to_index[lead_name]
        self.composition_belief.update_lead_revealed(lead_index)

        # 先発の型信念を初期化
        self._get_or_create_type_belief(lead_name)

    def update_pokemon_revealed(self, pokemon_name: str) -> None:
        """
        新しいポケモンが判明したときの更新

        Args:
            pokemon_name: 判明したポケモンの名前
        """
        if pokemon_name not in self.name_to_index:
            return

        pokemon_index = self.name_to_index[pokemon_name]
        self.composition_belief.update_pokemon_revealed(pokemon_index)

        # 型信念を初期化
        self._get_or_create_type_belief(pokemon_name)

    def update(self, observation: Observation) -> None:
        """
        観測に基づいて信念を更新

        Args:
            observation: 観測イベント
        """
        self.observation_history.append(observation)
        pokemon_name = observation.pokemon_name

        # ポケモンが判明していなければ、まず判明させる
        if pokemon_name in self.name_to_index:
            idx = self.name_to_index[pokemon_name]
            if idx not in self.composition_belief.confirmed_selected:
                self.update_pokemon_revealed(pokemon_name)

        # 型信念を更新
        if pokemon_name in self._type_beliefs:
            self._type_beliefs[pokemon_name].update(observation)

    def get_selection_probability(self, pokemon_name: str) -> float:
        """ポケモンが選出されている確率を取得"""
        if pokemon_name not in self.name_to_index:
            return 0.0
        idx = self.name_to_index[pokemon_name]
        return self.composition_belief.get_selection_probability(idx)

    def get_lead_probability(self, pokemon_name: str) -> float:
        """ポケモンが先発である確率を取得"""
        if pokemon_name not in self.name_to_index:
            return 0.0
        idx = self.name_to_index[pokemon_name]
        return self.composition_belief.get_lead_probability(idx)

    def get_item_distribution(self, pokemon_name: str) -> dict[str, float]:
        """持ち物の周辺確率分布を取得"""
        belief = self._get_or_create_type_belief(pokemon_name)
        return belief.get_item_distribution(pokemon_name)

    def get_tera_distribution(self, pokemon_name: str) -> dict[str, float]:
        """テラスタイプの周辺確率分布を取得"""
        belief = self._get_or_create_type_belief(pokemon_name)
        return belief.get_tera_distribution(pokemon_name)

    def get_move_probability(self, pokemon_name: str, move: str) -> float:
        """特定の技を持っている確率を取得"""
        belief = self._get_or_create_type_belief(pokemon_name)
        return belief.get_move_probability(pokemon_name, move)

    def get_confirmed_selected_names(self) -> list[str]:
        """選出が確定したポケモンの名前リストを取得"""
        return self.composition_belief.get_selected_names()

    def sample_world(self) -> SampledWorld:
        """
        現在の信念から1つの「世界」をサンプリング

        Returns:
            チーム構成と各ポケモンの型を含む完全な仮説
        """
        # チーム構成をサンプリング
        composition = self.composition_belief.sample_composition()

        # 選出されたポケモンの型をサンプリング
        types = {}
        for idx in composition.selected_indices:
            name = self.team_preview_names[idx]
            belief = self._get_or_create_type_belief(name)
            world = belief.sample_world()
            if name in world:
                types[name] = world[name]

        return SampledWorld(composition=composition, types=types)

    def sample_worlds(self, n: int) -> list[SampledWorld]:
        """複数の世界をサンプリング"""
        return [self.sample_world() for _ in range(n)]

    def get_selected_pokemon_beliefs(self) -> dict[str, PokemonBeliefState]:
        """
        選出が確定したポケモンの型信念を取得

        Returns:
            {pokemon_name: belief} の辞書
        """
        result = {}
        for name in self.get_confirmed_selected_names():
            if name in self._type_beliefs:
                result[name] = self._type_beliefs[name]
        return result

    def to_pokemon_belief_state(self) -> Optional[PokemonBeliefState]:
        """
        選出が確定した3匹のPokemonBeliefStateを作成

        3匹確定していない場合はNoneを返す。
        ReBeL学習との互換性のため。
        """
        confirmed = self.get_confirmed_selected_names()
        if len(confirmed) != 3:
            return None

        # 新しいPokemonBeliefStateを作成
        combined = PokemonBeliefState(
            opponent_pokemon_names=confirmed,
            usage_db=self.usage_db,
            max_hypotheses_per_pokemon=self.max_hypotheses,
        )

        # 各ポケモンの信念をコピー
        for name in confirmed:
            if name in self._type_beliefs:
                src_belief = self._type_beliefs[name]
                if name in src_belief.beliefs:
                    combined.beliefs[name] = dict(src_belief.beliefs[name])
                combined.revealed_moves[name] = set(src_belief.revealed_moves.get(name, set()))
                combined.revealed_items[name] = src_belief.revealed_items.get(name)
                combined.revealed_tera[name] = src_belief.revealed_tera.get(name)

        return combined

    def copy(self) -> "FullBeliefState":
        """ディープコピー"""
        new_state = FullBeliefState.__new__(FullBeliefState)
        new_state.team_preview_names = list(self.team_preview_names)
        new_state.team_preview_data = deepcopy(self.team_preview_data)
        new_state.usage_db = self.usage_db
        new_state.max_hypotheses = self.max_hypotheses
        new_state.name_to_index = dict(self.name_to_index)
        new_state.composition_belief = self.composition_belief.copy()
        new_state._type_beliefs = {
            name: belief.copy() for name, belief in self._type_beliefs.items()
        }
        new_state.observation_history = list(self.observation_history)
        return new_state

    def summary(self) -> str:
        """信念状態の概要"""
        lines = ["FullBeliefState Summary", "=" * 60]

        # チーム構成の概要
        lines.append("\n【チーム構成】")
        lines.append(self.composition_belief.summary())

        # 型信念の概要
        lines.append("\n【型信念】")
        for name in self.get_confirmed_selected_names():
            if name in self._type_beliefs:
                lines.append(f"\n{name}:")
                belief = self._type_beliefs[name]

                # 持ち物
                item_dist = belief.get_item_distribution(name)
                if item_dist:
                    top_items = sorted(item_dist.items(), key=lambda x: -x[1])[:3]
                    items_str = ", ".join(f"{i}({p:.1%})" for i, p in top_items)
                    lines.append(f"  持ち物: {items_str}")

                # テラス
                tera_dist = belief.get_tera_distribution(name)
                if tera_dist:
                    top_tera = sorted(tera_dist.items(), key=lambda x: -x[1])[:3]
                    tera_str = ", ".join(f"{t}({p:.1%})" for t, p in top_tera)
                    lines.append(f"  テラス: {tera_str}")

                # 判明した技
                revealed = belief.revealed_moves.get(name, set())
                if revealed:
                    lines.append(f"  判明技: {', '.join(revealed)}")

        return "\n".join(lines)

    def __repr__(self) -> str:
        confirmed = len(self.composition_belief.confirmed_selected)
        type_beliefs = len(self._type_beliefs)
        return f"FullBeliefState(confirmed={confirmed}/3, type_beliefs={type_beliefs})"
