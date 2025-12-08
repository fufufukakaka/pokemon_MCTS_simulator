"""
チーム構成に対する信念状態の管理

相手の選出（6匹から3匹）と先発に対する不確実性を管理する。
TeamSelectionNetworkの出力を事前分布として使用し、
観測に基づいてベイズ更新を行う。
"""

from __future__ import annotations

import random
from copy import deepcopy
from dataclasses import dataclass, field
from itertools import combinations
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from src.policy_value_network.team_selector import NNTeamSelector


@dataclass
class TeamCompositionHypothesis:
    """
    相手のチーム構成仮説

    6匹から選出された3匹と、その中での先発を表す。
    """

    selected_indices: tuple[int, int, int]  # 選出された3匹のインデックス（0-5）
    lead_index: int  # 先発のインデックス（0-5、selected_indicesに含まれる）

    def __post_init__(self):
        assert len(self.selected_indices) == 3
        assert self.lead_index in self.selected_indices

    def __hash__(self):
        return hash((self.selected_indices, self.lead_index))

    def __eq__(self, other):
        if not isinstance(other, TeamCompositionHypothesis):
            return False
        return (
            self.selected_indices == other.selected_indices
            and self.lead_index == other.lead_index
        )


@dataclass
class TeamCompositionBelief:
    """
    相手のチーム構成に対する信念状態

    チームプレビューで見た6匹から、相手がどの3匹を選出し、
    誰を先発にしたかの確率分布を管理する。

    信念は以下の情報により更新される：
    - 相手の先発が判明（バトル開始時）
    - 相手が交代して新しいポケモンが判明
    - 3匹目が判明（残りは選出されていないことが確定）
    """

    # チームプレビューで見た6匹の名前
    team_preview_names: list[str] = field(default_factory=list)

    # 仮説と確率の辞書
    hypotheses: dict[TeamCompositionHypothesis, float] = field(default_factory=dict)

    # 確定情報
    confirmed_lead: Optional[int] = None  # 先発のインデックス（判明後）
    confirmed_selected: set[int] = field(default_factory=set)  # 選出が確定したインデックス
    confirmed_not_selected: set[int] = field(default_factory=set)  # 選出されていないことが確定

    def __post_init__(self):
        if not self.hypotheses and self.team_preview_names:
            # 初期化時に一様分布で仮説を生成
            self._initialize_uniform()

    def _initialize_uniform(self) -> None:
        """一様分布で初期化"""
        self.hypotheses = {}
        n = len(self.team_preview_names)
        if n < 3:
            return

        # 全ての選出組み合わせ × 先発の組み合わせ
        for selected in combinations(range(n), 3):
            for lead in selected:
                hypothesis = TeamCompositionHypothesis(
                    selected_indices=tuple(sorted(selected)),
                    lead_index=lead,
                )
                self.hypotheses[hypothesis] = 1.0

        self._normalize()

    @classmethod
    def from_selection_network(
        cls,
        team_preview_names: list[str],
        team_preview_data: list[dict],
        my_team_data: list[dict],
        selector: "NNTeamSelector",
    ) -> "TeamCompositionBelief":
        """
        TeamSelectionNetworkの出力を事前分布として信念を初期化

        Args:
            team_preview_names: 相手の6匹の名前
            team_preview_data: 相手の6匹のデータ（エンコード用）
            my_team_data: 自分の6匹のデータ（相手視点での予測に使用）
            selector: NNTeamSelector

        Returns:
            初期化されたTeamCompositionBelief
        """
        belief = cls(team_preview_names=team_preview_names)

        # 相手視点での選出確率・先発確率を取得
        probs = selector.get_selection_and_lead_probs(
            my_team=team_preview_data,  # 相手のチーム
            opp_team=my_team_data,  # 自分のチーム（相手から見た相手）
        )

        selection_probs = probs["selection_probs"]
        lead_probs = probs["lead_probs"]

        # 仮説の確率を計算
        belief.hypotheses = {}
        n = len(team_preview_names)

        for selected in combinations(range(n), 3):
            # 選出確率（独立仮定で近似）
            # P(A, B, C selected) ≈ P(A) * P(B) * P(C) / Z
            selection_prob = 1.0
            for idx in selected:
                selection_prob *= selection_probs[idx] if idx < len(selection_probs) else 0.1

            for lead in selected:
                # 先発確率（選出された3匹の中での条件付き確率）
                lead_prob = lead_probs[lead] if lead < len(lead_probs) else 0.1

                # 先発確率を選出された3匹内で正規化
                lead_sum = sum(
                    lead_probs[i] if i < len(lead_probs) else 0.1
                    for i in selected
                )
                if lead_sum > 0:
                    normalized_lead_prob = lead_prob / lead_sum
                else:
                    normalized_lead_prob = 1.0 / 3.0

                hypothesis = TeamCompositionHypothesis(
                    selected_indices=tuple(sorted(selected)),
                    lead_index=lead,
                )
                belief.hypotheses[hypothesis] = selection_prob * normalized_lead_prob

        belief._normalize()
        return belief

    def _normalize(self) -> None:
        """確率を正規化"""
        total = sum(self.hypotheses.values())
        if total > 0:
            self.hypotheses = {h: p / total for h, p in self.hypotheses.items()}

    def update_lead_revealed(self, lead_index: int) -> None:
        """
        先発が判明したときの更新

        Args:
            lead_index: 先発のインデックス（0-5）
        """
        self.confirmed_lead = lead_index
        self.confirmed_selected.add(lead_index)

        # 先発が一致しない仮説を除外
        self.hypotheses = {
            h: p for h, p in self.hypotheses.items() if h.lead_index == lead_index
        }
        self._normalize()

    def update_pokemon_revealed(self, pokemon_index: int) -> None:
        """
        新しいポケモンが判明したときの更新（交代で出てきた等）

        Args:
            pokemon_index: 判明したポケモンのインデックス（0-5）
        """
        self.confirmed_selected.add(pokemon_index)

        # そのポケモンが選出されていない仮説を除外
        self.hypotheses = {
            h: p
            for h, p in self.hypotheses.items()
            if pokemon_index in h.selected_indices
        }
        self._normalize()

        # 3匹確定したら残りは選出されていない
        if len(self.confirmed_selected) == 3:
            all_indices = set(range(len(self.team_preview_names)))
            self.confirmed_not_selected = all_indices - self.confirmed_selected

    def get_selection_probability(self, pokemon_index: int) -> float:
        """
        指定したポケモンが選出されている確率を取得

        Args:
            pokemon_index: ポケモンのインデックス（0-5）

        Returns:
            選出確率
        """
        if pokemon_index in self.confirmed_selected:
            return 1.0
        if pokemon_index in self.confirmed_not_selected:
            return 0.0

        prob = sum(
            p
            for h, p in self.hypotheses.items()
            if pokemon_index in h.selected_indices
        )
        return prob

    def get_lead_probability(self, pokemon_index: int) -> float:
        """
        指定したポケモンが先発である確率を取得

        Args:
            pokemon_index: ポケモンのインデックス（0-5）

        Returns:
            先発確率
        """
        if self.confirmed_lead is not None:
            return 1.0 if pokemon_index == self.confirmed_lead else 0.0

        prob = sum(p for h, p in self.hypotheses.items() if h.lead_index == pokemon_index)
        return prob

    def get_selected_names(self) -> list[str]:
        """
        選出が確定したポケモンの名前リストを取得

        Returns:
            確定した選出ポケモンの名前
        """
        return [self.team_preview_names[i] for i in sorted(self.confirmed_selected)]

    def get_most_likely_composition(self) -> TeamCompositionHypothesis:
        """最も確率の高いチーム構成を取得"""
        if not self.hypotheses:
            raise ValueError("No hypotheses available")
        return max(self.hypotheses.items(), key=lambda x: x[1])[0]

    def sample_composition(self) -> TeamCompositionHypothesis:
        """現在の信念から1つのチーム構成をサンプリング"""
        if not self.hypotheses:
            raise ValueError("No hypotheses available")

        hypotheses = list(self.hypotheses.keys())
        probs = list(self.hypotheses.values())
        return random.choices(hypotheses, weights=probs, k=1)[0]

    def sample_compositions(self, n: int) -> list[TeamCompositionHypothesis]:
        """複数のチーム構成をサンプリング"""
        return [self.sample_composition() for _ in range(n)]

    def copy(self) -> "TeamCompositionBelief":
        """ディープコピー"""
        return TeamCompositionBelief(
            team_preview_names=list(self.team_preview_names),
            hypotheses=dict(self.hypotheses),
            confirmed_lead=self.confirmed_lead,
            confirmed_selected=set(self.confirmed_selected),
            confirmed_not_selected=set(self.confirmed_not_selected),
        )

    def summary(self) -> str:
        """信念状態の概要"""
        lines = ["TeamCompositionBelief Summary", "=" * 50]

        lines.append(f"\nチームプレビュー: {', '.join(self.team_preview_names)}")

        if self.confirmed_lead is not None:
            lines.append(f"確定先発: {self.team_preview_names[self.confirmed_lead]}")

        if self.confirmed_selected:
            selected_names = [self.team_preview_names[i] for i in self.confirmed_selected]
            lines.append(f"確定選出: {', '.join(selected_names)}")

        if self.confirmed_not_selected:
            not_selected_names = [
                self.team_preview_names[i] for i in self.confirmed_not_selected
            ]
            lines.append(f"非選出確定: {', '.join(not_selected_names)}")

        lines.append("\n各ポケモンの選出・先発確率:")
        for i, name in enumerate(self.team_preview_names):
            sel_prob = self.get_selection_probability(i)
            lead_prob = self.get_lead_probability(i)
            lines.append(f"  {name}: 選出={sel_prob:.1%}, 先発={lead_prob:.1%}")

        lines.append(f"\n仮説数: {len(self.hypotheses)}")

        return "\n".join(lines)

    def __repr__(self) -> str:
        confirmed = len(self.confirmed_selected)
        return f"TeamCompositionBelief(confirmed={confirmed}/3, hypotheses={len(self.hypotheses)})"
