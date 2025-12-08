"""
Team Selection Encoder

チーム選出用のエンコーダー。
自分の6匹と相手の6匹のポケモン情報をテンソルに変換する。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F


@dataclass
class TeamSelectionEncodingConfig:
    """チーム選出エンコーディング設定"""

    max_pokemon_id: int = 1500
    max_move_id: int = 1000
    max_item_id: int = 500
    max_ability_id: int = 400
    max_type_id: int = 20

    max_moves: int = 4
    team_size: int = 6  # 選出前のチームサイズ


class TeamSelectionEncoder:
    """
    チーム選出用エンコーダー

    ポケモンのJSON形式データをテンソルに変換する。
    battle開始前の選出段階で使用。
    """

    def __init__(
        self,
        config: Optional[TeamSelectionEncodingConfig] = None,
        pokemon_to_id: Optional[dict[str, int]] = None,
        move_to_id: Optional[dict[str, int]] = None,
        item_to_id: Optional[dict[str, int]] = None,
        ability_to_id: Optional[dict[str, int]] = None,
        type_to_id: Optional[dict[str, int]] = None,
    ):
        self.config = config or TeamSelectionEncodingConfig()

        self.pokemon_to_id = pokemon_to_id or {}
        self.move_to_id = move_to_id or {}
        self.item_to_id = item_to_id or {}
        self.ability_to_id = ability_to_id or {}
        self.type_to_id = type_to_id or self._default_type_to_id()

        self._next_pokemon_id = len(self.pokemon_to_id) + 1
        self._next_move_id = len(self.move_to_id) + 1
        self._next_item_id = len(self.item_to_id) + 1
        self._next_ability_id = len(self.ability_to_id) + 1

    def _default_type_to_id(self) -> dict[str, int]:
        """デフォルトのタイプID辞書"""
        types = [
            "ノーマル", "ほのお", "みず", "でんき", "くさ", "こおり",
            "かくとう", "どく", "じめん", "ひこう", "エスパー", "むし",
            "いわ", "ゴースト", "ドラゴン", "あく", "はがね", "フェアリー",
        ]
        return {t: i + 1 for i, t in enumerate(types)}

    def _get_pokemon_id(self, name: str) -> int:
        if not name:
            return 0
        if name not in self.pokemon_to_id:
            self.pokemon_to_id[name] = self._next_pokemon_id
            self._next_pokemon_id += 1
        return self.pokemon_to_id[name]

    def _get_move_id(self, name: str) -> int:
        if not name:
            return 0
        if name not in self.move_to_id:
            self.move_to_id[name] = self._next_move_id
            self._next_move_id += 1
        return self.move_to_id[name]

    def _get_item_id(self, name: str) -> int:
        if not name:
            return 0
        if name not in self.item_to_id:
            self.item_to_id[name] = self._next_item_id
            self._next_item_id += 1
        return self.item_to_id[name]

    def _get_ability_id(self, name: str) -> int:
        if not name:
            return 0
        if name not in self.ability_to_id:
            self.ability_to_id[name] = self._next_ability_id
            self._next_ability_id += 1
        return self.ability_to_id[name]

    def _get_type_id(self, name: str) -> int:
        if not name:
            return 0
        return self.type_to_id.get(name, 0)

    def encode_pokemon_for_selection(self, pokemon_data: dict) -> torch.Tensor:
        """
        選出用にポケモンをエンコード

        Args:
            pokemon_data: JSONから読み込んだポケモンデータ
                {name, item, nature, ability, Ttype, moves, effort}

        Returns:
            [pokemon_feature_dim] float tensor
        """
        features = []

        # ポケモン種族ID
        pokemon_id = self._get_pokemon_id(pokemon_data.get("name", ""))
        features.append(float(pokemon_id))

        # 持ち物ID
        item_id = self._get_item_id(pokemon_data.get("item", ""))
        features.append(float(item_id))

        # 特性ID
        ability_id = self._get_ability_id(pokemon_data.get("ability", ""))
        features.append(float(ability_id))

        # テラスタイプID
        tera_type_id = self._get_type_id(pokemon_data.get("Ttype", ""))
        features.append(float(tera_type_id))

        # 技ID（4つ）
        moves = pokemon_data.get("moves", [])
        for i in range(self.config.max_moves):
            if i < len(moves):
                move_id = self._get_move_id(moves[i])
            else:
                move_id = 0
            features.append(float(move_id))

        # 努力値（6つ、正規化）
        effort = pokemon_data.get("effort", [0] * 6)
        for i in range(6):
            if i < len(effort):
                features.append(effort[i] / 252.0)
            else:
                features.append(0.0)

        # 性格（数値化は簡略化 - 将来的にはone-hotなど）
        # ここでは性格による補正を直接特徴量として使わず、IDのみ
        nature = pokemon_data.get("nature", "まじめ")
        nature_id = hash(nature) % 25 + 1  # 簡易的なID化
        features.append(float(nature_id))

        return torch.tensor(features, dtype=torch.float32)

    def encode_team(self, team: list[dict]) -> torch.Tensor:
        """
        チーム全体（6匹）をエンコード

        Args:
            team: 6匹のポケモンデータのリスト

        Returns:
            [6, pokemon_feature_dim] float tensor
        """
        pokemon_tensors = []
        for i in range(self.config.team_size):
            if i < len(team):
                tensor = self.encode_pokemon_for_selection(team[i])
            else:
                # パディング
                tensor = torch.zeros(self.get_pokemon_feature_dim())
            pokemon_tensors.append(tensor)

        return torch.stack(pokemon_tensors)

    def encode_matchup(
        self, my_team: list[dict], opp_team: list[dict]
    ) -> dict[str, torch.Tensor]:
        """
        マッチアップ全体をエンコード

        Args:
            my_team: 自分のチーム（6匹）
            opp_team: 相手のチーム（6匹）

        Returns:
            {
                "my_team": [6, pokemon_feature_dim],
                "opp_team": [6, pokemon_feature_dim],
            }
        """
        return {
            "my_team": self.encode_team(my_team),
            "opp_team": self.encode_team(opp_team),
        }

    def get_pokemon_feature_dim(self) -> int:
        """ポケモン1体の特徴量次元"""
        # pokemon_id: 1
        # item_id: 1
        # ability_id: 1
        # tera_type_id: 1
        # moves: 4
        # effort: 6
        # nature_id: 1
        return 15

    def save(self, path: str | Path) -> None:
        """エンコーダーの状態を保存"""
        state = {
            "config": self.config.__dict__,
            "pokemon_to_id": self.pokemon_to_id,
            "move_to_id": self.move_to_id,
            "item_to_id": self.item_to_id,
            "ability_to_id": self.ability_to_id,
            "type_to_id": self.type_to_id,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "TeamSelectionEncoder":
        """エンコーダーの状態を読み込み"""
        with open(path, "r", encoding="utf-8") as f:
            state = json.load(f)

        config = TeamSelectionEncodingConfig(**state["config"])
        return cls(
            config=config,
            pokemon_to_id=state["pokemon_to_id"],
            move_to_id=state["move_to_id"],
            item_to_id=state["item_to_id"],
            ability_to_id=state["ability_to_id"],
            type_to_id=state["type_to_id"],
        )
