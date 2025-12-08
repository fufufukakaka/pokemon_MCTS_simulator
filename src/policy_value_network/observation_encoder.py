"""
Observation Encoder

盤面情報（TurnRecord）をニューラルネットワークの入力テンソルに変換する。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F

from src.hypothesis.selfplay import FieldCondition, PokemonState, TurnRecord


@dataclass
class EncodingConfig:
    """エンコーディング設定"""

    # ポケモン関連
    max_pokemon_id: int = 1500  # ポケモン種族数の上限
    max_move_id: int = 1000  # 技数の上限
    max_item_id: int = 500  # 持ち物数の上限
    max_ability_id: int = 400  # 特性数の上限
    max_type_id: int = 20  # タイプ数

    # 埋め込み次元
    pokemon_embed_dim: int = 64
    move_embed_dim: int = 32
    item_embed_dim: int = 32
    ability_embed_dim: int = 32
    type_embed_dim: int = 16

    # その他
    max_moves: int = 4  # 1体あたりの技数上限
    max_team_size: int = 3  # チームサイズ（場1 + 控え2）
    num_belief_items: int = 10  # 信念状態で考慮する持ち物数


class ObservationEncoder:
    """
    盤面情報をテンソルに変換するエンコーダー

    入力: TurnRecord（盤面の詳細情報）
    出力: torch.Tensor（ニューラルネットワークの入力）
    """

    def __init__(
        self,
        config: Optional[EncodingConfig] = None,
        pokemon_to_id: Optional[dict[str, int]] = None,
        move_to_id: Optional[dict[str, int]] = None,
        item_to_id: Optional[dict[str, int]] = None,
        ability_to_id: Optional[dict[str, int]] = None,
        type_to_id: Optional[dict[str, int]] = None,
    ):
        self.config = config or EncodingConfig()

        # ID変換辞書（Noneの場合は動的に構築）
        self.pokemon_to_id = pokemon_to_id or {}
        self.move_to_id = move_to_id or {}
        self.item_to_id = item_to_id or {}
        self.ability_to_id = ability_to_id or {}
        self.type_to_id = type_to_id or self._default_type_to_id()

        # 未知のエントリ用カウンター
        self._next_pokemon_id = len(self.pokemon_to_id) + 1
        self._next_move_id = len(self.move_to_id) + 1
        self._next_item_id = len(self.item_to_id) + 1
        self._next_ability_id = len(self.ability_to_id) + 1

    def _default_type_to_id(self) -> dict[str, int]:
        """デフォルトのタイプID辞書"""
        types = [
            "ノーマル",
            "ほのお",
            "みず",
            "でんき",
            "くさ",
            "こおり",
            "かくとう",
            "どく",
            "じめん",
            "ひこう",
            "エスパー",
            "むし",
            "いわ",
            "ゴースト",
            "ドラゴン",
            "あく",
            "はがね",
            "フェアリー",
        ]
        return {t: i + 1 for i, t in enumerate(types)}

    def _get_pokemon_id(self, name: str) -> int:
        """ポケモン名からIDを取得（未知なら新規割り当て）"""
        if not name:
            return 0
        if name not in self.pokemon_to_id:
            self.pokemon_to_id[name] = self._next_pokemon_id
            self._next_pokemon_id += 1
        return self.pokemon_to_id[name]

    def _get_move_id(self, name: str) -> int:
        """技名からIDを取得"""
        if not name:
            return 0
        if name not in self.move_to_id:
            self.move_to_id[name] = self._next_move_id
            self._next_move_id += 1
        return self.move_to_id[name]

    def _get_item_id(self, name: str) -> int:
        """持ち物名からIDを取得"""
        if not name:
            return 0
        if name not in self.item_to_id:
            self.item_to_id[name] = self._next_item_id
            self._next_item_id += 1
        return self.item_to_id[name]

    def _get_ability_id(self, name: str) -> int:
        """特性名からIDを取得"""
        if not name:
            return 0
        if name not in self.ability_to_id:
            self.ability_to_id[name] = self._next_ability_id
            self._next_ability_id += 1
        return self.ability_to_id[name]

    def _get_type_id(self, name: str) -> int:
        """タイプ名からIDを取得"""
        if not name:
            return 0
        return self.type_to_id.get(name, 0)

    def encode_pokemon(self, pokemon: PokemonState) -> dict[str, torch.Tensor]:
        """
        ポケモンの状態をテンソルに変換

        Returns:
            dict with keys:
            - pokemon_id: [1] int
            - hp_ratio: [1] float
            - ailment: [7] one-hot (なし, どく, もうどく, やけど, まひ, ねむり, こおり)
            - rank: [8] float (正規化済み)
            - types: [2] int
            - ability_id: [1] int
            - item_id: [1] int
            - moves: [4] int
            - terastallized: [1] float
            - tera_type: [1] int
        """
        # ポケモンID
        pokemon_id = torch.tensor([self._get_pokemon_id(pokemon.name)], dtype=torch.long)

        # HP比率
        hp_ratio = torch.tensor([pokemon.hp_ratio], dtype=torch.float32)

        # 状態異常（one-hot）
        ailment_map = {
            "": 0,
            "どく": 1,
            "もうどく": 2,
            "やけど": 3,
            "まひ": 4,
            "ねむり": 5,
            "こおり": 6,
        }
        ailment_idx = ailment_map.get(pokemon.ailment, 0)
        ailment = F.one_hot(torch.tensor(ailment_idx), num_classes=7).float()

        # ランク変化（-6〜+6 を -1〜+1 に正規化）
        rank = torch.tensor(pokemon.rank, dtype=torch.float32) / 6.0

        # タイプ（最大2つ）
        type_ids = [self._get_type_id(t) for t in pokemon.types[:2]]
        while len(type_ids) < 2:
            type_ids.append(0)
        types = torch.tensor(type_ids, dtype=torch.long)

        # 特性
        ability_id = torch.tensor([self._get_ability_id(pokemon.ability)], dtype=torch.long)

        # 持ち物
        item_id = torch.tensor([self._get_item_id(pokemon.item)], dtype=torch.long)

        # 技（最大4つ）
        move_ids = [self._get_move_id(m) for m in pokemon.moves[: self.config.max_moves]]
        while len(move_ids) < self.config.max_moves:
            move_ids.append(0)
        moves = torch.tensor(move_ids, dtype=torch.long)

        # テラスタル
        terastallized = torch.tensor([1.0 if pokemon.terastallized else 0.0], dtype=torch.float32)
        tera_type = torch.tensor([self._get_type_id(pokemon.tera_type)], dtype=torch.long)

        return {
            "pokemon_id": pokemon_id,
            "hp_ratio": hp_ratio,
            "ailment": ailment,
            "rank": rank,
            "types": types,
            "ability_id": ability_id,
            "item_id": item_id,
            "moves": moves,
            "terastallized": terastallized,
            "tera_type": tera_type,
        }

    def encode_field(self, field: FieldCondition) -> torch.Tensor:
        """
        場の状態をテンソルに変換

        Returns:
            [24] float tensor:
            - 天候: [4] (sunny, rainy, snow, sandstorm) 残りターン/5で正規化
            - フィールド: [4] (electric, grass, psychic, mist) 残りターン/5で正規化
            - その他: [2] (gravity, trick_room) 残りターン/5で正規化
            - 壁（自分）: [2] (reflector, light_screen) 残りターン/5で正規化
            - 壁（相手）: [2]
            - おいかぜ: [2] (自分, 相手)
            - 設置技（自分側）: [4] (spikes/3, toxic_spikes/2, stealth_rock, sticky_web)
            - 設置技（相手側）: [4]
        """
        features = []

        # 天候（残りターン/5で正規化）
        features.extend(
            [
                field.sunny / 5.0,
                field.rainy / 5.0,
                field.snow / 5.0,
                field.sandstorm / 5.0,
            ]
        )

        # フィールド
        features.extend(
            [
                field.electric_field / 5.0,
                field.grass_field / 5.0,
                field.psychic_field / 5.0,
                field.mist_field / 5.0,
            ]
        )

        # その他
        features.extend([field.gravity / 5.0, field.trick_room / 5.0])

        # 壁（自分=index 0）
        features.extend([field.reflector[0] / 5.0, field.light_screen[0] / 5.0])
        # 壁（相手=index 1）
        features.extend([field.reflector[1] / 5.0, field.light_screen[1] / 5.0])

        # おいかぜ
        features.extend([field.tailwind[0] / 4.0, field.tailwind[1] / 4.0])

        # 設置技（自分側=相手が撒いた）
        features.extend(
            [
                field.spikes[0] / 3.0,
                field.toxic_spikes[0] / 2.0,
                float(field.stealth_rock[0]),
                float(field.sticky_web[0]),
            ]
        )

        # 設置技（相手側=自分が撒いた）
        features.extend(
            [
                field.spikes[1] / 3.0,
                field.toxic_spikes[1] / 2.0,
                float(field.stealth_rock[1]),
                float(field.sticky_web[1]),
            ]
        )

        return torch.tensor(features, dtype=torch.float32)

    def encode_item_beliefs(
        self, item_beliefs: dict[str, dict[str, float]]
    ) -> torch.Tensor:
        """
        持ち物信念状態をテンソルに変換

        各ポケモンについて上位N個の持ち物の確率を格納

        Returns:
            [num_pokemon * num_belief_items * 2] float tensor
            (item_id embedding用のIDと確率のペア)
        """
        features = []

        for pokemon_name, beliefs in item_beliefs.items():
            # 確率順にソート
            sorted_beliefs = sorted(beliefs.items(), key=lambda x: -x[1])

            for i in range(self.config.num_belief_items):
                if i < len(sorted_beliefs):
                    item_name, prob = sorted_beliefs[i]
                    item_id = self._get_item_id(item_name)
                    features.extend([float(item_id), prob])
                else:
                    features.extend([0.0, 0.0])

        return torch.tensor(features, dtype=torch.float32)

    def encode(self, turn_record: TurnRecord) -> dict[str, torch.Tensor]:
        """
        TurnRecordを完全なテンソル表現に変換

        Returns:
            dict with:
            - my_pokemon: dict of tensors
            - my_bench: list of dict of tensors
            - opp_pokemon: dict of tensors
            - opp_bench: list of dict of tensors
            - field: [24] float tensor
            - item_beliefs: float tensor
        """
        # 自分の場のポケモン
        my_pokemon = self.encode_pokemon(turn_record.my_pokemon)

        # 自分の控え
        my_bench = [self.encode_pokemon(p) for p in turn_record.my_bench]

        # 相手の場のポケモン
        opp_pokemon = self.encode_pokemon(turn_record.opp_pokemon)

        # 相手の控え
        opp_bench = [self.encode_pokemon(p) for p in turn_record.opp_bench]

        # 場の状態
        field = self.encode_field(turn_record.field)

        # 持ち物信念
        item_beliefs = self.encode_item_beliefs(turn_record.item_beliefs)

        return {
            "my_pokemon": my_pokemon,
            "my_bench": my_bench,
            "opp_pokemon": opp_pokemon,
            "opp_bench": opp_bench,
            "field": field,
            "item_beliefs": item_beliefs,
        }

    def encode_flat(self, turn_record: TurnRecord) -> torch.Tensor:
        """
        TurnRecordを1次元のフラットなテンソルに変換

        シンプルなMLPで使用する場合に便利

        Returns:
            [D] float tensor（すべての特徴量を連結）
        """
        encoded = self.encode(turn_record)

        features = []

        # ポケモン情報をフラット化するヘルパー
        def flatten_pokemon(p: dict) -> list[float]:
            flat = []
            flat.append(float(p["pokemon_id"].item()))
            flat.append(p["hp_ratio"].item())
            flat.extend(p["ailment"].tolist())
            flat.extend(p["rank"].tolist())
            flat.extend([float(x) for x in p["types"].tolist()])
            flat.append(float(p["ability_id"].item()))
            flat.append(float(p["item_id"].item()))
            flat.extend([float(x) for x in p["moves"].tolist()])
            flat.append(p["terastallized"].item())
            flat.append(float(p["tera_type"].item()))
            return flat

        # 自分の場のポケモン
        features.extend(flatten_pokemon(encoded["my_pokemon"]))

        # 自分の控え（パディング）
        for i in range(2):  # 控えは最大2体
            if i < len(encoded["my_bench"]):
                features.extend(flatten_pokemon(encoded["my_bench"][i]))
            else:
                features.extend([0.0] * 27)  # 27 = ポケモン特徴量の次元

        # 相手の場のポケモン
        features.extend(flatten_pokemon(encoded["opp_pokemon"]))

        # 相手の控え（パディング）
        for i in range(2):
            if i < len(encoded["opp_bench"]):
                features.extend(flatten_pokemon(encoded["opp_bench"][i]))
            else:
                features.extend([0.0] * 27)

        # 場の状態
        features.extend(encoded["field"].tolist())

        # 持ち物信念
        features.extend(encoded["item_beliefs"].tolist())

        return torch.tensor(features, dtype=torch.float32)

    def get_flat_dim(self) -> int:
        """encode_flatの出力次元を計算"""
        # ポケモン1体: 27次元
        # - pokemon_id: 1
        # - hp_ratio: 1
        # - ailment: 7
        # - rank: 8
        # - types: 2
        # - ability_id: 1
        # - item_id: 1
        # - moves: 4
        # - terastallized: 1
        # - tera_type: 1
        pokemon_dim = 27

        # 自分: 場1 + 控え2 = 3体
        # 相手: 場1 + 控え2 = 3体
        total_pokemon_dim = pokemon_dim * 6

        # 場の状態: 24
        field_dim = 24

        # 持ち物信念: pokemon数 * num_belief_items * 2
        # 相手3体 * 10アイテム * 2(id, prob) = 60
        belief_dim = 3 * self.config.num_belief_items * 2

        return total_pokemon_dim + field_dim + belief_dim

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
    def load(cls, path: str | Path) -> "ObservationEncoder":
        """エンコーダーの状態を読み込み"""
        with open(path, "r", encoding="utf-8") as f:
            state = json.load(f)

        config = EncodingConfig(**state["config"])
        return cls(
            config=config,
            pokemon_to_id=state["pokemon_to_id"],
            move_to_id=state["move_to_id"],
            item_to_id=state["item_to_id"],
            ability_to_id=state["ability_to_id"],
            type_to_id=state["type_to_id"],
        )
