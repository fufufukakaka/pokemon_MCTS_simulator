"""
Battle Sequence Tokenizer

バトル軌跡をトークン系列に変換する。
Decision Transformer の入力形式に合わせた系列を生成する。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import torch

from src.selection_bert.dataset import POKEMON_NAME_ALIASES, PokemonVocab

from .config import PokemonBattleTransformerConfig

if TYPE_CHECKING:
    from src.pokemon_battle_sim.battle import Battle


# 技名→ID マッピング（動的に構築）
# 持ち物名→ID マッピング（動的に構築）
# 特性名→ID マッピング（動的に構築）


@dataclass
class BattleVocab:
    """バトル関連の語彙管理（Pokemon以外）"""

    move_to_id: dict[str, int] = field(default_factory=dict)
    id_to_move: dict[int, str] = field(default_factory=dict)
    item_to_id: dict[str, int] = field(default_factory=dict)
    id_to_item: dict[int, str] = field(default_factory=dict)
    ability_to_id: dict[str, int] = field(default_factory=dict)
    id_to_ability: dict[int, str] = field(default_factory=dict)

    # 次のID (0はパディング用に予約)
    _next_move_id: int = 1
    _next_item_id: int = 1
    _next_ability_id: int = 1

    def get_move_id(self, move: str) -> int:
        """技のIDを取得（なければ新規登録）"""
        if not move:
            return 0
        if move not in self.move_to_id:
            self.move_to_id[move] = self._next_move_id
            self.id_to_move[self._next_move_id] = move
            self._next_move_id += 1
        return self.move_to_id[move]

    def get_item_id(self, item: str) -> int:
        """持ち物のIDを取得（なければ新規登録）"""
        if not item:
            return 0
        if item not in self.item_to_id:
            self.item_to_id[item] = self._next_item_id
            self.id_to_item[self._next_item_id] = item
            self._next_item_id += 1
        return self.item_to_id[item]

    def get_ability_id(self, ability: str) -> int:
        """特性のIDを取得（なければ新規登録）"""
        if not ability:
            return 0
        if ability not in self.ability_to_id:
            self.ability_to_id[ability] = self._next_ability_id
            self.id_to_ability[self._next_ability_id] = ability
            self._next_ability_id += 1
        return self.ability_to_id[ability]

    def save(self, path: Path) -> None:
        """語彙を保存"""
        data = {
            "move_to_id": self.move_to_id,
            "item_to_id": self.item_to_id,
            "ability_to_id": self.ability_to_id,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: Path) -> "BattleVocab":
        """語彙をロード"""
        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        vocab = cls()
        vocab.move_to_id = data.get("move_to_id", {})
        vocab.id_to_move = {int(v): k for k, v in vocab.move_to_id.items()}
        vocab.item_to_id = data.get("item_to_id", {})
        vocab.id_to_item = {int(v): k for k, v in vocab.item_to_id.items()}
        vocab.ability_to_id = data.get("ability_to_id", {})
        vocab.id_to_ability = {int(v): k for k, v in vocab.ability_to_id.items()}

        vocab._next_move_id = max(vocab.move_to_id.values(), default=0) + 1
        vocab._next_item_id = max(vocab.item_to_id.values(), default=0) + 1
        vocab._next_ability_id = max(vocab.ability_to_id.values(), default=0) + 1

        return vocab


# 状態異常のエンコード
AILMENT_TO_IDX = {
    "": 0,
    None: 0,
    "どく": 1,
    "もうどく": 2,
    "やけど": 3,
    "まひ": 4,
    "ねむり": 5,
    "こおり": 6,
}

# タイプのエンコード
TYPE_TO_IDX = {
    "": 0,
    None: 0,
    "ノーマル": 1,
    "ほのお": 2,
    "みず": 3,
    "でんき": 4,
    "くさ": 5,
    "こおり": 6,
    "かくとう": 7,
    "どく": 8,
    "じめん": 9,
    "ひこう": 10,
    "エスパー": 11,
    "むし": 12,
    "いわ": 13,
    "ゴースト": 14,
    "ドラゴン": 15,
    "あく": 16,
    "はがね": 17,
    "フェアリー": 18,
    "ステラ": 19,
}


@dataclass
class TokenizedBattle:
    """トークン化されたバトルデータ"""

    # トークンID系列
    input_ids: torch.Tensor  # [seq_len]

    # ポジション情報
    position_ids: torch.Tensor  # [seq_len]
    timestep_ids: torch.Tensor  # [seq_len] ターン番号
    segment_ids: torch.Tensor  # [seq_len] 0=preview, 1=selection, 2=battle

    # Attention mask
    attention_mask: torch.Tensor  # [seq_len]

    # Return-to-go 値
    rtg_values: torch.Tensor  # [seq_len]

    # 状態特徴量（連続値）
    state_features: torch.Tensor  # [seq_len, state_dim]

    # ラベル（学習用）
    selection_labels: Optional[torch.Tensor] = None  # [6] 選出ラベル
    action_labels: Optional[torch.Tensor] = None  # [num_action_positions]
    value_labels: Optional[torch.Tensor] = None  # [1] 勝敗

    # 行動マスク
    action_masks: Optional[torch.Tensor] = None  # [num_action_positions, num_actions]

    # メタ情報
    team_token_positions: Optional[torch.Tensor] = None  # [6] 自チームのトークン位置
    action_token_positions: Optional[torch.Tensor] = None  # 行動トークンの位置


class BattleSequenceTokenizer:
    """バトル軌跡をトークン系列に変換"""

    def __init__(
        self,
        config: PokemonBattleTransformerConfig,
        pokemon_vocab: Optional[PokemonVocab] = None,
        battle_vocab: Optional[BattleVocab] = None,
    ):
        self.config = config

        # Pokemon 語彙
        if pokemon_vocab is None:
            zukan_path = Path(__file__).parent.parent.parent / "data" / "zukan.txt"
            if zukan_path.exists():
                self.pokemon_vocab = PokemonVocab.from_zukan(zukan_path)
            else:
                raise FileNotFoundError(f"zukan.txt not found at {zukan_path}")
        else:
            self.pokemon_vocab = pokemon_vocab

        # バトル語彙（技、持ち物、特性）
        self.battle_vocab = battle_vocab or BattleVocab()

    def normalize_pokemon_name(self, name: str) -> str:
        """ポケモン名を正規化"""
        return POKEMON_NAME_ALIASES.get(name, name)

    def encode_pokemon_id(self, name: str) -> int:
        """ポケモン名をIDに変換"""
        normalized = self.normalize_pokemon_name(name)
        return self.pokemon_vocab.encode(normalized)

    def encode_team_preview(
        self,
        my_team: list[str],
        opp_team: list[str],
        rtg: float = 1.0,
    ) -> dict[str, torch.Tensor]:
        """
        チームプレビューをエンコード

        系列形式:
        [RTG] [PREVIEW] [MY_TEAM] my1 my2 my3 my4 my5 my6 [SEP] [OPP_TEAM] opp1 ... opp6 [SEP]
        """
        config = self.config
        tokens = []
        positions = []
        timesteps = []
        segments = []
        rtg_values = []

        pos = 0
        turn = 0

        # [RTG] トークン
        tokens.append(config.rtg_token_id)
        positions.append(pos)
        timesteps.append(turn)
        segments.append(config.segment_preview)
        rtg_values.append(rtg)
        pos += 1

        # [PREVIEW] トークン
        tokens.append(config.preview_token_id)
        positions.append(pos)
        timesteps.append(turn)
        segments.append(config.segment_preview)
        rtg_values.append(rtg)
        pos += 1

        # [MY_TEAM] トークン
        tokens.append(config.my_team_token_id)
        positions.append(pos)
        timesteps.append(turn)
        segments.append(config.segment_preview)
        rtg_values.append(rtg)
        pos += 1

        # 自チームの Pokemon
        my_team_positions = []
        for i, pokemon in enumerate(my_team[:6]):
            pokemon_id = self.encode_pokemon_id(pokemon)
            tokens.append(pokemon_id)
            positions.append(pos)
            timesteps.append(turn)
            segments.append(config.segment_preview)
            rtg_values.append(rtg)
            my_team_positions.append(pos)
            pos += 1

        # パディング（6匹未満の場合）
        while len(my_team_positions) < 6:
            tokens.append(config.pad_token_id)
            positions.append(pos)
            timesteps.append(turn)
            segments.append(config.segment_preview)
            rtg_values.append(rtg)
            my_team_positions.append(pos)
            pos += 1

        # [SEP]
        tokens.append(config.sep_token_id)
        positions.append(pos)
        timesteps.append(turn)
        segments.append(config.segment_preview)
        rtg_values.append(rtg)
        pos += 1

        # [OPP_TEAM] トークン
        tokens.append(config.opp_team_token_id)
        positions.append(pos)
        timesteps.append(turn)
        segments.append(config.segment_preview)
        rtg_values.append(rtg)
        pos += 1

        # 相手チームの Pokemon
        for i, pokemon in enumerate(opp_team[:6]):
            pokemon_id = self.encode_pokemon_id(pokemon)
            tokens.append(pokemon_id)
            positions.append(pos)
            timesteps.append(turn)
            segments.append(config.segment_preview)
            rtg_values.append(rtg)
            pos += 1

        # パディング（6匹未満の場合）
        for _ in range(6 - len(opp_team)):
            tokens.append(config.pad_token_id)
            positions.append(pos)
            timesteps.append(turn)
            segments.append(config.segment_preview)
            rtg_values.append(rtg)
            pos += 1

        # [SEP]
        tokens.append(config.sep_token_id)
        positions.append(pos)
        timesteps.append(turn)
        segments.append(config.segment_preview)
        rtg_values.append(rtg)
        pos += 1

        return {
            "input_ids": torch.tensor(tokens, dtype=torch.long),
            "position_ids": torch.tensor(positions, dtype=torch.long),
            "timestep_ids": torch.tensor(timesteps, dtype=torch.long),
            "segment_ids": torch.tensor(segments, dtype=torch.long),
            "rtg_values": torch.tensor(rtg_values, dtype=torch.float),
            "attention_mask": torch.ones(len(tokens), dtype=torch.long),
            "team_token_positions": torch.tensor(my_team_positions, dtype=torch.long),
        }

    def encode_selection(
        self,
        selected_indices: list[int],
        lead_index: int,
        context: dict[str, torch.Tensor],
        rtg: float = 1.0,
    ) -> dict[str, torch.Tensor]:
        """
        選出情報を追加

        系列形式 (継続):
        ... [SELECT] sel1 sel2 sel3 [SEP]
        """
        config = self.config

        # 既存のコンテキストをコピー
        tokens = context["input_ids"].tolist()
        positions = context["position_ids"].tolist()
        timesteps = context["timestep_ids"].tolist()
        segments = context["segment_ids"].tolist()
        rtg_values_list = context["rtg_values"].tolist()

        pos = len(tokens)
        turn = 0

        # [SELECT] トークン
        tokens.append(config.select_token_id)
        positions.append(pos)
        timesteps.append(turn)
        segments.append(config.segment_selection)
        rtg_values_list.append(rtg)
        pos += 1

        # 選出した Pokemon（先発を最初に）
        # lead_index を先頭に移動
        ordered_selection = [lead_index] + [i for i in selected_indices if i != lead_index]

        for idx in ordered_selection[:3]:
            # Pokemon ID は preview で既にエンコード済みなので、
            # ここでは選出インデックスを特別なトークンとしてエンコード
            # または、team_token_positions から Pokemon ID を取得
            team_positions = context["team_token_positions"]
            pokemon_token = context["input_ids"][team_positions[idx]].item()
            tokens.append(pokemon_token)
            positions.append(pos)
            timesteps.append(turn)
            segments.append(config.segment_selection)
            rtg_values_list.append(rtg)
            pos += 1

        # [SEP]
        tokens.append(config.sep_token_id)
        positions.append(pos)
        timesteps.append(turn)
        segments.append(config.segment_selection)
        rtg_values_list.append(rtg)
        pos += 1

        # 選出ラベルを作成
        selection_labels = torch.zeros(6, dtype=torch.long)
        for idx in selected_indices:
            if idx == lead_index:
                selection_labels[idx] = 2  # LEAD
            else:
                selection_labels[idx] = 1  # SELECTED

        result = {
            "input_ids": torch.tensor(tokens, dtype=torch.long),
            "position_ids": torch.tensor(positions, dtype=torch.long),
            "timestep_ids": torch.tensor(timesteps, dtype=torch.long),
            "segment_ids": torch.tensor(segments, dtype=torch.long),
            "rtg_values": torch.tensor(rtg_values_list, dtype=torch.float),
            "attention_mask": torch.ones(len(tokens), dtype=torch.long),
            "team_token_positions": context["team_token_positions"],
            "selection_labels": selection_labels,
        }

        return result

    def encode_pokemon_state(self, pokemon) -> torch.Tensor:
        """
        ポケモンの状態を連続値ベクトルにエンコード

        Returns:
            [pokemon_state_dim] のテンソル
        """
        features = []

        # HP ratio
        if hasattr(pokemon, "hp") and hasattr(pokemon, "status"):
            max_hp = pokemon.status[0] if pokemon.status else 1
            hp_ratio = pokemon.hp / max_hp if max_hp > 0 else 0
        else:
            hp_ratio = 1.0
        features.append(hp_ratio)

        # Ailment (one-hot, 7 classes)
        ailment_vec = [0.0] * 7
        if hasattr(pokemon, "ailment") and pokemon.ailment:
            ailment_idx = AILMENT_TO_IDX.get(pokemon.ailment, 0)
            if 0 <= ailment_idx < 7:
                ailment_vec[ailment_idx] = 1.0
        features.extend(ailment_vec)

        # Rank changes (8 values: HP, ATK, DEF, SPA, SPD, SPE, ACC, EVA)
        if hasattr(pokemon, "rank") and pokemon.rank:
            # Normalize to [-1, 1]
            ranks = [r / 6.0 for r in pokemon.rank[:8]]
            while len(ranks) < 8:
                ranks.append(0.0)
        else:
            ranks = [0.0] * 8
        features.extend(ranks)

        # Terastallized flag
        if hasattr(pokemon, "terastal") and pokemon.terastal:
            features.append(1.0)
        else:
            features.append(0.0)

        # Tera type (index, normalized)
        if hasattr(pokemon, "Ttype") and pokemon.Ttype:
            tera_idx = TYPE_TO_IDX.get(pokemon.Ttype, 0)
            features.append(tera_idx / 19.0)  # Normalize
        else:
            features.append(0.0)

        return torch.tensor(features, dtype=torch.float)

    def encode_field_state(self, battle: "Battle") -> torch.Tensor:
        """
        フィールド状態をエンコード

        Returns:
            [field_state_dim] のテンソル (24 dims)
        """
        features = []
        condition = battle.condition

        # Weather (4): sunny, rainy, snow, sandstorm (normalized by 5)
        features.append(condition.get("sunny", 0) / 5.0)
        features.append(condition.get("rainy", 0) / 5.0)
        features.append(condition.get("snow", 0) / 5.0)
        features.append(condition.get("sandstorm", 0) / 5.0)

        # Terrain (4): electric, grass, psychic, mist
        features.append(1.0 if condition.get("elecfield", 0) > 0 else 0.0)
        features.append(1.0 if condition.get("glassfield", 0) > 0 else 0.0)
        features.append(1.0 if condition.get("psycofield", 0) > 0 else 0.0)
        features.append(1.0 if condition.get("mistfield", 0) > 0 else 0.0)

        # Misc (2): gravity, trick room
        features.append(1.0 if condition.get("gravity", 0) > 0 else 0.0)
        features.append(1.0 if condition.get("trickroom", 0) > 0 else 0.0)

        # Screens for both players (4): reflector, light_screen
        reflector = condition.get("reflector", [0, 0])
        lightwall = condition.get("lightwall", [0, 0])
        features.append(1.0 if reflector[0] > 0 else 0.0)
        features.append(1.0 if lightwall[0] > 0 else 0.0)
        features.append(1.0 if reflector[1] > 0 else 0.0)
        features.append(1.0 if lightwall[1] > 0 else 0.0)

        # Tailwind (2)
        oikaze = condition.get("oikaze", [0, 0])
        features.append(1.0 if oikaze[0] > 0 else 0.0)
        features.append(1.0 if oikaze[1] > 0 else 0.0)

        # Entry hazards for both players (8)
        # Stealth rock, spikes (3 levels), toxic spikes (2 levels), sticky web
        stealthrock = condition.get("stealthrock", [0, 0])
        makibishi = condition.get("makibishi", [0, 0])
        dokubishi = condition.get("dokubishi", [0, 0])
        nebanet = condition.get("nebanet", [0, 0])

        # Player 0
        features.append(1.0 if stealthrock[0] > 0 else 0.0)
        features.append(makibishi[0] / 3.0)  # 0-3 layers
        features.append(dokubishi[0] / 2.0)  # 0-2 layers
        features.append(1.0 if nebanet[0] > 0 else 0.0)

        # Player 1
        features.append(1.0 if stealthrock[1] > 0 else 0.0)
        features.append(makibishi[1] / 3.0)
        features.append(dokubishi[1] / 2.0)
        features.append(1.0 if nebanet[1] > 0 else 0.0)

        return torch.tensor(features, dtype=torch.float)

    def encode_turn_state(
        self,
        battle: "Battle",
        player: int,
        turn: int,
        rtg: float,
        context: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """
        ターン状態をエンコードしてコンテキストに追加

        系列形式 (継続):
        ... [TURN] [STATE] state_features [ACTION]
        """
        config = self.config

        # 既存のコンテキストをコピー
        tokens = context["input_ids"].tolist()
        positions = context["position_ids"].tolist()
        timesteps = context["timestep_ids"].tolist()
        segments = context["segment_ids"].tolist()
        rtg_values_list = context["rtg_values"].tolist()

        # state_features の収集
        state_features_list = []
        if "state_features" in context:
            state_features_list = context["state_features"].tolist()

        pos = len(tokens)

        # [TURN] トークン
        tokens.append(config.turn_token_id)
        positions.append(pos)
        timesteps.append(turn)
        segments.append(config.segment_battle)
        rtg_values_list.append(rtg)
        state_features_list.append([0.0] * (config.pokemon_state_dim + config.field_state_dim))
        pos += 1

        # [STATE] トークン
        tokens.append(config.state_token_id)
        positions.append(pos)
        timesteps.append(turn)
        segments.append(config.segment_battle)
        rtg_values_list.append(rtg)

        # 状態特徴量を構築
        my_pokemon = battle.pokemon[player]
        opp_pokemon = battle.pokemon[1 - player]

        my_state = self.encode_pokemon_state(my_pokemon) if my_pokemon else torch.zeros(config.pokemon_state_dim)
        opp_state = self.encode_pokemon_state(opp_pokemon) if opp_pokemon else torch.zeros(config.pokemon_state_dim)
        field_state = self.encode_field_state(battle)

        # 結合: [my_state, opp_state, field_state]
        # 注: pokemon_state_dim を 2 倍にして my + opp を含める場合
        # ここでは簡略化のため、field_state のみを state_features に追加
        combined_state = torch.cat([my_state, field_state])
        state_features_list.append(combined_state.tolist())
        pos += 1

        # [ACTION] トークン（行動を予測する位置）
        tokens.append(config.action_token_id)
        positions.append(pos)
        timesteps.append(turn)
        segments.append(config.segment_battle)
        rtg_values_list.append(rtg)
        state_features_list.append([0.0] * (config.pokemon_state_dim + config.field_state_dim))
        action_position = pos
        pos += 1

        # action_token_positions を更新
        action_positions = context.get("action_token_positions", torch.tensor([], dtype=torch.long)).tolist()
        action_positions.append(action_position)

        result = {
            "input_ids": torch.tensor(tokens, dtype=torch.long),
            "position_ids": torch.tensor(positions, dtype=torch.long),
            "timestep_ids": torch.tensor(timesteps, dtype=torch.long),
            "segment_ids": torch.tensor(segments, dtype=torch.long),
            "rtg_values": torch.tensor(rtg_values_list, dtype=torch.float),
            "attention_mask": torch.ones(len(tokens), dtype=torch.long),
            "state_features": torch.tensor(state_features_list, dtype=torch.float),
            "team_token_positions": context.get("team_token_positions"),
            "selection_labels": context.get("selection_labels"),
            "action_token_positions": torch.tensor(action_positions, dtype=torch.long),
        }

        return result

    def encode_action(
        self,
        action_id: int,
        context: dict[str, torch.Tensor],
        turn: int,
        rtg: float,
    ) -> dict[str, torch.Tensor]:
        """
        行動をエンコードしてコンテキストに追加

        行動IDの範囲:
        - 0-3: MOVE
        - 10-13: TERA + MOVE
        - 20-25: SWITCH
        - 30: STRUGGLE
        - -1: SKIP
        """
        config = self.config

        tokens = context["input_ids"].tolist()
        positions = context["position_ids"].tolist()
        timesteps = context["timestep_ids"].tolist()
        segments = context["segment_ids"].tolist()
        rtg_values_list = context["rtg_values"].tolist()
        state_features_list = context["state_features"].tolist() if "state_features" in context else []

        pos = len(tokens)

        # 行動トークンを追加
        # 行動ID を特殊トークン空間にマッピング
        # ここでは簡略化のため、action_id + offset を使用
        action_token = config.action_token_id  # 実際にはより細かくエンコードする
        tokens.append(action_token)
        positions.append(pos)
        timesteps.append(turn)
        segments.append(config.segment_battle)
        rtg_values_list.append(rtg)

        if state_features_list:
            state_features_list.append([0.0] * len(state_features_list[0]))

        result = {
            "input_ids": torch.tensor(tokens, dtype=torch.long),
            "position_ids": torch.tensor(positions, dtype=torch.long),
            "timestep_ids": torch.tensor(timesteps, dtype=torch.long),
            "segment_ids": torch.tensor(segments, dtype=torch.long),
            "rtg_values": torch.tensor(rtg_values_list, dtype=torch.float),
            "attention_mask": torch.ones(len(tokens), dtype=torch.long),
            "team_token_positions": context.get("team_token_positions"),
            "selection_labels": context.get("selection_labels"),
            "action_token_positions": context.get("action_token_positions"),
        }

        if state_features_list:
            result["state_features"] = torch.tensor(state_features_list, dtype=torch.float)

        return result

    def create_action_mask(self, battle: "Battle", player: int) -> torch.Tensor:
        """
        利用可能な行動のマスクを作成

        Returns:
            [num_action_outputs] のバイナリマスク
        """
        available = battle.available_commands(player, phase="battle")

        mask = torch.zeros(self.config.num_action_outputs, dtype=torch.float)
        for cmd in available:
            if 0 <= cmd < self.config.num_action_outputs:
                mask[cmd] = 1.0
            elif cmd == 30:  # STRUGGLE
                mask[30] = 1.0

        return mask

    def save(self, path: Path) -> None:
        """トークナイザを保存"""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Pokemon vocab
        self.pokemon_vocab.save(path / "pokemon_vocab.json")

        # Battle vocab
        self.battle_vocab.save(path / "battle_vocab.json")

        # Config は別途保存（config.py で管理）

    @classmethod
    def load(cls, path: Path, config: Optional[PokemonBattleTransformerConfig] = None) -> "BattleSequenceTokenizer":
        """トークナイザをロード"""
        path = Path(path)
        config = config or PokemonBattleTransformerConfig()

        pokemon_vocab = PokemonVocab.load(path / "pokemon_vocab.json")
        battle_vocab = BattleVocab.load(path / "battle_vocab.json")

        return cls(config=config, pokemon_vocab=pokemon_vocab, battle_vocab=battle_vocab)
