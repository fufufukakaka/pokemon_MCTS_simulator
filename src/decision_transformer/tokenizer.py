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

        # state_features: チームプレビュー段階ではすべてゼロ
        # 次元: my_state + opp_state + field_state
        state_dim = config.pokemon_state_dim * 2 + config.field_state_dim
        state_features = [[0.0] * state_dim for _ in range(len(tokens))]

        return {
            "input_ids": torch.tensor(tokens, dtype=torch.long),
            "position_ids": torch.tensor(positions, dtype=torch.long),
            "timestep_ids": torch.tensor(timesteps, dtype=torch.long),
            "segment_ids": torch.tensor(segments, dtype=torch.long),
            "rtg_values": torch.tensor(rtg_values, dtype=torch.float),
            "attention_mask": torch.ones(len(tokens), dtype=torch.long),
            "team_token_positions": torch.tensor(my_team_positions, dtype=torch.long),
            "state_features": torch.tensor(state_features, dtype=torch.float),
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

        # state_features も引き継ぐ
        # 次元: my_state + opp_state + field_state
        state_dim = config.pokemon_state_dim * 2 + config.field_state_dim
        if "state_features" in context:
            state_features_list = context["state_features"].tolist()
        else:
            state_features_list = [[0.0] * state_dim for _ in range(len(tokens))]

        pos = len(tokens)
        turn = 0

        # [SELECT] トークン
        tokens.append(config.select_token_id)
        positions.append(pos)
        timesteps.append(turn)
        segments.append(config.segment_selection)
        rtg_values_list.append(rtg)
        state_features_list.append([0.0] * state_dim)
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
            state_features_list.append([0.0] * state_dim)
            pos += 1

        # [SEP]
        tokens.append(config.sep_token_id)
        positions.append(pos)
        timesteps.append(turn)
        segments.append(config.segment_selection)
        rtg_values_list.append(rtg)
        state_features_list.append([0.0] * state_dim)
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
            "state_features": torch.tensor(state_features_list, dtype=torch.float),
        }

        return result

    def encode_pokemon_state(
        self,
        pokemon,
        revealed_moves: list[str] | None = None,
        revealed_item: str | None = None,
        revealed_ability: str | None = None,
        is_opponent: bool = False,
    ) -> torch.Tensor:
        """
        ポケモンの状態を連続値ベクトルにエンコード

        Args:
            pokemon: Pokemon object, PokemonState, または ObservedPokemonState
            revealed_moves: 観測された技リスト（相手用、Noneなら全技を使用）
            revealed_item: 観測された持ち物（相手用、Noneなら実際の持ち物を使用）
            revealed_ability: 観測された特性（相手用、Noneなら実際の特性を使用）
            is_opponent: 相手のポケモンかどうか

        Returns:
            [pokemon_state_dim] のテンソル (59 dims)
            - HP ratio: 1
            - Ailment one-hot: 7
            - Rank changes: 8
            - Terastallized: 1
            - Tera type: 1
            - Types (2 slots): 2
            - Stats (ATK/DEF/SPA/SPD/SPE normalized): 5
            - Condition states: 10
            - Moves (4 slots * 2): 8 (move_id normalized + revealed flag)
            - Move details (4 slots * 3): 12 (power, accuracy, class)
            - Item: 2 (item_id normalized + revealed flag)
            - Ability: 2 (ability_id normalized + revealed flag)
        """
        features = []

        # HP ratio
        # Pokemon object: hp / status[0]
        # PokemonState/ObservedPokemonState: hp_ratio 属性
        if hasattr(pokemon, "hp_ratio"):
            # PokemonState or ObservedPokemonState
            hp_ratio = pokemon.hp_ratio
        elif hasattr(pokemon, "hp") and hasattr(pokemon, "status"):
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
        # Pokemon: terastal, PokemonState/ObservedPokemonState: terastallized
        terastallized = getattr(pokemon, "terastallized", None) or getattr(pokemon, "terastal", None)
        if terastallized:
            features.append(1.0)
        else:
            features.append(0.0)

        # Tera type (index, normalized)
        # Pokemon: Ttype, PokemonState/ObservedPokemonState: tera_type
        tera_type = getattr(pokemon, "tera_type", None) or getattr(pokemon, "Ttype", None)
        if tera_type:
            tera_idx = TYPE_TO_IDX.get(tera_type, 0)
            features.append(tera_idx / 19.0)  # Normalize
        else:
            features.append(0.0)

        # === 新規追加: タイプ ===
        # Pokemon: types (プロパティ), PokemonState/ObservedPokemonState: types (リスト)
        types = getattr(pokemon, "types", None) or []
        for i in range(2):
            if i < len(types) and types[i]:
                type_idx = TYPE_TO_IDX.get(types[i], 0)
                features.append(type_idx / 19.0)  # Normalize
            else:
                features.append(0.0)

        # === 新規追加: 実ステータス (ATK/DEF/SPA/SPD/SPE) ===
        # Pokemon: status[1:6], PokemonState: stats属性がないので推定不可（0で埋める）
        # 相手の場合は観測不能なので0
        if hasattr(pokemon, "status") and pokemon.status and len(pokemon.status) >= 6:
            # Pokemon object: statusはリスト [H, A, B, C, D, S]
            # 正規化: 最大値を200程度と仮定
            stats = pokemon.status[1:6]  # ATK, DEF, SPA, SPD, SPE
            for stat in stats:
                features.append(min(stat / 200.0, 1.5))  # Clamp to reasonable range
        else:
            # PokemonState/ObservedPokemonState: ステータス情報がない場合は0
            features.extend([0.0] * 5)

        # === 新規追加: 状態変化 (10 dims) ===
        # confusion (こんらん残りターン normalized by 5)
        confusion = getattr(pokemon, "confusion", 0)
        if not confusion and hasattr(pokemon, "condition"):
            confusion = pokemon.condition.get("confusion", 0)
        features.append(min(confusion / 5.0, 1.0) if confusion else 0.0)

        # encore (アンコール残りターン normalized by 3)
        encore = getattr(pokemon, "encore", 0)
        if not encore and hasattr(pokemon, "condition"):
            encore = pokemon.condition.get("encore", 0)
        features.append(min(encore / 3.0, 1.0) if encore else 0.0)

        # chohatsu (ちょうはつ残りターン normalized by 3)
        chohatsu = getattr(pokemon, "chohatsu", 0)
        if not chohatsu and hasattr(pokemon, "condition"):
            chohatsu = pokemon.condition.get("chohatsu", 0)
        features.append(min(chohatsu / 3.0, 1.0) if chohatsu else 0.0)

        # change_block (にげられない binary)
        change_block = getattr(pokemon, "change_block", False)
        if not change_block and hasattr(pokemon, "condition"):
            change_block = pokemon.condition.get("change_block", 0)
        features.append(1.0 if change_block else 0.0)

        # bind (バインド残りターン normalized by 5)
        bind = getattr(pokemon, "bind", 0)
        if not bind and hasattr(pokemon, "condition"):
            bind = pokemon.condition.get("bind", 0)
        # bindは小数点を含む場合がある (残りターン + 0.1 * ダメージ割合)
        features.append(min(float(int(bind)) / 5.0, 1.0) if bind else 0.0)

        # yadorigi (やどりぎのタネ binary)
        yadorigi = getattr(pokemon, "yadorigi", False)
        if not yadorigi and hasattr(pokemon, "condition"):
            yadorigi = pokemon.condition.get("yadorigi", 0)
        features.append(1.0 if yadorigi else 0.0)

        # sub_hp (みがわり残りHP normalized by max HP)
        sub_hp = getattr(pokemon, "sub_hp", 0)
        if not sub_hp and hasattr(pokemon, "condition"):
            sub_hp = pokemon.condition.get("sub_hp", 0)
        # max HP は status[0] から取得可能
        max_hp = 1
        if hasattr(pokemon, "status") and pokemon.status:
            max_hp = pokemon.status[0] if pokemon.status[0] > 0 else 1
        features.append(min(sub_hp / max_hp, 1.0) if sub_hp else 0.0)

        # meromero (メロメロ binary)
        meromero = getattr(pokemon, "meromero", False)
        if not meromero and hasattr(pokemon, "condition"):
            meromero = pokemon.condition.get("meromero", 0)
        features.append(1.0 if meromero else 0.0)

        # noroi (のろい binary)
        noroi = getattr(pokemon, "noroi", False)
        if not noroi and hasattr(pokemon, "condition"):
            noroi = pokemon.condition.get("noroi", 0)
        features.append(1.0 if noroi else 0.0)

        # horobi (ほろびのうたカウント normalized by 4)
        horobi = getattr(pokemon, "horobi", 0)
        if not horobi and hasattr(pokemon, "condition"):
            horobi = pokemon.condition.get("horobi", 0)
        features.append(min(horobi / 4.0, 1.0) if horobi else 0.0)

        # === 技関連情報 ===
        # Moves (4 slots)
        # Pokemon: move, PokemonState: moves, ObservedPokemonState: revealed_moves
        if is_opponent and revealed_moves is not None:
            # 相手: 観測された技のみ（引数で指定）
            moves = revealed_moves
        else:
            # 自分: 全技を使用
            # 属性名: move (Pokemon), moves (PokemonState), revealed_moves (ObservedPokemonState)
            moves = getattr(pokemon, "moves", None) or getattr(pokemon, "move", None) or getattr(pokemon, "revealed_moves", None) or []

        # Move IDs (4 slots * 2 = 8 dims)
        for i in range(4):
            if i < len(moves) and moves[i]:
                move_id = self.battle_vocab.get_move_id(moves[i])
                # Normalize by vocab size
                features.append(move_id / max(self.config.move_vocab_size, 1))
                features.append(1.0)  # Revealed/Known flag
            else:
                features.append(0.0)
                features.append(0.0)

        # Move details (4 slots * 3 = 12 dims: power, accuracy, class)
        for i in range(4):
            if i < len(moves) and moves[i]:
                move_name = moves[i]
                move_info = self._get_move_info(move_name)
                # Power (normalized by 200, max realistic power)
                features.append(min(move_info.get("power", 0) / 200.0, 1.0))
                # Accuracy (normalized by 100, 0 = always hits)
                hit = move_info.get("hit", 100)
                features.append(hit / 100.0 if hit > 0 else 1.0)
                # Class (0=変化sta, 0.5=物理phy, 1.0=特殊spe)
                move_class = move_info.get("class", "sta")
                if "phy" in move_class:
                    features.append(0.5)
                elif "spe" in move_class:
                    features.append(1.0)
                else:  # sta (変化技)
                    features.append(0.0)
            else:
                features.append(0.0)  # power
                features.append(0.0)  # accuracy
                features.append(0.0)  # class

        # Item
        # Pokemon/PokemonState: item, ObservedPokemonState: revealed_item
        if is_opponent and revealed_item is not None:
            item = revealed_item if revealed_item else None
        else:
            item = getattr(pokemon, "item", None) or getattr(pokemon, "revealed_item", None)

        if item:
            item_id = self.battle_vocab.get_item_id(item)
            features.append(item_id / max(self.config.item_vocab_size, 1))
            features.append(1.0)
        else:
            features.append(0.0)
            features.append(0.0)

        # Ability
        # Pokemon/PokemonState: ability, ObservedPokemonState: revealed_ability
        if is_opponent and revealed_ability is not None:
            ability = revealed_ability if revealed_ability else None
        else:
            ability = getattr(pokemon, "ability", None) or getattr(pokemon, "revealed_ability", None)

        if ability:
            ability_id = self.battle_vocab.get_ability_id(ability)
            features.append(ability_id / max(self.config.ability_vocab_size, 1))
            features.append(1.0)
        else:
            features.append(0.0)
            features.append(0.0)

        return torch.tensor(features, dtype=torch.float)

    def _get_move_info(self, move_name: str) -> dict:
        """
        技の情報を取得

        Returns:
            dict with keys: type, class, power, hit, pp
        """
        try:
            from src.pokemon_battle_sim.pokemon import Pokemon
            if move_name and move_name in Pokemon.all_moves:
                return Pokemon.all_moves[move_name]
        except (ImportError, AttributeError):
            pass
        # Default empty info
        return {"type": "", "class": "sta", "power": 0, "hit": 100, "pp": 0}

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
        opp_revealed_moves: list[str] | None = None,
        opp_revealed_item: str | None = None,
        opp_revealed_ability: str | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        ターン状態をエンコードしてコンテキストに追加

        Args:
            battle: バトル状態
            player: プレイヤー番号 (0 or 1)
            turn: ターン番号
            rtg: Return-to-go (目標リターン)
            context: 既存のエンコード済みコンテキスト
            opp_revealed_moves: 相手の観測済み技リスト
            opp_revealed_item: 相手の観測済み持ち物
            opp_revealed_ability: 相手の観測済み特性

        系列形式 (継続):
        ... [TURN] [STATE] state_features [ACTION]
        """
        config = self.config

        # 状態特徴量の次元: my_state + opp_state + field_state
        state_dim = config.pokemon_state_dim * 2 + config.field_state_dim

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
        state_features_list.append([0.0] * state_dim)
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

        # 自分のポケモン: 全情報を使用
        my_state = self.encode_pokemon_state(
            my_pokemon,
            is_opponent=False,
        ) if my_pokemon else torch.zeros(config.pokemon_state_dim)

        # 相手のポケモン: 観測情報のみ使用
        opp_state = self.encode_pokemon_state(
            opp_pokemon,
            revealed_moves=opp_revealed_moves,
            revealed_item=opp_revealed_item,
            revealed_ability=opp_revealed_ability,
            is_opponent=True,
        ) if opp_pokemon else torch.zeros(config.pokemon_state_dim)

        field_state = self.encode_field_state(battle)

        # 結合: [my_state, opp_state, field_state]
        combined_state = torch.cat([my_state, opp_state, field_state])
        state_features_list.append(combined_state.tolist())
        pos += 1

        # [ACTION] トークン（行動を予測する位置）
        tokens.append(config.action_token_id)
        positions.append(pos)
        timesteps.append(turn)
        segments.append(config.segment_battle)
        rtg_values_list.append(rtg)
        state_features_list.append([0.0] * state_dim)
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

    def encode_turn_state_from_data(
        self,
        turn_state: "TurnState",
        context: dict[str, torch.Tensor],
        rtg: float = 1.0,
    ) -> dict[str, torch.Tensor]:
        """
        TurnState データクラスからターン状態をエンコード

        ai_service.py で使用するための互換メソッド。
        Battle オブジェクトではなく TurnState データクラスを受け取る。

        Args:
            turn_state: TurnState データクラス
            context: 既存のエンコード済みコンテキスト
            rtg: Return-to-go (目標リターン)
        """
        config = self.config

        # 状態特徴量の次元: my_state + opp_state + field_state
        state_dim = config.pokemon_state_dim * 2 + config.field_state_dim

        # 既存のコンテキストをコピー
        tokens = context["input_ids"].tolist()
        positions = context["position_ids"].tolist()
        timesteps = context["timestep_ids"].tolist()
        segments = context["segment_ids"].tolist()
        rtg_values_list = context["rtg_values"].tolist()

        state_features_list = []
        if "state_features" in context:
            state_features_list = context["state_features"].tolist()

        pos = len(tokens)
        turn = turn_state.turn

        # [TURN] トークン
        tokens.append(config.turn_token_id)
        positions.append(pos)
        timesteps.append(turn)
        segments.append(config.segment_battle)
        rtg_values_list.append(rtg)
        state_features_list.append([0.0] * state_dim)
        pos += 1

        # [STATE] トークン
        tokens.append(config.state_token_id)
        positions.append(pos)
        timesteps.append(turn)
        segments.append(config.segment_battle)
        rtg_values_list.append(rtg)

        # 状態特徴量を構築（TurnState から）
        my_state = self.encode_pokemon_state(
            turn_state.my_active,
            is_opponent=False,
        ) if turn_state.my_active else torch.zeros(config.pokemon_state_dim)

        opp_state = self.encode_pokemon_state(
            turn_state.opp_active,
            is_opponent=True,
        ) if turn_state.opp_active else torch.zeros(config.pokemon_state_dim)

        # フィールド状態
        field_features = self._encode_field_state_from_data(turn_state.field_state)

        # 結合: [my_state, opp_state, field_state]
        combined_state = torch.cat([my_state, opp_state, field_features])
        state_features_list.append(combined_state.tolist())
        pos += 1

        # [ACTION] トークン
        tokens.append(config.action_token_id)
        positions.append(pos)
        timesteps.append(turn)
        segments.append(config.segment_battle)
        rtg_values_list.append(rtg)
        state_features_list.append([0.0] * state_dim)
        action_position = pos
        pos += 1

        # action_token_positions を更新
        action_positions = context.get("action_token_positions", torch.tensor([], dtype=torch.long)).tolist()
        action_positions.append(action_position)

        return {
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

    def _encode_field_state_from_data(self, field_state: "FieldState") -> torch.Tensor:
        """FieldState データクラスからフィールド状態をエンコード"""
        features = []

        # Weather (4)
        weather_map = {"sunny": 0, "rainy": 1, "snow": 2, "sandstorm": 3}
        weather_vec = [0.0] * 4
        if field_state.weather and field_state.weather in weather_map:
            idx = weather_map[field_state.weather]
            weather_vec[idx] = field_state.weather_turns / 5.0
        features.extend(weather_vec)

        # Terrain (4)
        terrain_map = {"electric": 0, "grass": 1, "psychic": 2, "mist": 3}
        terrain_vec = [0.0] * 4
        if field_state.terrain and field_state.terrain in terrain_map:
            idx = terrain_map[field_state.terrain]
            terrain_vec[idx] = field_state.terrain_turns / 5.0
        features.extend(terrain_vec)

        # Trick room and gravity (2)
        features.append(field_state.trick_room / 5.0 if field_state.trick_room else 0.0)
        features.append(field_state.gravity / 5.0 if field_state.gravity else 0.0)

        # Screens for both players (4): reflector, light_screen
        features.append(field_state.reflector[0] / 5.0 if field_state.reflector else 0.0)
        features.append(field_state.reflector[1] / 5.0 if field_state.reflector and len(field_state.reflector) > 1 else 0.0)
        features.append(field_state.light_screen[0] / 5.0 if field_state.light_screen else 0.0)
        features.append(field_state.light_screen[1] / 5.0 if field_state.light_screen and len(field_state.light_screen) > 1 else 0.0)

        # Tailwind (2)
        features.append(1.0 if field_state.tailwind and field_state.tailwind[0] > 0 else 0.0)
        features.append(1.0 if field_state.tailwind and len(field_state.tailwind) > 1 and field_state.tailwind[1] > 0 else 0.0)

        # Entry hazards (8)
        features.append(1.0 if field_state.stealth_rock and field_state.stealth_rock[0] else 0.0)
        features.append(field_state.spikes[0] / 3.0 if field_state.spikes else 0.0)
        features.append(field_state.toxic_spikes[0] / 2.0 if field_state.toxic_spikes else 0.0)
        features.append(1.0 if field_state.sticky_web and field_state.sticky_web[0] else 0.0)

        features.append(1.0 if field_state.stealth_rock and len(field_state.stealth_rock) > 1 and field_state.stealth_rock[1] else 0.0)
        features.append(field_state.spikes[1] / 3.0 if field_state.spikes and len(field_state.spikes) > 1 else 0.0)
        features.append(field_state.toxic_spikes[1] / 2.0 if field_state.toxic_spikes and len(field_state.toxic_spikes) > 1 else 0.0)
        features.append(1.0 if field_state.sticky_web and len(field_state.sticky_web) > 1 and field_state.sticky_web[1] else 0.0)

        return torch.tensor(features, dtype=torch.float)

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
