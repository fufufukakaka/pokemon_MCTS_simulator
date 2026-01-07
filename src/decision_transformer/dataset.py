"""
Battle Trajectory Dataset

バトル軌跡のデータ構造と Dataset クラス。
"""

from __future__ import annotations

import json
import random
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

import torch
from torch.utils.data import Dataset

from .config import PokemonBattleTransformerConfig
from .tokenizer import BattleSequenceTokenizer


@dataclass
class PokemonState:
    """自分のポケモンの状態（完全情報）"""

    name: str
    hp_ratio: float  # 0.0 - 1.0
    ailment: str = ""  # どく, やけど, まひ, etc.
    rank: list[int] = field(default_factory=lambda: [0] * 8)  # ランク変化
    types: list[str] = field(default_factory=list)  # タイプ
    terastallized: bool = False
    tera_type: str = ""

    # 実ステータス [H, A, B, C, D, S] - 観戦可能な自分の情報
    status: list[int] = field(default_factory=lambda: [0] * 6)

    # 自分の情報は全て見える
    item: str = ""
    ability: str = ""
    moves: list[str] = field(default_factory=list)

    # 状態変化 (pokemon.condition)
    confusion: int = 0  # こんらん残りターン
    critical_rank: int = 0  # 急所ランク上昇
    aquaring: bool = False  # アクアリング
    healblock: int = 0  # かいふくふうじ残りターン
    magnetrise: int = 0  # でんじふゆう残りターン
    noroi: bool = False  # のろい
    horobi: int = 0  # ほろびのうたカウント
    yadorigi: bool = False  # やどりぎのタネ
    encore: int = 0  # アンコール残りターン
    chohatsu: int = 0  # ちょうはつ残りターン
    change_block: bool = False  # にげられない
    meromero: bool = False  # メロメロ
    bind: int = 0  # バインド残りターン
    sub_hp: int = 0  # みがわり残りHP
    fixed_move: str = ""  # こだわっている技
    inaccessible: int = 0  # そらをとぶ等で隠れ中


@dataclass
class ObservedPokemonState:
    """相手ポケモンの観測済み状態（不完全情報）

    観測された情報のみを保持し、未観測の情報は空/None のまま。
    """

    name: str
    hp_ratio: float = 1.0  # 常に観測可能
    ailment: str = ""  # 常に観測可能
    rank: list[int] = field(default_factory=lambda: [0] * 8)  # 常に観測可能
    types: list[str] = field(default_factory=list)  # 現在のタイプ（テラスタル含む）
    terastallized: bool = False  # テラスタルしたか
    tera_type: str = ""  # テラスタル時のみ判明

    # 観測済み情報（累積）
    revealed_moves: list[str] = field(default_factory=list)  # 使用された技
    revealed_item: str | None = None  # 発動/消費で判明した持ち物
    revealed_ability: str | None = None  # 発動で判明した特性

    # 状態変化（これらは観測可能）
    confusion: int = 0
    critical_rank: int = 0
    aquaring: bool = False
    healblock: int = 0
    magnetrise: int = 0
    noroi: bool = False
    horobi: int = 0
    yadorigi: bool = False
    encore: int = 0
    chohatsu: int = 0
    change_block: bool = False
    meromero: bool = False
    bind: int = 0
    sub_hp: int = 0
    inaccessible: int = 0

    # このポケモンが場に出たことがあるか
    is_revealed: bool = False  # 場に出るまで False


@dataclass
class ObservationTracker:
    """相手チームの観測情報を追跡

    バトル中に得られた情報を累積的に記録する。
    """

    # 相手の6匹それぞれの観測状態
    # キー: ポケモン名, 値: 観測情報
    pokemon_observations: dict[str, ObservedPokemonState] = field(default_factory=dict)

    # 相手の選出で判明したポケモン（場に出た順）
    revealed_selection: list[str] = field(default_factory=list)

    def get_or_create(self, name: str) -> ObservedPokemonState:
        """ポケモンの観測状態を取得または作成"""
        if name not in self.pokemon_observations:
            self.pokemon_observations[name] = ObservedPokemonState(name=name)
        return self.pokemon_observations[name]

    def reveal_pokemon(self, name: str) -> None:
        """ポケモンが場に出たことを記録"""
        obs = self.get_or_create(name)
        obs.is_revealed = True
        if name not in self.revealed_selection:
            self.revealed_selection.append(name)

    def reveal_move(self, name: str, move: str) -> None:
        """技が使用されたことを記録"""
        obs = self.get_or_create(name)
        if move and move not in obs.revealed_moves:
            obs.revealed_moves.append(move)

    def reveal_item(self, name: str, item: str) -> None:
        """持ち物が判明したことを記録"""
        obs = self.get_or_create(name)
        if item:
            obs.revealed_item = item

    def reveal_ability(self, name: str, ability: str) -> None:
        """特性が判明したことを記録"""
        obs = self.get_or_create(name)
        if ability:
            obs.revealed_ability = ability

    def reveal_tera(self, name: str, tera_type: str) -> None:
        """テラスタルしたことを記録"""
        obs = self.get_or_create(name)
        obs.terastallized = True
        obs.tera_type = tera_type

    def update_battle_state(
        self,
        name: str,
        hp_ratio: float,
        ailment: str = "",
        rank: list[int] | None = None,
        types: list[str] | None = None,
    ) -> None:
        """バトル中に観測可能な状態を更新"""
        obs = self.get_or_create(name)
        obs.hp_ratio = hp_ratio
        obs.ailment = ailment
        if rank is not None:
            obs.rank = rank
        if types is not None:
            obs.types = types

    def get_observed_state(self, name: str) -> ObservedPokemonState:
        """ポケモンの観測状態を取得"""
        return self.get_or_create(name)

    def to_dict(self) -> dict:
        """シリアライズ用"""
        return {
            "pokemon_observations": {
                name: asdict(obs) for name, obs in self.pokemon_observations.items()
            },
            "revealed_selection": self.revealed_selection,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ObservationTracker":
        """デシリアライズ"""
        tracker = cls()
        for name, obs_dict in data.get("pokemon_observations", {}).items():
            tracker.pokemon_observations[name] = ObservedPokemonState(**obs_dict)
        tracker.revealed_selection = data.get("revealed_selection", [])
        return tracker


@dataclass
class FieldState:
    """フィールドの状態"""

    # 天候
    weather: str = ""  # sunny, rainy, snow, sandstorm
    weather_turns: int = 0

    # フィールド
    terrain: str = ""  # electric, grass, psychic, mist
    terrain_turns: int = 0

    # その他
    trick_room: int = 0
    gravity: int = 0

    # 壁・設置物 (player 0, player 1)
    reflector: tuple[int, int] = (0, 0)
    light_screen: tuple[int, int] = (0, 0)
    tailwind: tuple[int, int] = (0, 0)
    stealth_rock: tuple[bool, bool] = (False, False)
    spikes: tuple[int, int] = (0, 0)
    toxic_spikes: tuple[int, int] = (0, 0)
    sticky_web: tuple[bool, bool] = (False, False)

    # 追加の盤面状態
    safeguard: tuple[int, int] = (0, 0)  # しんぴのまもり残りターン
    white_mist: tuple[int, int] = (0, 0)  # しろいきり残りターン
    wish: tuple[int, int] = (0, 0)  # ねがいごと残りターン


@dataclass
class TurnState:
    """ターンの状態（不完全情報を考慮）"""

    turn: int
    player: int  # 視点 (0 or 1)

    # 自分のポケモン状態（完全情報）
    my_active: PokemonState
    my_bench: list[PokemonState]

    # 相手のポケモン状態（観測済み情報のみ）
    opp_active: ObservedPokemonState
    opp_bench: list[ObservedPokemonState]  # 場に出たことがあるポケモンのみ

    # 相手の未公開ポケモン数（6 - len(revealed_selection)）
    opp_unrevealed_count: int = 0

    # フィールド
    field_state: FieldState = field(default_factory=FieldState)

    # 利用可能な行動
    available_actions: list[int] = field(default_factory=list)


@dataclass
class TurnRecord:
    """ターンの記録"""

    state: TurnState
    action: int  # 選択した行動ID
    action_name: str = ""  # 行動名（デバッグ用）
    reward: float = 0.0  # 即時報酬（通常は0、勝敗時に1/-1）


@dataclass
class BattleTrajectory:
    """バトル軌跡（1試合分）"""

    game_id: str

    # チームプレビュー
    player0_team: list[dict[str, Any]]  # 6匹のデータ
    player1_team: list[dict[str, Any]]

    # 選出
    player0_selection: list[int]  # 選出した3匹のインデックス（先頭が先発）
    player1_selection: list[int]

    # バトル履歴（各プレイヤー視点）
    player0_turns: list[TurnRecord] = field(default_factory=list)
    player1_turns: list[TurnRecord] = field(default_factory=list)

    # 観測履歴（各プレイヤー視点での相手の情報）
    player0_observations: ObservationTracker = field(default_factory=ObservationTracker)
    player1_observations: ObservationTracker = field(default_factory=ObservationTracker)

    # 結果
    winner: Optional[int] = None  # 0, 1, or None (ongoing)
    total_turns: int = 0

    def compute_returns(self, player: int, gamma: float = 1.0) -> list[float]:
        """
        指定プレイヤー視点の Return-to-go を計算

        Args:
            player: プレイヤー番号 (0 or 1)
            gamma: 割引率

        Returns:
            各ターンの Return-to-go リスト
        """
        turns = self.player0_turns if player == 0 else self.player1_turns

        if not turns:
            return []

        # 最終報酬
        if self.winner == player:
            final_reward = 1.0
        elif self.winner is not None:
            final_reward = 0.0
        else:
            final_reward = 0.5  # 引き分け or 未決着

        # Return-to-go を逆順に計算
        returns = []
        rtg = final_reward
        for turn in reversed(turns):
            rtg = turn.reward + gamma * rtg
            returns.append(rtg)

        returns.reverse()
        return returns

    def to_dict(self) -> dict[str, Any]:
        """辞書に変換（JSONL保存用）"""
        return {
            "game_id": self.game_id,
            "player0_team": self.player0_team,
            "player1_team": self.player1_team,
            "player0_selection": self.player0_selection,
            "player1_selection": self.player1_selection,
            "player0_turns": [
                {
                    "state": asdict(t.state),
                    "action": t.action,
                    "action_name": t.action_name,
                    "reward": t.reward,
                }
                for t in self.player0_turns
            ],
            "player1_turns": [
                {
                    "state": asdict(t.state),
                    "action": t.action,
                    "action_name": t.action_name,
                    "reward": t.reward,
                }
                for t in self.player1_turns
            ],
            "player0_observations": self.player0_observations.to_dict(),
            "player1_observations": self.player1_observations.to_dict(),
            "winner": self.winner,
            "total_turns": self.total_turns,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BattleTrajectory":
        """辞書から復元"""

        def parse_turn_record(d: dict) -> TurnRecord:
            state_dict = d["state"]
            state = TurnState(
                turn=state_dict["turn"],
                player=state_dict["player"],
                my_active=PokemonState(**state_dict["my_active"]),
                my_bench=[PokemonState(**p) for p in state_dict["my_bench"]],
                opp_active=ObservedPokemonState(**state_dict["opp_active"]),
                opp_bench=[ObservedPokemonState(**p) for p in state_dict["opp_bench"]],
                opp_unrevealed_count=state_dict.get("opp_unrevealed_count", 0),
                field_state=FieldState(**state_dict.get("field_state", state_dict.get("field", {}))),
                available_actions=state_dict.get("available_actions", []),
            )
            return TurnRecord(
                state=state,
                action=d["action"],
                action_name=d.get("action_name", ""),
                reward=d.get("reward", 0.0),
            )

        # ObservationTracker をパース
        p0_obs_data = data.get("player0_observations", {})
        p1_obs_data = data.get("player1_observations", {})

        return cls(
            game_id=data["game_id"],
            player0_team=data["player0_team"],
            player1_team=data["player1_team"],
            player0_selection=data["player0_selection"],
            player1_selection=data["player1_selection"],
            player0_turns=[parse_turn_record(t) for t in data.get("player0_turns", [])],
            player1_turns=[parse_turn_record(t) for t in data.get("player1_turns", [])],
            player0_observations=ObservationTracker.from_dict(p0_obs_data),
            player1_observations=ObservationTracker.from_dict(p1_obs_data),
            winner=data.get("winner"),
            total_turns=data.get("total_turns", 0),
        )


def save_trajectories_to_jsonl(
    trajectories: list[BattleTrajectory],
    path: Path,
    append: bool = False,
) -> None:
    """軌跡をJSONLファイルに保存"""
    mode = "a" if append else "w"
    with open(path, mode, encoding="utf-8") as f:
        for traj in trajectories:
            f.write(json.dumps(traj.to_dict(), ensure_ascii=False) + "\n")


def load_trajectories_from_jsonl(path: Path) -> list[BattleTrajectory]:
    """JSONLファイルから軌跡を読み込み"""
    trajectories = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data = json.loads(line)
                trajectories.append(BattleTrajectory.from_dict(data))
    return trajectories


class TrajectoryPool:
    """
    軌跡データプール

    AlphaZero スタイルの自己対戦ループで使用。
    最新のデータを優先しつつ、サイズ制限を維持。
    """

    def __init__(self, max_size: int = 100000):
        self.max_size = max_size
        self.trajectories: deque[BattleTrajectory] = deque(maxlen=max_size)

        # 統計
        self.total_added = 0
        self.wins_player0 = 0
        self.wins_player1 = 0
        self.draws = 0

    def add(self, trajectories: list[BattleTrajectory]) -> None:
        """軌跡を追加"""
        for traj in trajectories:
            self.trajectories.append(traj)
            self.total_added += 1

            if traj.winner == 0:
                self.wins_player0 += 1
            elif traj.winner == 1:
                self.wins_player1 += 1
            else:
                self.draws += 1

    def sample(self, n: int) -> list[BattleTrajectory]:
        """ランダムにサンプリング"""
        if n >= len(self.trajectories):
            return list(self.trajectories)
        return random.sample(list(self.trajectories), n)

    def sample_balanced(self, n: int) -> list[BattleTrajectory]:
        """勝敗バランスを取ってサンプリング"""
        wins_p0 = [t for t in self.trajectories if t.winner == 0]
        wins_p1 = [t for t in self.trajectories if t.winner == 1]
        others = [t for t in self.trajectories if t.winner not in (0, 1)]

        # 各カテゴリから均等にサンプリング
        per_category = n // 3
        result = []

        if wins_p0:
            result.extend(random.sample(wins_p0, min(per_category, len(wins_p0))))
        if wins_p1:
            result.extend(random.sample(wins_p1, min(per_category, len(wins_p1))))
        if others:
            result.extend(random.sample(others, min(per_category, len(others))))

        # 不足分を全体から補充
        remaining = n - len(result)
        if remaining > 0 and len(self.trajectories) > len(result):
            pool = [t for t in self.trajectories if t not in result]
            result.extend(random.sample(pool, min(remaining, len(pool))))

        random.shuffle(result)
        return result

    def __len__(self) -> int:
        return len(self.trajectories)

    def stats(self) -> dict[str, Any]:
        """統計情報を取得"""
        return {
            "size": len(self.trajectories),
            "max_size": self.max_size,
            "total_added": self.total_added,
            "wins_player0": self.wins_player0,
            "wins_player1": self.wins_player1,
            "draws": self.draws,
        }


class BattleTrajectoryDataset(Dataset):
    """
    バトル軌跡の PyTorch Dataset

    各サンプルは (trajectory, player, turn_index) のトリプル。
    """

    def __init__(
        self,
        trajectories: list[BattleTrajectory] | Path,
        tokenizer: BattleSequenceTokenizer,
        config: PokemonBattleTransformerConfig,
        max_length: int = 256,
        include_selection: bool = True,
        augment_rtg: bool = True,
    ):
        """
        Args:
            trajectories: 軌跡リストまたはJSONLファイルパス
            tokenizer: BattleSequenceTokenizer
            config: モデル設定
            max_length: 最大系列長
            include_selection: 選出フェーズを含めるか
            augment_rtg: RTG をランダムに変化させるか（データ拡張）
        """
        self.tokenizer = tokenizer
        self.config = config
        self.max_length = max_length
        self.include_selection = include_selection
        self.augment_rtg = augment_rtg

        # 軌跡をロード
        if isinstance(trajectories, Path):
            self.trajectories = load_trajectories_from_jsonl(trajectories)
        else:
            self.trajectories = trajectories

        # サンプルインデックスを構築
        # 各 (trajectory_idx, player, turn_idx) のリスト
        self.samples = []
        for traj_idx, traj in enumerate(self.trajectories):
            # Player 0 視点
            for turn_idx in range(len(traj.player0_turns)):
                self.samples.append((traj_idx, 0, turn_idx))
            # Player 1 視点
            for turn_idx in range(len(traj.player1_turns)):
                self.samples.append((traj_idx, 1, turn_idx))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        traj_idx, player, turn_idx = self.samples[idx]
        traj = self.trajectories[traj_idx]

        # RTG を計算
        returns = traj.compute_returns(player)
        if turn_idx < len(returns):
            target_rtg = returns[turn_idx]
        else:
            target_rtg = 1.0 if traj.winner == player else 0.0

        # RTG 拡張
        if self.augment_rtg and random.random() < 0.3:
            target_rtg = random.choice([0.0, 0.5, 1.0])

        # チーム情報
        if player == 0:
            my_team = [p.get("name", "") for p in traj.player0_team]
            opp_team = [p.get("name", "") for p in traj.player1_team]
            my_selection = traj.player0_selection
            turns = traj.player0_turns
        else:
            my_team = [p.get("name", "") for p in traj.player1_team]
            opp_team = [p.get("name", "") for p in traj.player0_team]
            my_selection = traj.player1_selection
            turns = traj.player1_turns

        # チームプレビューをエンコード
        context = self.tokenizer.encode_team_preview(my_team, opp_team, rtg=target_rtg)

        # 選出をエンコード
        if self.include_selection and my_selection:
            lead_index = my_selection[0]
            context = self.tokenizer.encode_selection(
                my_selection, lead_index, context, rtg=target_rtg
            )

        # ターン履歴をエンコード（turn_idx まで）
        # 注: 完全な実装には各ターンの状態をエンコードする必要がある
        # ここでは簡略化のため、ターン数のみを記録

        # ラベル
        current_turn = turns[turn_idx] if turn_idx < len(turns) else None
        action_label = current_turn.action if current_turn else 0
        # SKIP (-1) などの無効なアクションは 0 に置き換え
        if action_label < 0 or action_label >= self.config.num_action_outputs:
            action_label = 0
        value_label = 1.0 if traj.winner == player else 0.0

        # 選出ラベル
        selection_labels = torch.zeros(6, dtype=torch.long)
        if my_selection:
            lead_index = my_selection[0]
            for i, idx in enumerate(my_selection):
                if idx == lead_index:
                    selection_labels[idx] = 2  # LEAD
                else:
                    selection_labels[idx] = 1  # SELECTED

        # 行動マスク
        action_mask = torch.zeros(self.config.num_action_outputs, dtype=torch.float)
        if current_turn and current_turn.state.available_actions:
            for cmd in current_turn.state.available_actions:
                if 0 <= cmd < self.config.num_action_outputs:
                    action_mask[cmd] = 1.0

        # パディング
        seq_len = len(context["input_ids"])
        if seq_len < self.max_length:
            pad_len = self.max_length - seq_len
            context["input_ids"] = torch.cat([
                context["input_ids"],
                torch.zeros(pad_len, dtype=torch.long),
            ])
            context["position_ids"] = torch.cat([
                context["position_ids"],
                torch.zeros(pad_len, dtype=torch.long),
            ])
            context["timestep_ids"] = torch.cat([
                context["timestep_ids"],
                torch.zeros(pad_len, dtype=torch.long),
            ])
            context["segment_ids"] = torch.cat([
                context["segment_ids"],
                torch.zeros(pad_len, dtype=torch.long),
            ])
            context["rtg_values"] = torch.cat([
                context["rtg_values"],
                torch.zeros(pad_len, dtype=torch.float),
            ])
            context["attention_mask"] = torch.cat([
                context["attention_mask"],
                torch.zeros(pad_len, dtype=torch.long),
            ])
        elif seq_len > self.max_length:
            # 切り詰め
            for key in ["input_ids", "position_ids", "timestep_ids", "segment_ids", "rtg_values", "attention_mask"]:
                context[key] = context[key][:self.max_length]

        return {
            "input_ids": context["input_ids"],
            "position_ids": context["position_ids"],
            "timestep_ids": context["timestep_ids"],
            "segment_ids": context["segment_ids"],
            "rtg_values": context["rtg_values"],
            "attention_mask": context["attention_mask"],
            "team_token_positions": context.get("team_token_positions", torch.zeros(6, dtype=torch.long)),
            "selection_labels": selection_labels,
            "action_label": torch.tensor(action_label, dtype=torch.long),
            "value_label": torch.tensor(value_label, dtype=torch.float),
            "action_mask": action_mask,
        }


def collate_fn(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """バッチのコレート関数"""
    result = {}
    for key in batch[0].keys():
        tensors = [item[key] for item in batch]
        result[key] = torch.stack(tensors)
    return result
