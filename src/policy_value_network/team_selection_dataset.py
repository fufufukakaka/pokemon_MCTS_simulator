"""
Team Selection Dataset

チーム選出学習用のデータセット。
Self-Playの結果から選出と勝敗のデータを生成する。
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset

from .team_selection_encoder import TeamSelectionEncoder


@dataclass
class TeamSelectionRecord:
    """選出記録"""

    # 自分のチーム（6匹）
    my_team: list[dict]
    # 相手のチーム（6匹）
    opp_team: list[dict]
    # 選出した3匹のインデックス
    selected_indices: list[int]
    # 勝敗（1=勝ち, 0=負け, 0.5=引き分け）
    outcome: float
    # メタデータ
    game_id: str = ""


def generate_selection_data_from_battles(
    trainer_data_path: str | Path,
    num_samples: int = 10000,
    output_path: Optional[str | Path] = None,
) -> list[TeamSelectionRecord]:
    """
    トレーナーデータからランダムな選出データを生成

    注意: 実際のSelf-Playではなく、ランダム選出と
    ランダム勝敗でデータを生成する（初期学習用）。
    本格的な学習にはSelf-Playデータを使用すべき。

    Args:
        trainer_data_path: トレーナーデータのJSONパス
        num_samples: 生成するサンプル数
        output_path: 保存先（指定時）

    Returns:
        TeamSelectionRecordのリスト
    """
    with open(trainer_data_path, "r", encoding="utf-8") as f:
        trainers = json.load(f)

    records = []

    for i in range(num_samples):
        # ランダムに2人のトレーナーを選択
        trainer0, trainer1 = random.sample(trainers, 2)
        team0 = trainer0["pokemons"][:6]
        team1 = trainer1["pokemons"][:6]

        # 6匹未満の場合はスキップ
        if len(team0) < 6 or len(team1) < 6:
            continue

        # ランダムに3匹を選出
        indices0 = random.sample(range(6), 3)
        indices1 = random.sample(range(6), 3)

        # ランダム勝敗（初期データ用、後でSelf-Playで置き換え）
        outcome = random.choice([0.0, 0.5, 1.0])

        # 両方のプレイヤー視点で記録
        records.append(
            TeamSelectionRecord(
                my_team=team0,
                opp_team=team1,
                selected_indices=indices0,
                outcome=outcome,
                game_id=f"random_{i:06d}_p0",
            )
        )
        records.append(
            TeamSelectionRecord(
                my_team=team1,
                opp_team=team0,
                selected_indices=indices1,
                outcome=1.0 - outcome,  # 相手視点
                game_id=f"random_{i:06d}_p1",
            )
        )

    if output_path:
        save_selection_records(records, output_path)

    return records


def save_selection_records(records: list[TeamSelectionRecord], path: str | Path):
    """選出記録をJSONLで保存"""
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            data = {
                "my_team": record.my_team,
                "opp_team": record.opp_team,
                "selected_indices": record.selected_indices,
                "outcome": record.outcome,
                "game_id": record.game_id,
            }
            f.write(json.dumps(data, ensure_ascii=False) + "\n")


def load_selection_records(path: str | Path) -> list[TeamSelectionRecord]:
    """選出記録をJSONLから読み込み"""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line.strip())
            records.append(
                TeamSelectionRecord(
                    my_team=data["my_team"],
                    opp_team=data["opp_team"],
                    selected_indices=data["selected_indices"],
                    outcome=data["outcome"],
                    game_id=data.get("game_id", ""),
                )
            )
    return records


class TeamSelectionDataset(Dataset):
    """
    Team Selection学習用Dataset

    Args:
        records_path: 選出記録のJSONLパス
        encoder: TeamSelectionEncoder（Noneなら新規作成）
    """

    def __init__(
        self,
        records_path: str | Path,
        encoder: Optional[TeamSelectionEncoder] = None,
    ):
        self.encoder = encoder or TeamSelectionEncoder()
        self.records = load_selection_records(records_path)

        # 事前にエンコード
        self._preprocess()

    def _preprocess(self):
        """データを事前処理"""
        self.samples = []

        for record in self.records:
            # チームをエンコード
            my_team_tensor = self.encoder.encode_team(record.my_team)
            opp_team_tensor = self.encoder.encode_team(record.opp_team)

            # 選出ラベル（one-hot的な0/1ベクトル）
            selection_label = torch.zeros(6, dtype=torch.float32)
            for idx in record.selected_indices:
                if 0 <= idx < 6:
                    selection_label[idx] = 1.0

            # 勝敗
            outcome = torch.tensor([record.outcome], dtype=torch.float32)

            self.samples.append(
                {
                    "my_team": my_team_tensor,
                    "opp_team": opp_team_tensor,
                    "selection_label": selection_label,
                    "outcome": outcome,
                }
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return self.samples[idx]


class TeamSelectionDatasetFromSelfPlay(Dataset):
    """
    Self-Playデータから選出データを生成するDataset

    Self-Playで生成されたGameRecordから、
    実際に使用された選出と勝敗を抽出する。
    """

    def __init__(
        self,
        selfplay_records_path: str | Path,
        trainer_data_path: str | Path,
        encoder: Optional[TeamSelectionEncoder] = None,
    ):
        self.encoder = encoder or TeamSelectionEncoder()

        # トレーナーデータ読み込み（名前→チームのマッピング）
        with open(trainer_data_path, "r", encoding="utf-8") as f:
            trainers = json.load(f)
        self.trainer_to_team = {t["name"]: t["pokemons"] for t in trainers}

        # Self-Playデータ読み込み
        self.records = self._load_selfplay_records(selfplay_records_path)

        # 事前処理
        self._preprocess()

    def _load_selfplay_records(self, path: str | Path) -> list[dict]:
        """Self-Play記録を読み込み"""
        records = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line.strip())
                records.append(data)
        return records

    def _preprocess(self):
        """データを事前処理"""
        self.samples = []

        for record in self.records:
            # トレーナー名から元のチーム（6匹）を取得
            trainer0_name = record.get("player0_trainer", "")
            trainer1_name = record.get("player1_trainer", "")

            if trainer0_name not in self.trainer_to_team:
                continue
            if trainer1_name not in self.trainer_to_team:
                continue

            team0_full = self.trainer_to_team[trainer0_name][:6]
            team1_full = self.trainer_to_team[trainer1_name][:6]

            if len(team0_full) < 6 or len(team1_full) < 6:
                continue

            # 選出された3匹を特定
            selected0 = record.get("player0_team", [])
            selected1 = record.get("player1_team", [])

            indices0 = self._find_selection_indices(team0_full, selected0)
            indices1 = self._find_selection_indices(team1_full, selected1)

            if len(indices0) != 3 or len(indices1) != 3:
                continue

            # 勝敗
            winner = record.get("winner")
            if winner == 0:
                outcome0, outcome1 = 1.0, 0.0
            elif winner == 1:
                outcome0, outcome1 = 0.0, 1.0
            else:
                outcome0, outcome1 = 0.5, 0.5

            # Player 0視点
            my_team0 = self.encoder.encode_team(team0_full)
            opp_team0 = self.encoder.encode_team(team1_full)
            selection_label0 = torch.zeros(6, dtype=torch.float32)
            for idx in indices0:
                selection_label0[idx] = 1.0

            self.samples.append(
                {
                    "my_team": my_team0,
                    "opp_team": opp_team0,
                    "selection_label": selection_label0,
                    "outcome": torch.tensor([outcome0], dtype=torch.float32),
                }
            )

            # Player 1視点
            my_team1 = self.encoder.encode_team(team1_full)
            opp_team1 = self.encoder.encode_team(team0_full)
            selection_label1 = torch.zeros(6, dtype=torch.float32)
            for idx in indices1:
                selection_label1[idx] = 1.0

            self.samples.append(
                {
                    "my_team": my_team1,
                    "opp_team": opp_team1,
                    "selection_label": selection_label1,
                    "outcome": torch.tensor([outcome1], dtype=torch.float32),
                }
            )

    def _find_selection_indices(
        self, full_team: list[dict], selected_names: list[str]
    ) -> list[int]:
        """選出されたポケモンのインデックスを特定"""
        indices = []
        for name in selected_names:
            for i, p in enumerate(full_team):
                if p["name"] == name and i not in indices:
                    indices.append(i)
                    break
        return indices

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return self.samples[idx]
