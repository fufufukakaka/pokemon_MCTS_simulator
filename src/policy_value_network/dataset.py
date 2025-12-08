"""
Self-Play Dataset

Self-Playで生成されたデータをPyTorchのDatasetとして提供する。
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset

from src.hypothesis.selfplay import TurnRecord, load_records_from_jsonl

from .observation_encoder import ObservationEncoder


class SelfPlayDataset(Dataset):
    """
    Self-Playデータセット

    TurnRecordのリストをPyTorch Datasetとして提供する。
    """

    def __init__(
        self,
        records_path: str | Path,
        encoder: Optional[ObservationEncoder] = None,
        action_to_id: Optional[dict[str, int]] = None,
        max_actions: int = 10,
    ):
        """
        Args:
            records_path: JOSNLファイルのパス
            encoder: ObservationEncoder（Noneなら新規作成）
            action_to_id: 行動文字列→IDの変換辞書
            max_actions: 行動数の上限
        """
        self.records_path = Path(records_path)
        self.encoder = encoder or ObservationEncoder()
        self.max_actions = max_actions

        # 行動→ID変換
        self.action_to_id = action_to_id or {}
        self._next_action_id = len(self.action_to_id)

        # データ読み込み
        self.game_records = load_records_from_jsonl(records_path)

        # すべてのTurnRecordをフラット化
        self.turn_records: list[TurnRecord] = []
        for game in self.game_records:
            self.turn_records.extend(game.turns)

        # 行動IDを構築
        self._build_action_ids()

    def _build_action_ids(self):
        """行動IDを構築"""
        for turn in self.turn_records:
            for action_str in turn.policy.keys():
                if action_str not in self.action_to_id:
                    self.action_to_id[action_str] = self._next_action_id
                    self._next_action_id += 1

    def _get_action_id(self, action_str: str) -> int:
        """行動文字列からIDを取得"""
        if action_str not in self.action_to_id:
            self.action_to_id[action_str] = self._next_action_id
            self._next_action_id += 1
        return self.action_to_id[action_str]

    def __len__(self) -> int:
        return len(self.turn_records)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """
        データ取得

        Returns:
            dict with:
            - features: [D] 盤面特徴量
            - policy_target: [max_actions] ターゲットPolicy分布
            - value_target: [1] ターゲット勝率
            - action_mask: [max_actions] 有効な行動のマスク
        """
        turn = self.turn_records[idx]

        # 盤面をエンコード
        features = self.encoder.encode_flat(turn)

        # Policy target
        policy_target = torch.zeros(self.max_actions, dtype=torch.float32)
        action_mask = torch.zeros(self.max_actions, dtype=torch.float32)

        for action_str, prob in turn.policy.items():
            action_id = self._get_action_id(action_str)
            if action_id < self.max_actions:
                policy_target[action_id] = prob
                action_mask[action_id] = 1.0

        # 正規化（有効な行動の確率が1になるように）
        if action_mask.sum() > 0:
            policy_target = policy_target / (policy_target.sum() + 1e-8)

        # Value target
        value_target = torch.tensor([turn.value], dtype=torch.float32)

        return {
            "features": features,
            "policy_target": policy_target,
            "value_target": value_target,
            "action_mask": action_mask,
        }

    def get_action_vocab(self) -> dict[str, int]:
        """行動→ID変換辞書を取得"""
        return self.action_to_id.copy()

    def get_id_to_action(self) -> dict[int, str]:
        """ID→行動変換辞書を取得"""
        return {v: k for k, v in self.action_to_id.items()}


class SelfPlayDatasetInMemory(Dataset):
    """
    メモリ上にすべてのデータを保持するデータセット

    高速だがメモリを多く消費する。
    """

    def __init__(
        self,
        records_path: str | Path,
        encoder: Optional[ObservationEncoder] = None,
        max_actions: int = 10,
    ):
        self.encoder = encoder or ObservationEncoder()
        self.max_actions = max_actions

        # 行動→ID変換
        self.action_to_id: dict[str, int] = {}
        self._next_action_id = 0

        # データ読み込み
        game_records = load_records_from_jsonl(records_path)

        # すべてのTurnRecordをフラット化
        turn_records: list[TurnRecord] = []
        for game in game_records:
            turn_records.extend(game.turns)

        # まず行動IDを構築
        for turn in turn_records:
            for action_str in turn.policy.keys():
                if action_str not in self.action_to_id:
                    self.action_to_id[action_str] = self._next_action_id
                    self._next_action_id += 1

        # データをテンソルに変換
        self.features_list: list[torch.Tensor] = []
        self.policy_targets: list[torch.Tensor] = []
        self.value_targets: list[torch.Tensor] = []
        self.action_masks: list[torch.Tensor] = []

        for turn in turn_records:
            # 盤面をエンコード
            features = self.encoder.encode_flat(turn)
            self.features_list.append(features)

            # Policy target
            policy_target = torch.zeros(max_actions, dtype=torch.float32)
            action_mask = torch.zeros(max_actions, dtype=torch.float32)

            for action_str, prob in turn.policy.items():
                action_id = self.action_to_id[action_str]
                if action_id < max_actions:
                    policy_target[action_id] = prob
                    action_mask[action_id] = 1.0

            if action_mask.sum() > 0:
                policy_target = policy_target / (policy_target.sum() + 1e-8)

            self.policy_targets.append(policy_target)
            self.action_masks.append(action_mask)

            # Value target
            self.value_targets.append(torch.tensor([turn.value], dtype=torch.float32))

        # スタック
        self.features = torch.stack(self.features_list)
        self.policy_targets_tensor = torch.stack(self.policy_targets)
        self.value_targets_tensor = torch.stack(self.value_targets)
        self.action_masks_tensor = torch.stack(self.action_masks)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "features": self.features[idx],
            "policy_target": self.policy_targets_tensor[idx],
            "value_target": self.value_targets_tensor[idx],
            "action_mask": self.action_masks_tensor[idx],
        }

    def get_action_vocab(self) -> dict[str, int]:
        return self.action_to_id.copy()
