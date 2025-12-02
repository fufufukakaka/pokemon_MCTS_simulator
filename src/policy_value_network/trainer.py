"""
Policy-Value Network Trainer

Policy-Value Networkの学習を行うトレーナークラス。
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from .dataset import SelfPlayDataset, SelfPlayDatasetInMemory
from .network import PolicyValueNetwork
from .observation_encoder import ObservationEncoder

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """学習設定"""

    # モデル設定
    hidden_dim: int = 256
    num_res_blocks: int = 4
    dropout: float = 0.1

    # 学習設定
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    num_epochs: int = 100
    early_stopping_patience: int = 10

    # 損失関数の重み
    policy_loss_weight: float = 1.0
    value_loss_weight: float = 1.0

    # データ分割
    val_ratio: float = 0.1

    # その他
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers: int = 0
    save_every_n_epochs: int = 10
    log_every_n_steps: int = 50


class PolicyValueTrainer:
    """
    Policy-Value Network トレーナー

    Self-Playデータを使ってPolicy-Value Networkを学習する。
    """

    def __init__(
        self,
        config: Optional[TrainingConfig] = None,
        model: Optional[PolicyValueNetwork] = None,
        encoder: Optional[ObservationEncoder] = None,
    ):
        self.config = config or TrainingConfig()
        self.encoder = encoder or ObservationEncoder()
        self.device = torch.device(self.config.device)

        # モデル（後で初期化）
        self.model = model
        self.optimizer: Optional[optim.Optimizer] = None
        self.scheduler: Optional[optim.lr_scheduler._LRScheduler] = None

        # 学習履歴
        self.history: dict[str, list[float]] = {
            "train_loss": [],
            "train_policy_loss": [],
            "train_value_loss": [],
            "val_loss": [],
            "val_policy_loss": [],
            "val_value_loss": [],
        }

    def _init_model(self, input_dim: int, num_actions: int):
        """モデル初期化"""
        self.model = PolicyValueNetwork(
            input_dim=input_dim,
            hidden_dim=self.config.hidden_dim,
            num_res_blocks=self.config.num_res_blocks,
            num_actions=num_actions,
            dropout=self.config.dropout,
        ).to(self.device)

        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.config.num_epochs
        )

    def train(
        self,
        dataset_path: str | Path,
        output_dir: str | Path,
        max_actions: int = 50,
    ) -> dict[str, list[float]]:
        """
        学習実行

        Args:
            dataset_path: Self-Playデータセットのパス
            output_dir: 出力ディレクトリ
            max_actions: 行動数の上限

        Returns:
            学習履歴
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Loading dataset from {dataset_path}")

        # データセット読み込み
        dataset = SelfPlayDatasetInMemory(
            records_path=dataset_path,
            encoder=self.encoder,
            max_actions=max_actions,
        )

        logger.info(f"Dataset size: {len(dataset)}")
        logger.info(f"Number of actions: {len(dataset.action_to_id)}")

        # 行動数を更新
        num_actions = min(max_actions, len(dataset.action_to_id) + 1)

        # データ分割
        val_size = int(len(dataset) * self.config.val_ratio)
        train_size = len(dataset) - val_size

        train_dataset, val_dataset = random_split(
            dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
        )

        # モデル初期化
        input_dim = self.encoder.get_flat_dim()
        logger.info(f"Input dimension: {input_dim}")

        if self.model is None:
            self._init_model(input_dim, num_actions)

        # 損失関数
        # Policy: KL Divergence（ソフトターゲット用）
        policy_criterion = nn.KLDivLoss(reduction="batchmean")
        value_criterion = nn.MSELoss()

        best_val_loss = float("inf")
        patience_counter = 0

        logger.info("Starting training...")

        for epoch in range(self.config.num_epochs):
            # 学習
            train_metrics = self._train_epoch(
                train_loader, policy_criterion, value_criterion
            )

            # 検証
            val_metrics = self._validate(val_loader, policy_criterion, value_criterion)

            # 履歴に追加
            self.history["train_loss"].append(train_metrics["loss"])
            self.history["train_policy_loss"].append(train_metrics["policy_loss"])
            self.history["train_value_loss"].append(train_metrics["value_loss"])
            self.history["val_loss"].append(val_metrics["loss"])
            self.history["val_policy_loss"].append(val_metrics["policy_loss"])
            self.history["val_value_loss"].append(val_metrics["value_loss"])

            # スケジューラー更新
            if self.scheduler:
                self.scheduler.step()

            # ログ出力
            logger.info(
                f"Epoch {epoch + 1}/{self.config.num_epochs} - "
                f"Train Loss: {train_metrics['loss']:.4f} "
                f"(Policy: {train_metrics['policy_loss']:.4f}, Value: {train_metrics['value_loss']:.4f}) - "
                f"Val Loss: {val_metrics['loss']:.4f} "
                f"(Policy: {val_metrics['policy_loss']:.4f}, Value: {val_metrics['value_loss']:.4f})"
            )

            # チェックポイント保存
            if (epoch + 1) % self.config.save_every_n_epochs == 0:
                self._save_checkpoint(output_dir / f"checkpoint_epoch{epoch + 1}.pt")

            # 早期終了判定
            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                patience_counter = 0
                self._save_checkpoint(output_dir / "best_model.pt")
            else:
                patience_counter += 1
                if patience_counter >= self.config.early_stopping_patience:
                    logger.info(f"Early stopping at epoch {epoch + 1}")
                    break

        # 最終モデル保存
        self._save_checkpoint(output_dir / "final_model.pt")

        # エンコーダー保存
        self.encoder.save(output_dir / "encoder.json")

        # 行動辞書保存
        with open(output_dir / "action_vocab.json", "w", encoding="utf-8") as f:
            json.dump(dataset.action_to_id, f, ensure_ascii=False, indent=2)

        # 履歴保存
        with open(output_dir / "history.json", "w") as f:
            json.dump(self.history, f, indent=2)

        # 設定保存（max_actionsとinput_dimも含める）
        config_to_save = {
            **self.config.__dict__,
            "max_actions": max_actions,
            "num_actions": num_actions,
            "input_dim": input_dim,
        }
        with open(output_dir / "config.json", "w") as f:
            json.dump(config_to_save, f, indent=2)

        logger.info(f"Training completed. Models saved to {output_dir}")

        return self.history

    def _train_epoch(
        self,
        train_loader: DataLoader,
        policy_criterion: nn.Module,
        value_criterion: nn.Module,
    ) -> dict[str, float]:
        """1エポックの学習"""
        self.model.train()

        total_loss = 0.0
        total_policy_loss = 0.0
        total_value_loss = 0.0
        num_batches = 0

        for batch in train_loader:
            features = batch["features"].to(self.device)
            policy_target = batch["policy_target"].to(self.device)
            value_target = batch["value_target"].to(self.device)
            action_mask = batch["action_mask"].to(self.device)

            self.optimizer.zero_grad()

            # 順伝播
            policy_logits = self.model.get_policy_logits(features, action_mask)
            _, value_pred = self.model(features, action_mask)

            # Policy Loss（Cross Entropy with soft targets）
            # マスクされていない部分のみでsoftmaxを計算
            # -infをマスクしてからsoftmaxを計算し直す
            policy_logits_masked = policy_logits.clone()
            policy_logits_masked[action_mask == 0] = -1e9  # -infではなく大きな負の値
            log_policy = torch.nn.functional.log_softmax(policy_logits_masked, dim=-1)

            # KL divergenceの代わりにCross Entropyを使用
            # CE = -sum(target * log(pred))
            policy_loss = -(policy_target * log_policy).sum(dim=-1).mean()

            # Value Loss（MSE）
            value_loss = value_criterion(value_pred, value_target)

            # 合計損失
            loss = (
                self.config.policy_loss_weight * policy_loss
                + self.config.value_loss_weight * value_loss
            )

            # 逆伝播
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            num_batches += 1

        return {
            "loss": total_loss / num_batches,
            "policy_loss": total_policy_loss / num_batches,
            "value_loss": total_value_loss / num_batches,
        }

    def _validate(
        self,
        val_loader: DataLoader,
        policy_criterion: nn.Module,
        value_criterion: nn.Module,
    ) -> dict[str, float]:
        """検証"""
        self.model.eval()

        total_loss = 0.0
        total_policy_loss = 0.0
        total_value_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                features = batch["features"].to(self.device)
                policy_target = batch["policy_target"].to(self.device)
                value_target = batch["value_target"].to(self.device)
                action_mask = batch["action_mask"].to(self.device)

                # 順伝播
                policy_logits = self.model.get_policy_logits(features, action_mask)
                _, value_pred = self.model(features, action_mask)

                # Policy Loss（Cross Entropy with soft targets）
                policy_logits_masked = policy_logits.clone()
                policy_logits_masked[action_mask == 0] = -1e9
                log_policy = torch.nn.functional.log_softmax(policy_logits_masked, dim=-1)
                policy_loss = -(policy_target * log_policy).sum(dim=-1).mean()

                # Value Loss
                value_loss = value_criterion(value_pred, value_target)

                # 合計損失
                loss = (
                    self.config.policy_loss_weight * policy_loss
                    + self.config.value_loss_weight * value_loss
                )

                total_loss += loss.item()
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                num_batches += 1

        return {
            "loss": total_loss / num_batches,
            "policy_loss": total_policy_loss / num_batches,
            "value_loss": total_value_loss / num_batches,
        }

    def _save_checkpoint(self, path: Path):
        """チェックポイント保存"""
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": (
                    self.scheduler.state_dict() if self.scheduler else None
                ),
                "config": self.config.__dict__,
            },
            path,
        )

    def load_checkpoint(self, path: Path):
        """チェックポイント読み込み"""
        checkpoint = torch.load(path, map_location=self.device)

        if self.model is None:
            raise ValueError("Model must be initialized before loading checkpoint")

        self.model.load_state_dict(checkpoint["model_state_dict"])

        if self.optimizer and "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if self.scheduler and checkpoint.get("scheduler_state_dict"):
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    @classmethod
    def load_trained_model(
        cls, model_dir: str | Path, device: str = "cpu"
    ) -> tuple["PolicyValueTrainer", dict[str, int]]:
        """
        学習済みモデルを読み込み

        Args:
            model_dir: モデルディレクトリ
            device: デバイス

        Returns:
            (trainer, action_vocab)
        """
        model_dir = Path(model_dir)

        # 設定読み込み
        with open(model_dir / "config.json", "r") as f:
            config_dict = json.load(f)
        config = TrainingConfig(**config_dict)
        config.device = device

        # エンコーダー読み込み
        encoder = ObservationEncoder.load(model_dir / "encoder.json")

        # 行動辞書読み込み
        with open(model_dir / "action_vocab.json", "r", encoding="utf-8") as f:
            action_vocab = json.load(f)

        # モデル初期化
        trainer = cls(config=config, encoder=encoder)
        input_dim = encoder.get_flat_dim()
        num_actions = len(action_vocab) + 1

        trainer._init_model(input_dim, num_actions)

        # 重み読み込み
        checkpoint = torch.load(model_dir / "best_model.pt", map_location=device)
        trainer.model.load_state_dict(checkpoint["model_state_dict"])
        trainer.model.eval()

        return trainer, action_vocab
