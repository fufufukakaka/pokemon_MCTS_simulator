"""
Team Selection Trainer

Team Selection Networkの学習を行うトレーナークラス。
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from .team_selection_dataset import TeamSelectionDataset
from .team_selection_encoder import TeamSelectionEncoder
from .team_selection_network import (
    TeamSelectionLoss,
    TeamSelectionNetwork,
    TeamSelectionNetworkConfig,
)

logger = logging.getLogger(__name__)


@dataclass
class TeamSelectionTrainingConfig:
    """学習設定"""

    # モデル設定
    pokemon_embed_dim: int = 128
    hidden_dim: int = 256
    num_heads: int = 4
    num_layers: int = 2
    dropout: float = 0.1

    # 学習設定
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    num_epochs: int = 100
    early_stopping_patience: int = 10

    # 損失関数の重み
    selection_loss_weight: float = 1.0
    value_loss_weight: float = 0.5

    # データ分割
    val_ratio: float = 0.1

    # その他
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers: int = 0
    save_every_n_epochs: int = 10


class TeamSelectionTrainer:
    """
    Team Selection Network トレーナー
    """

    def __init__(
        self,
        config: Optional[TeamSelectionTrainingConfig] = None,
        encoder: Optional[TeamSelectionEncoder] = None,
    ):
        self.config = config or TeamSelectionTrainingConfig()
        self.encoder = encoder or TeamSelectionEncoder()
        self.device = torch.device(self.config.device)

        self.model: Optional[TeamSelectionNetwork] = None
        self.optimizer: Optional[optim.Optimizer] = None
        self.scheduler: Optional[optim.lr_scheduler._LRScheduler] = None

        self.history: dict[str, list[float]] = {
            "train_loss": [],
            "train_selection_loss": [],
            "train_value_loss": [],
            "val_loss": [],
            "val_selection_loss": [],
            "val_value_loss": [],
        }

    def _init_model(self):
        """モデル初期化"""
        network_config = TeamSelectionNetworkConfig(
            pokemon_feature_dim=self.encoder.get_pokemon_feature_dim(),
            pokemon_embed_dim=self.config.pokemon_embed_dim,
            hidden_dim=self.config.hidden_dim,
            num_heads=self.config.num_heads,
            num_layers=self.config.num_layers,
            dropout=self.config.dropout,
        )

        self.model = TeamSelectionNetwork(config=network_config).to(self.device)

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
    ) -> dict[str, list[float]]:
        """
        学習実行

        Args:
            dataset_path: 選出データセットのパス（JSONL）
            output_dir: 出力ディレクトリ

        Returns:
            学習履歴
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Loading dataset from {dataset_path}")

        # データセット読み込み
        dataset = TeamSelectionDataset(
            records_path=dataset_path,
            encoder=self.encoder,
        )

        logger.info(f"Dataset size: {len(dataset)}")

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
        if self.model is None:
            self._init_model()

        # 損失関数
        criterion = TeamSelectionLoss(
            selection_weight=self.config.selection_loss_weight,
            value_weight=self.config.value_loss_weight,
        )

        best_val_loss = float("inf")
        patience_counter = 0

        logger.info("Starting training...")

        for epoch in range(self.config.num_epochs):
            # 学習
            train_metrics = self._train_epoch(train_loader, criterion)

            # 検証
            val_metrics = self._validate(val_loader, criterion)

            # 履歴に追加
            self.history["train_loss"].append(train_metrics["loss"])
            self.history["train_selection_loss"].append(train_metrics["selection_loss"])
            self.history["train_value_loss"].append(train_metrics["value_loss"])
            self.history["val_loss"].append(val_metrics["loss"])
            self.history["val_selection_loss"].append(val_metrics["selection_loss"])
            self.history["val_value_loss"].append(val_metrics["value_loss"])

            # スケジューラー更新
            if self.scheduler:
                self.scheduler.step()

            # ログ出力
            logger.info(
                f"Epoch {epoch + 1}/{self.config.num_epochs} - "
                f"Train Loss: {train_metrics['loss']:.4f} "
                f"(Sel: {train_metrics['selection_loss']:.4f}, Val: {train_metrics['value_loss']:.4f}) - "
                f"Val Loss: {val_metrics['loss']:.4f} "
                f"(Sel: {val_metrics['selection_loss']:.4f}, Val: {val_metrics['value_loss']:.4f})"
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

        # 履歴保存
        with open(output_dir / "history.json", "w") as f:
            json.dump(self.history, f, indent=2)

        # 設定保存
        config_to_save = {
            **self.config.__dict__,
            "pokemon_feature_dim": self.encoder.get_pokemon_feature_dim(),
        }
        with open(output_dir / "config.json", "w") as f:
            json.dump(config_to_save, f, indent=2)

        logger.info(f"Training completed. Models saved to {output_dir}")

        return self.history

    def _train_epoch(
        self,
        train_loader: DataLoader,
        criterion: TeamSelectionLoss,
    ) -> dict[str, float]:
        """1エポックの学習"""
        self.model.train()

        total_loss = 0.0
        total_selection_loss = 0.0
        total_value_loss = 0.0
        num_batches = 0

        for batch in train_loader:
            my_team = batch["my_team"].to(self.device)
            opp_team = batch["opp_team"].to(self.device)
            selection_label = batch["selection_label"].to(self.device)
            outcome = batch["outcome"].to(self.device)

            self.optimizer.zero_grad()

            # 順伝播
            output = self.model(my_team, opp_team)

            # 損失計算
            losses = criterion(
                selection_logits=output["selection_logits"],
                selection_target=selection_label,
                value_pred=output["value"],
                value_target=outcome,
            )

            # 逆伝播
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += losses["loss"].item()
            total_selection_loss += losses["selection_loss"].item()
            total_value_loss += losses["value_loss"].item()
            num_batches += 1

        return {
            "loss": total_loss / num_batches,
            "selection_loss": total_selection_loss / num_batches,
            "value_loss": total_value_loss / num_batches,
        }

    def _validate(
        self,
        val_loader: DataLoader,
        criterion: TeamSelectionLoss,
    ) -> dict[str, float]:
        """検証"""
        self.model.eval()

        total_loss = 0.0
        total_selection_loss = 0.0
        total_value_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                my_team = batch["my_team"].to(self.device)
                opp_team = batch["opp_team"].to(self.device)
                selection_label = batch["selection_label"].to(self.device)
                outcome = batch["outcome"].to(self.device)

                output = self.model(my_team, opp_team)

                losses = criterion(
                    selection_logits=output["selection_logits"],
                    selection_target=selection_label,
                    value_pred=output["value"],
                    value_target=outcome,
                )

                total_loss += losses["loss"].item()
                total_selection_loss += losses["selection_loss"].item()
                total_value_loss += losses["value_loss"].item()
                num_batches += 1

        return {
            "loss": total_loss / num_batches,
            "selection_loss": total_selection_loss / num_batches,
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

    @classmethod
    def load_trained_model(
        cls, model_dir: str | Path, device: str = "cpu"
    ) -> tuple["TeamSelectionTrainer", TeamSelectionEncoder]:
        """
        学習済みモデルを読み込み

        Args:
            model_dir: モデルディレクトリ
            device: デバイス

        Returns:
            (trainer, encoder)
        """
        model_dir = Path(model_dir)

        # 設定読み込み
        with open(model_dir / "config.json", "r") as f:
            config_dict = json.load(f)

        config = TeamSelectionTrainingConfig(
            pokemon_embed_dim=config_dict.get("pokemon_embed_dim", 128),
            hidden_dim=config_dict.get("hidden_dim", 256),
            num_heads=config_dict.get("num_heads", 4),
            num_layers=config_dict.get("num_layers", 2),
            dropout=config_dict.get("dropout", 0.1),
            device=device,
        )

        # エンコーダー読み込み
        encoder = TeamSelectionEncoder.load(model_dir / "encoder.json")

        # トレーナー初期化
        trainer = cls(config=config, encoder=encoder)
        trainer._init_model()

        # 重み読み込み
        checkpoint = torch.load(model_dir / "best_model.pt", map_location=device)
        trainer.model.load_state_dict(checkpoint["model_state_dict"])
        trainer.model.eval()

        return trainer, encoder


def load_team_selection_model(
    model_dir: str | Path, device: str = "cpu"
) -> tuple[TeamSelectionNetwork, TeamSelectionEncoder]:
    """
    学習済みTeam Selection Networkを読み込む便利関数

    Args:
        model_dir: モデルディレクトリ
        device: デバイス

    Returns:
        (model, encoder)
    """
    trainer, encoder = TeamSelectionTrainer.load_trained_model(model_dir, device)
    return trainer.model, encoder
