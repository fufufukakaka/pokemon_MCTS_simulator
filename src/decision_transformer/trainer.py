"""
Decision Transformer Trainer

AlphaZero スタイルの自己対戦ループを含む学習パイプライン。
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from .config import PokemonBattleTransformerConfig, TrainingConfig
from .data_generator import GeneratorConfig, TrajectoryGenerator
from .dataset import (
    BattleTrajectory,
    BattleTrajectoryDataset,
    TrajectoryPool,
    collate_fn,
    save_trajectories_to_jsonl,
)
from .model import PokemonBattleTransformer
from .tokenizer import BattleSequenceTokenizer

logger = logging.getLogger(__name__)


class DecisionTransformerTrainer:
    """
    Decision Transformer のトレーナー

    AlphaZero スタイルの自己対戦ループをサポート。
    """

    def __init__(
        self,
        config: TrainingConfig,
        model: PokemonBattleTransformer | None = None,
        tokenizer: BattleSequenceTokenizer | None = None,
    ):
        """
        Args:
            config: 学習設定
            model: モデル（None なら新規作成）
            tokenizer: トークナイザ（None なら新規作成）
        """
        self.config = config
        self.device = torch.device(config.device)

        # モデル
        if model is None:
            model_config = config.model_config or PokemonBattleTransformerConfig()
            self.model = PokemonBattleTransformer(model_config)
        else:
            self.model = model
        self.model.to(self.device)

        # トークナイザ
        if tokenizer is None:
            self.tokenizer = BattleSequenceTokenizer(self.model.config)
        else:
            self.tokenizer = tokenizer

        # オプティマイザ
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        # スケジューラ
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.num_iterations * config.training_epochs_per_iteration,
        )

        # 学習履歴
        self.training_history: list[dict[str, Any]] = []
        self.current_iteration = 0

        # データプール
        self.trajectory_pool = TrajectoryPool(max_size=config.trajectory_pool_size)

    def compute_loss(
        self,
        batch: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """
        損失を計算

        Args:
            batch: バッチデータ

        Returns:
            (total_loss, {selection_loss, action_loss, value_loss})
        """
        # デバイスに移動
        input_ids = batch["input_ids"].to(self.device)
        position_ids = batch["position_ids"].to(self.device)
        timestep_ids = batch["timestep_ids"].to(self.device)
        segment_ids = batch["segment_ids"].to(self.device)
        rtg_values = batch["rtg_values"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        team_token_positions = batch["team_token_positions"].to(self.device)
        selection_labels = batch["selection_labels"].to(self.device)
        action_labels = batch["action_label"].to(self.device)
        value_labels = batch["value_label"].to(self.device)
        action_mask = batch["action_mask"].to(self.device)

        # state_features があれば使用
        state_features = None
        if "state_features" in batch:
            state_features = batch["state_features"].to(self.device)

        # Forward
        outputs = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            timestep_ids=timestep_ids,
            segment_ids=segment_ids,
            rtg_values=rtg_values,
            attention_mask=attention_mask,
            state_features=state_features,
            team_token_positions=team_token_positions,
            action_mask=action_mask,
        )

        # Selection loss
        # selection_logits: [batch, 6, 3]
        # selection_labels: [batch, 6]
        selection_logits = outputs.get("selection_logits")
        if selection_logits is not None:
            selection_loss = F.cross_entropy(
                selection_logits.view(-1, 3),
                selection_labels.view(-1),
            )
        else:
            selection_loss = torch.tensor(0.0, device=self.device)

        # Action loss
        # action_logits: [batch, num_actions]
        # action_labels: [batch]
        action_logits = outputs["action_logits"]
        action_loss = F.cross_entropy(action_logits, action_labels)

        # Value loss
        # value: [batch, 1]
        # value_labels: [batch]
        value = outputs["value"].squeeze(-1)
        value_loss = F.mse_loss(value, value_labels)

        # Total loss
        total_loss = (
            self.config.selection_loss_weight * selection_loss
            + self.config.action_loss_weight * action_loss
            + self.config.value_loss_weight * value_loss
        )

        return total_loss, {
            "selection_loss": selection_loss.item(),
            "action_loss": action_loss.item(),
            "value_loss": value_loss.item(),
        }

    def train_epoch(
        self,
        dataloader: DataLoader,
    ) -> dict[str, float]:
        """
        1エポックの学習

        Args:
            dataloader: データローダー

        Returns:
            エポックの統計
        """
        self.model.train()

        total_loss = 0.0
        total_selection_loss = 0.0
        total_action_loss = 0.0
        total_value_loss = 0.0
        num_batches = 0

        for batch in dataloader:
            self.optimizer.zero_grad()

            loss, loss_dict = self.compute_loss(batch)

            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.max_grad_norm,
            )

            self.optimizer.step()

            total_loss += loss.item()
            total_selection_loss += loss_dict["selection_loss"]
            total_action_loss += loss_dict["action_loss"]
            total_value_loss += loss_dict["value_loss"]
            num_batches += 1

        self.scheduler.step()

        return {
            "loss": total_loss / num_batches,
            "selection_loss": total_selection_loss / num_batches,
            "action_loss": total_action_loss / num_batches,
            "value_loss": total_value_loss / num_batches,
            "lr": self.scheduler.get_last_lr()[0],
        }

    def train_on_trajectories(
        self,
        trajectories: list[BattleTrajectory],
        num_epochs: int | None = None,
    ) -> list[dict[str, float]]:
        """
        軌跡データで学習

        Args:
            trajectories: 軌跡リスト
            num_epochs: エポック数（None なら設定値を使用）

        Returns:
            各エポックの統計リスト
        """
        # 空の軌跡リストの場合はスキップ
        if not trajectories:
            logger.warning("No trajectories to train on, skipping training")
            return []

        num_epochs = num_epochs or self.config.training_epochs_per_iteration

        # Dataset と DataLoader を作成
        dataset = BattleTrajectoryDataset(
            trajectories=trajectories,
            tokenizer=self.tokenizer,
            config=self.model.config,
        )

        dataloader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=0,  # メインプロセスで実行
        )

        epoch_stats = []
        for epoch in range(num_epochs):
            stats = self.train_epoch(dataloader)
            epoch_stats.append(stats)
            logger.info(
                f"Epoch {epoch + 1}/{num_epochs}: "
                f"loss={stats['loss']:.4f}, "
                f"sel={stats['selection_loss']:.4f}, "
                f"act={stats['action_loss']:.4f}, "
                f"val={stats['value_loss']:.4f}"
            )

        return epoch_stats

    def run_self_play_iteration(
        self,
        trainer_data: list[dict[str, Any]],
        iteration: int,
        output_dir: Path | None = None,
    ) -> dict[str, Any]:
        """
        1イテレーションの自己対戦と学習

        Args:
            trainer_data: トレーナーデータ
            iteration: イテレーション番号
            output_dir: 出力ディレクトリ（並列MCTS用の一時保存先）

        Returns:
            イテレーションの統計
        """
        # 探索パラメータを計算
        epsilon = self.config.get_epsilon(iteration)
        temperature = self.config.get_temperature(iteration)

        logger.info(
            f"Iteration {iteration}: "
            f"epsilon={epsilon:.3f}, temperature={temperature:.3f}"
        )

        # 並列MCTS用にモデルを一時保存（iteration > 0 かつ num_workers > 1 の場合）
        temp_checkpoint_path = None
        if (
            iteration > 0
            and self.config.num_workers > 1
            and self.config.use_mcts
        ):
            # 一時チェックポイントを保存
            temp_checkpoint_path = output_dir / "_temp_checkpoint" if output_dir else Path("./_temp_checkpoint")
            temp_checkpoint_path.mkdir(parents=True, exist_ok=True)
            self.save_checkpoint(temp_checkpoint_path)
            logger.info(f"Saved temporary checkpoint for parallel MCTS: {temp_checkpoint_path}")

        # 自己対戦でデータ生成
        generator_config = GeneratorConfig(
            epsilon=epsilon,
            temperature=temperature,
            num_workers=self.config.num_workers,
            usage_data_path=self.config.usage_data_path,
            # MCTS設定
            use_mcts=self.config.use_mcts,
            mcts_simulations=self.config.mcts_simulations,
            mcts_max_depth=self.config.mcts_max_depth,
            mcts_c_puct=self.config.mcts_c_puct,
            device=self.config.device,
            # 並列ワーカー用のチェックポイントパス
            model_checkpoint_path=str(temp_checkpoint_path) if temp_checkpoint_path else None,
        )

        # 初期イテレーションはランダムポリシー
        if iteration == 0:
            generator = TrajectoryGenerator(
                trainer_data=trainer_data,
                config=generator_config,
            )
        else:
            generator = TrajectoryGenerator(
                trainer_data=trainer_data,
                config=generator_config,
                model=self.model,
                tokenizer=self.tokenizer,
            )

        logger.info(f"Generating {self.config.games_per_iteration} games...")
        trajectories = generator.generate_batch(self.config.games_per_iteration)

        # データプールに追加
        self.trajectory_pool.add(trajectories)

        # 統計
        wins_p0 = sum(1 for t in trajectories if t.winner == 0)
        wins_p1 = sum(1 for t in trajectories if t.winner == 1)
        draws = sum(1 for t in trajectories if t.winner is None)

        logger.info(
            f"Games: {len(trajectories)}, "
            f"P0 wins: {wins_p0}, P1 wins: {wins_p1}, Draws: {draws}"
        )

        # データプールからサンプリングして学習
        train_trajectories = self.trajectory_pool.sample_balanced(
            min(len(self.trajectory_pool), self.config.trajectory_pool_size // 2)
        )

        logger.info(f"Training on {len(train_trajectories)} trajectories...")
        epoch_stats = self.train_on_trajectories(train_trajectories)

        # 統計をまとめる
        iteration_stats = {
            "iteration": iteration,
            "epsilon": epsilon,
            "temperature": temperature,
            "games_generated": len(trajectories),
            "wins_p0": wins_p0,
            "wins_p1": wins_p1,
            "draws": draws,
            "pool_size": len(self.trajectory_pool),
            "train_trajectories": len(train_trajectories),
            "final_loss": epoch_stats[-1]["loss"] if epoch_stats else 0.0,
            "final_selection_loss": epoch_stats[-1]["selection_loss"] if epoch_stats else 0.0,
            "final_action_loss": epoch_stats[-1]["action_loss"] if epoch_stats else 0.0,
            "final_value_loss": epoch_stats[-1]["value_loss"] if epoch_stats else 0.0,
        }

        self.training_history.append(iteration_stats)
        self.current_iteration = iteration + 1

        return iteration_stats

    def train(
        self,
        trainer_data: list[dict[str, Any]],
        output_dir: str | Path,
        resume_from: str | Path | None = None,
    ) -> dict[str, Any]:
        """
        自己対戦ループで学習

        Args:
            trainer_data: トレーナーデータ
            output_dir: 出力ディレクトリ
            resume_from: 再開するチェックポイント（オプション）

        Returns:
            学習結果
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # チェックポイントから再開
        if resume_from:
            self.load_checkpoint(Path(resume_from))
            logger.info(f"Resumed from iteration {self.current_iteration}")

        start_iteration = self.current_iteration

        for iteration in range(start_iteration, self.config.num_iterations):
            logger.info(f"\n{'='*50}")
            logger.info(f"Starting iteration {iteration + 1}/{self.config.num_iterations}")
            logger.info(f"{'='*50}")

            # 自己対戦と学習
            stats = self.run_self_play_iteration(trainer_data, iteration, output_dir)

            # チェックポイント保存
            if (iteration + 1) % self.config.save_interval == 0:
                checkpoint_path = output_dir / f"checkpoint_iter{iteration + 1}"
                self.save_checkpoint(checkpoint_path)
                logger.info(f"Saved checkpoint to {checkpoint_path}")

            # 評価（オプション）
            if (iteration + 1) % self.config.eval_interval == 0:
                eval_result = self.evaluate(trainer_data)
                logger.info(f"Evaluation: {eval_result}")

        # 最終モデルを保存
        final_path = output_dir / "final"
        self.save_checkpoint(final_path)
        logger.info(f"Saved final model to {final_path}")

        # 学習履歴を保存
        history_path = output_dir / "training_history.json"
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(self.training_history, f, ensure_ascii=False, indent=2)

        return {
            "total_iterations": self.current_iteration,
            "final_pool_size": len(self.trajectory_pool),
            "history": self.training_history,
        }

    def evaluate(
        self,
        trainer_data: list[dict[str, Any]],
        num_games: int | None = None,
    ) -> dict[str, float]:
        """
        モデルを評価

        Args:
            trainer_data: トレーナーデータ
            num_games: 評価ゲーム数

        Returns:
            評価結果
        """
        num_games = num_games or self.config.eval_games

        # 現在のモデル vs ランダム
        generator = TrajectoryGenerator(
            trainer_data=trainer_data,
            config=GeneratorConfig(
                epsilon=0.0,
                temperature=0.1,
                usage_data_path=self.config.usage_data_path,
            ),
            model=self.model,
            tokenizer=self.tokenizer,
        )

        trajectories = generator.generate_batch(num_games)

        wins = sum(1 for t in trajectories if t.winner == 0)
        losses = sum(1 for t in trajectories if t.winner == 1)
        draws = sum(1 for t in trajectories if t.winner is None)

        win_rate = wins / len(trajectories) if trajectories else 0.0

        return {
            "games": len(trajectories),
            "wins": wins,
            "losses": losses,
            "draws": draws,
            "win_rate": win_rate,
        }

    def save_checkpoint(self, path: Path) -> None:
        """チェックポイントを保存"""
        path.mkdir(parents=True, exist_ok=True)

        # モデル
        torch.save(self.model.state_dict(), path / "model.pt")

        # オプティマイザ
        torch.save(self.optimizer.state_dict(), path / "optimizer.pt")

        # トークナイザ
        self.tokenizer.save(path / "tokenizer")

        # メタデータ
        meta = {
            "iteration": self.current_iteration,
            "config": asdict(self.config),
            "model_config": asdict(self.model.config),
            "training_history": self.training_history,
            "pool_stats": self.trajectory_pool.stats(),
        }
        with open(path / "meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        logger.info(f"Checkpoint saved to {path}")

    def load_checkpoint(self, path: Path) -> None:
        """チェックポイントをロード"""
        # モデル
        model_path = path / "model.pt"
        if model_path.exists():
            self.model.load_state_dict(
                torch.load(model_path, map_location=self.device, weights_only=True)
            )

        # オプティマイザ
        optimizer_path = path / "optimizer.pt"
        if optimizer_path.exists():
            self.optimizer.load_state_dict(
                torch.load(optimizer_path, map_location=self.device, weights_only=True)
            )

        # トークナイザ
        tokenizer_path = path / "tokenizer"
        if tokenizer_path.exists():
            self.tokenizer = BattleSequenceTokenizer.load(
                tokenizer_path, self.model.config
            )

        # メタデータ
        meta_path = path / "meta.json"
        if meta_path.exists():
            with open(meta_path, encoding="utf-8") as f:
                meta = json.load(f)
            self.current_iteration = meta.get("iteration", 0)
            self.training_history = meta.get("training_history", [])

        logger.info(f"Checkpoint loaded from {path}")
