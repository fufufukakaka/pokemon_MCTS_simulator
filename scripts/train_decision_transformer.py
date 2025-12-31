#!/usr/bin/env python3
"""
Train Pokemon Battle Decision Transformer

Usage:
    # Self-play training loop (recommended)
    uv run python scripts/train_decision_transformer.py \
        --trainer-json data/top_rankers/season_36.json \
        --output models/decision_transformer \
        --num-iterations 50 \
        --games-per-iteration 100

    # Quick test run
    uv run python scripts/train_decision_transformer.py \
        --trainer-json data/top_rankers/season_36.json \
        --output models/decision_transformer_test \
        --num-iterations 3 \
        --games-per-iteration 20 \
        --epochs-per-iteration 5

    # Resume from checkpoint
    uv run python scripts/train_decision_transformer.py \
        --trainer-json data/top_rankers/season_36.json \
        --output models/decision_transformer \
        --resume models/decision_transformer/checkpoint_iter20 \
        --num-iterations 50

    # Generate random trajectories only (no training)
    uv run python scripts/train_decision_transformer.py \
        --trainer-json data/top_rankers/season_36.json \
        --output data/dt_trajectories.jsonl \
        --generate-only \
        --num-games 1000
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.decision_transformer.config import (
    PokemonBattleTransformerConfig,
    TrainingConfig,
)
from src.decision_transformer.trainer import DecisionTransformerTrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Train Pokemon Battle Decision Transformer"
    )

    # 入出力
    parser.add_argument(
        "--trainer-json",
        type=str,
        nargs="+",
        required=True,
        help="Path to trainer JSON file(s). Multiple files can be specified.",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory for checkpoints, or output file for --generate-only",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume from checkpoint directory",
    )

    # 学習設定
    parser.add_argument(
        "--num-iterations",
        type=int,
        default=50,
        help="Number of self-play iterations (default: 50)",
    )
    parser.add_argument(
        "--games-per-iteration",
        type=int,
        default=100,
        help="Games to generate per iteration (default: 100)",
    )
    parser.add_argument(
        "--epochs-per-iteration",
        type=int,
        default=10,
        help="Training epochs per iteration (default: 10)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Training batch size (default: 32)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="Learning rate (default: 1e-4)",
    )

    # モデル設定
    parser.add_argument(
        "--hidden-size",
        type=int,
        default=256,
        help="Transformer hidden size (default: 256)",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=6,
        help="Number of transformer layers (default: 6)",
    )
    parser.add_argument(
        "--num-heads",
        type=int,
        default=8,
        help="Number of attention heads (default: 8)",
    )

    # 探索設定
    parser.add_argument(
        "--epsilon-start",
        type=float,
        default=0.3,
        help="Initial epsilon for exploration (default: 0.3)",
    )
    parser.add_argument(
        "--epsilon-end",
        type=float,
        default=0.05,
        help="Final epsilon (default: 0.05)",
    )
    parser.add_argument(
        "--temperature-start",
        type=float,
        default=1.0,
        help="Initial temperature (default: 1.0)",
    )
    parser.add_argument(
        "--temperature-end",
        type=float,
        default=0.5,
        help="Final temperature (default: 0.5)",
    )

    # チェックポイント
    parser.add_argument(
        "--save-interval",
        type=int,
        default=5,
        help="Save checkpoint every N iterations (default: 5)",
    )
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=5,
        help="Evaluate every N iterations (default: 5)",
    )

    # その他
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda", "mps"],
        help="Device to use (default: cpu)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Number of parallel workers for data generation (default: 1)",
    )
    parser.add_argument(
        "--pool-size",
        type=int,
        default=100000,
        help="Trajectory pool size (default: 100000)",
    )

    # データ生成のみモード
    parser.add_argument(
        "--generate-only",
        action="store_true",
        help="Generate trajectories without training",
    )
    parser.add_argument(
        "--num-games",
        type=int,
        default=1000,
        help="Number of games to generate in --generate-only mode (default: 1000)",
    )

    args = parser.parse_args()

    # トレーナーデータをロード（複数ファイル対応）
    trainer_data = []
    for json_path in args.trainer_json:
        logger.info(f"Loading trainer data from {json_path}")
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)
            # リストでない場合（単一トレーナー）はリストに変換
            if isinstance(data, dict):
                data = [data]
            trainer_data.extend(data)
            logger.info(f"  -> Loaded {len(data)} trainers from {json_path}")
    logger.info(f"Total: {len(trainer_data)} trainers from {len(args.trainer_json)} file(s)")

    # データ生成のみモード
    if args.generate_only:
        from src.decision_transformer.data_generator import TrajectoryGenerator
        from src.decision_transformer.dataset import save_trajectories_to_jsonl

        logger.info(f"Generating {args.num_games} random trajectories...")
        generator = TrajectoryGenerator(trainer_data=trainer_data)
        trajectories = generator.generate_batch(args.num_games)

        # 保存
        if args.output:
            save_trajectories_to_jsonl(trajectories, Path(args.output))
            logger.info(f"Saved {len(trajectories)} trajectories to {args.output}")

        # 統計
        wins_p0 = sum(1 for t in trajectories if t.winner == 0)
        wins_p1 = sum(1 for t in trajectories if t.winner == 1)
        draws = sum(1 for t in trajectories if t.winner is None)
        logger.info(f"Player 0 wins: {wins_p0}, Player 1 wins: {wins_p1}, Draws: {draws}")

        return

    # モデル設定
    model_config = PokemonBattleTransformerConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_layers,
        num_attention_heads=args.num_heads,
    )

    # 学習設定
    training_config = TrainingConfig(
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_iterations=args.num_iterations,
        games_per_iteration=args.games_per_iteration,
        training_epochs_per_iteration=args.epochs_per_iteration,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        temperature_start=args.temperature_start,
        temperature_end=args.temperature_end,
        save_interval=args.save_interval,
        eval_interval=args.eval_interval,
        device=args.device,
        num_workers=args.num_workers,
        trajectory_pool_size=args.pool_size,
        model_config=model_config,
    )

    # トレーナーを作成
    trainer = DecisionTransformerTrainer(config=training_config)

    logger.info("Starting training...")
    logger.info(f"Model config: hidden_size={args.hidden_size}, layers={args.num_layers}, heads={args.num_heads}")
    logger.info(f"Training config: iterations={args.num_iterations}, games/iter={args.games_per_iteration}")

    # 学習実行
    result = trainer.train(
        trainer_data=trainer_data,
        output_dir=args.output,
        resume_from=args.resume,
    )

    logger.info("Training completed!")
    logger.info(f"Total iterations: {result['total_iterations']}")
    logger.info(f"Final pool size: {result['final_pool_size']}")


if __name__ == "__main__":
    main()
