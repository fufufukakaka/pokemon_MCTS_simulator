#!/usr/bin/env python3
"""
ReBeL 強化学習トレーニングスクリプト

自己対戦でデータを生成し、Value Network を学習する。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.hypothesis.pokemon_usage_database import PokemonUsageDatabase
from src.rebel import (
    ReBeLTrainer,
    TrainingConfig,
    ReBeLValueNetwork,
)


def main():
    parser = argparse.ArgumentParser(description="ReBeL トレーニング")
    parser.add_argument(
        "--trainer-json",
        type=str,
        default="data/top_rankers/season_27.json",
        help="トレーナーデータのパス",
    )
    parser.add_argument(
        "--usage-db",
        type=str,
        default="data/pokedb_usage/season_37_top150.json",
        help="使用率データベースのパス",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="models/rebel",
        help="出力ディレクトリ",
    )
    parser.add_argument(
        "--num-iterations",
        type=int,
        default=10,
        help="学習イテレーション数",
    )
    parser.add_argument(
        "--games-per-iteration",
        type=int,
        default=20,
        help="各イテレーションでの自己対戦数",
    )
    parser.add_argument(
        "--cfr-iterations",
        type=int,
        default=30,
        help="CFR イテレーション数",
    )
    parser.add_argument(
        "--cfr-world-samples",
        type=int,
        default=10,
        help="CFR ワールドサンプル数",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="学習バッチサイズ",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="学習率",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=5,
        help="各イテレーションでのエポック数",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=256,
        help="ネットワークの隠れ層次元",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="デバイス (cpu/cuda)",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="学習を再開するチェックポイントのパス",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="学習後に評価を実行",
    )
    parser.add_argument(
        "--eval-games",
        type=int,
        default=30,
        help="評価試合数",
    )
    args = parser.parse_args()

    # データ読み込み
    print("Loading data...")
    with open(args.trainer_json, "r", encoding="utf-8") as f:
        trainer_data = json.load(f)

    usage_db = PokemonUsageDatabase.from_json(args.usage_db)

    print(f"Loaded {len(trainer_data)} trainers")
    print(f"Usage DB: {usage_db}")

    # 設定
    config = TrainingConfig(
        games_per_iteration=args.games_per_iteration,
        cfr_iterations=args.cfr_iterations,
        cfr_world_samples=args.cfr_world_samples,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        device=args.device,
        save_interval=5,
    )

    # Value Network
    value_network = ReBeLValueNetwork(
        hidden_dim=args.hidden_dim,
        num_res_blocks=4,
        dropout=0.1,
    )

    # トレーナー
    trainer = ReBeLTrainer(
        usage_db=usage_db,
        trainer_data=trainer_data,
        config=config,
        value_network=value_network,
    )

    # 再開
    if args.resume:
        print(f"Resuming from {args.resume}")
        trainer.load(Path(args.resume))

    # 学習
    print(f"\nStarting training for {args.num_iterations} iterations...")
    print(f"Config: {config}")
    print("=" * 60)

    trainer.train(args.num_iterations, args.output)

    print("\nTraining completed!")
    print(f"Model saved to {args.output}")

    # 評価
    if args.evaluate:
        print("\n" + "=" * 60)
        print("Running evaluation...")

        # vs Random
        results_random = trainer.evaluate_against_baseline(
            num_games=args.eval_games,
            baseline_type="random",
        )

        # vs CFR-only (no NN)
        results_cfr = trainer.evaluate_against_baseline(
            num_games=args.eval_games,
            baseline_type="cfr_only",
        )

        # 結果を保存
        eval_results = {
            "vs_random": results_random,
            "vs_cfr_only": results_cfr,
        }
        with open(Path(args.output) / "evaluation_results.json", "w") as f:
            json.dump(eval_results, f, indent=2)

        print(f"\nEvaluation results saved to {args.output}/evaluation_results.json")


if __name__ == "__main__":
    main()
