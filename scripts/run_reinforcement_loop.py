#!/usr/bin/env python3
"""
強化学習ループ実行スクリプト

AlphaZeroスタイルのSelf-Play + 学習 + 評価のループを実行する。

Usage:
    poetry run python scripts/run_reinforcement_loop.py \
        --trainer-json data/top_rankers/season_27.json \
        --output models/reinforcement \
        --num-generations 10 \
        --games-per-generation 100

    # 軽量テスト
    poetry run python scripts/run_reinforcement_loop.py \
        --trainer-json data/top_rankers/season_27.json \
        --output models/reinforcement_test \
        --num-generations 3 \
        --games-per-generation 20 \
        --evaluation-games 10 \
        --training-epochs 10
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# プロジェクトルートをパスに追加
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.hypothesis import ItemPriorDatabase
from src.pokemon_battle_sim.pokemon import Pokemon
from src.policy_value_network.reinforcement_loop import (
    ReinforcementLoop,
    ReinforcementLoopConfig,
)

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="強化学習ループ実行")

    # データ設定
    parser.add_argument(
        "--trainer-json",
        type=str,
        default="data/top_rankers/season_27.json",
        help="トレーナーデータのJSONファイル",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="models/reinforcement",
        help="出力ディレクトリ",
    )

    # ループ設定
    parser.add_argument(
        "--num-generations",
        type=int,
        default=10,
        help="世代数",
    )
    parser.add_argument(
        "--games-per-generation",
        type=int,
        default=100,
        help="Self-Play試合数/世代",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=100,
        help="最大ターン数",
    )

    # MCTS設定
    parser.add_argument(
        "--mcts-simulations",
        type=int,
        default=100,
        help="MCTSシミュレーション数",
    )
    parser.add_argument(
        "--n-hypotheses",
        type=int,
        default=10,
        help="仮説サンプリング数",
    )

    # 学習設定
    parser.add_argument(
        "--training-epochs",
        type=int,
        default=50,
        help="学習エポック数",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="バッチサイズ",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=256,
        help="隠れ層次元",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="学習率",
    )

    # 評価設定
    parser.add_argument(
        "--evaluation-games",
        type=int,
        default=50,
        help="評価対戦数",
    )
    parser.add_argument(
        "--win-rate-threshold",
        type=float,
        default=0.55,
        help="新モデル採用の勝率閾値",
    )

    # デバイス
    parser.add_argument(
        "--device",
        type=str,
        default="",
        help="デバイス（空なら自動検出）",
    )

    args = parser.parse_args()

    # Pokemon初期化
    logger.info("Pokemonデータを初期化中...")
    Pokemon.init()

    # 事前確率DB
    logger.info("持ち物事前確率DBを構築中...")
    prior_db = ItemPriorDatabase.from_trainer_json(args.trainer_json)

    # 設定
    config = ReinforcementLoopConfig(
        num_generations=args.num_generations,
        output_dir=args.output,
        games_per_generation=args.games_per_generation,
        max_turns=args.max_turns,
        mcts_simulations=args.mcts_simulations,
        n_hypotheses=args.n_hypotheses,
        training_epochs=args.training_epochs,
        batch_size=args.batch_size,
        hidden_dim=args.hidden_dim,
        learning_rate=args.learning_rate,
        evaluation_games=args.evaluation_games,
        win_rate_threshold=args.win_rate_threshold,
    )

    if args.device:
        config.device = args.device

    # 設定表示
    print("=" * 60)
    print("強化学習ループ")
    print("=" * 60)
    print(f"トレーナーデータ: {args.trainer_json}")
    print(f"出力ディレクトリ: {args.output}")
    print()
    print("ループ設定:")
    print(f"  世代数: {config.num_generations}")
    print(f"  Self-Play試合数/世代: {config.games_per_generation}")
    print(f"  MCTSシミュレーション: {config.mcts_simulations}")
    print()
    print("学習設定:")
    print(f"  エポック数: {config.training_epochs}")
    print(f"  バッチサイズ: {config.batch_size}")
    print(f"  隠れ層次元: {config.hidden_dim}")
    print()
    print("評価設定:")
    print(f"  評価対戦数: {config.evaluation_games}")
    print(f"  勝率閾値: {config.win_rate_threshold:.1%}")
    print()
    print(f"デバイス: {config.device}")
    print("=" * 60)

    # 実行
    loop = ReinforcementLoop(
        prior_db=prior_db,
        trainer_data_path=args.trainer_json,
        config=config,
    )

    loop.run()

    # 結果表示
    print()
    print("=" * 60)
    print("強化学習ループ完了!")
    print("=" * 60)
    print(f"ベストモデル: {args.output}/best_model")
    print(f"履歴: {args.output}/history.json")

    # 履歴サマリー
    if loop.history:
        print()
        print("世代別結果:")
        for h in loop.history:
            status = "✓ 採用" if h["adopted"] else "✗ 棄却"
            print(f"  Gen {h['generation']}: 勝率 {h['win_rate']:.1%} - {status}")


if __name__ == "__main__":
    main()
