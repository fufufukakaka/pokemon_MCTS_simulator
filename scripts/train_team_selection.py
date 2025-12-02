#!/usr/bin/env python3
"""
Team Selection Network 学習スクリプト

相手の6匹を見て、自分の6匹から最適な3匹を選出するネットワークを学習する。

Usage:
    # ランダムデータで初期学習
    poetry run python scripts/train_team_selection.py \
        --trainer-json data/top_rankers/season_27.json \
        --output models/team_selection \
        --num-samples 10000

    # Self-Playデータで学習（より高品質）
    poetry run python scripts/train_team_selection.py \
        --trainer-json data/top_rankers/season_27.json \
        --selfplay-data data/selfplay_records.jsonl \
        --output models/team_selection \
        --num-epochs 100
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# プロジェクトルートをパスに追加
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.policy_value_network.team_selection_dataset import (
    TeamSelectionDataset,
    TeamSelectionDatasetFromSelfPlay,
    generate_selection_data_from_battles,
)
from src.policy_value_network.team_selection_trainer import (
    TeamSelectionTrainer,
    TeamSelectionTrainingConfig,
)

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Team Selection Network 学習")

    # データ設定
    parser.add_argument(
        "--trainer-json",
        type=str,
        default="data/top_rankers/season_27.json",
        help="トレーナーデータのJSONファイル",
    )
    parser.add_argument(
        "--selfplay-data",
        type=str,
        default="",
        help="Self-Playデータのパス（指定時はこれを使用）",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="models/team_selection",
        help="出力ディレクトリ",
    )

    # データ生成設定（Self-Playデータがない場合）
    parser.add_argument(
        "--num-samples",
        type=int,
        default=10000,
        help="生成するサンプル数（ランダムデータ時）",
    )

    # モデル設定
    parser.add_argument(
        "--pokemon-embed-dim",
        type=int,
        default=128,
        help="ポケモン埋め込み次元",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=256,
        help="隠れ層次元",
    )
    parser.add_argument(
        "--num-heads",
        type=int,
        default=4,
        help="Attention Head数",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=2,
        help="Transformer Layer数",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.1,
        help="ドロップアウト率",
    )

    # 学習設定
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=100,
        help="エポック数",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="バッチサイズ",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="学習率",
    )
    parser.add_argument(
        "--early-stopping",
        type=int,
        default=10,
        help="Early Stopping patience",
    )

    # デバイス
    parser.add_argument(
        "--device",
        type=str,
        default="",
        help="デバイス（空なら自動検出）",
    )

    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # データ準備
    if args.selfplay_data:
        logger.info(f"Self-Playデータを使用: {args.selfplay_data}")
        dataset_path = args.selfplay_data
        use_selfplay = True
    else:
        logger.info(f"ランダムデータを生成: {args.num_samples}サンプル")
        dataset_path = output_dir / "random_selection_data.jsonl"
        generate_selection_data_from_battles(
            trainer_data_path=args.trainer_json,
            num_samples=args.num_samples,
            output_path=dataset_path,
        )
        use_selfplay = False
        logger.info(f"データ生成完了: {dataset_path}")

    # 設定
    config = TeamSelectionTrainingConfig(
        pokemon_embed_dim=args.pokemon_embed_dim,
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dropout=args.dropout,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        early_stopping_patience=args.early_stopping,
    )

    if args.device:
        config.device = args.device

    # 設定表示
    print("=" * 60)
    print("Team Selection Network 学習")
    print("=" * 60)
    print(f"トレーナーデータ: {args.trainer_json}")
    print(f"データソース: {'Self-Play' if use_selfplay else 'ランダム'}")
    print(f"出力ディレクトリ: {args.output}")
    print()
    print("モデル設定:")
    print(f"  ポケモン埋め込み次元: {config.pokemon_embed_dim}")
    print(f"  隠れ層次元: {config.hidden_dim}")
    print(f"  Attention Head数: {config.num_heads}")
    print(f"  Transformer Layer数: {config.num_layers}")
    print()
    print("学習設定:")
    print(f"  エポック数: {config.num_epochs}")
    print(f"  バッチサイズ: {config.batch_size}")
    print(f"  学習率: {config.learning_rate}")
    print()
    print(f"デバイス: {config.device}")
    print("=" * 60)

    # 学習
    trainer = TeamSelectionTrainer(config=config)
    history = trainer.train(
        dataset_path=dataset_path,
        output_dir=output_dir,
    )

    # 結果表示
    print()
    print("=" * 60)
    print("学習完了!")
    print("=" * 60)
    print(f"ベストモデル: {output_dir}/best_model.pt")
    print(f"履歴: {output_dir}/history.json")
    print()
    print(f"最終検証損失: {history['val_loss'][-1]:.4f}")
    print(f"  Selection Loss: {history['val_selection_loss'][-1]:.4f}")
    print(f"  Value Loss: {history['val_value_loss'][-1]:.4f}")


if __name__ == "__main__":
    main()
