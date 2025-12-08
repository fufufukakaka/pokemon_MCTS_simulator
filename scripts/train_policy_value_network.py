#!/usr/bin/env python3
"""
Policy-Value Network 学習スクリプト

Self-Playデータを使ってPolicy-Value Networkを学習する。

Usage:
    uv run python scripts/train_policy_value_network.py \
        --dataset data/selfplay_dataset.jsonl \
        --output models/policy_value_v1 \
        --num-epochs 100 \
        --batch-size 64 \
        --hidden-dim 256
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# プロジェクトルートをパスに追加
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.policy_value_network.trainer import PolicyValueTrainer, TrainingConfig

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Policy-Value Network 学習")

    # データ設定
    parser.add_argument(
        "--dataset",
        type=str,
        default="data/selfplay_dataset.jsonl",
        help="Self-Playデータセットのパス",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="models/policy_value_v1",
        help="出力ディレクトリ",
    )

    # モデル設定
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=256,
        help="隠れ層の次元",
    )
    parser.add_argument(
        "--num-res-blocks",
        type=int,
        default=4,
        help="残差ブロック数",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.1,
        help="ドロップアウト率",
    )

    # 学習設定
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
        "--weight-decay",
        type=float,
        default=1e-4,
        help="Weight decay",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=100,
        help="エポック数",
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=10,
        help="早期終了のpatience",
    )

    # 損失関数の重み
    parser.add_argument(
        "--policy-loss-weight",
        type=float,
        default=1.0,
        help="Policy損失の重み",
    )
    parser.add_argument(
        "--value-loss-weight",
        type=float,
        default=1.0,
        help="Value損失の重み",
    )

    # その他
    parser.add_argument(
        "--max-actions",
        type=int,
        default=50,
        help="行動数の上限",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="検証データの割合",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="",
        help="デバイス（空の場合は自動検出）",
    )
    parser.add_argument(
        "--save-every-n-epochs",
        type=int,
        default=10,
        help="チェックポイント保存間隔",
    )

    args = parser.parse_args()

    # 設定作成
    config = TrainingConfig(
        hidden_dim=args.hidden_dim,
        num_res_blocks=args.num_res_blocks,
        dropout=args.dropout,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_epochs=args.num_epochs,
        early_stopping_patience=args.early_stopping_patience,
        policy_loss_weight=args.policy_loss_weight,
        value_loss_weight=args.value_loss_weight,
        val_ratio=args.val_ratio,
        save_every_n_epochs=args.save_every_n_epochs,
    )

    if args.device:
        config.device = args.device

    # 情報表示
    print("=" * 60)
    print("Policy-Value Network 学習")
    print("=" * 60)
    print(f"データセット: {args.dataset}")
    print(f"出力ディレクトリ: {args.output}")
    print(f"デバイス: {config.device}")
    print()
    print("モデル設定:")
    print(f"  隠れ層次元: {config.hidden_dim}")
    print(f"  残差ブロック数: {config.num_res_blocks}")
    print(f"  ドロップアウト: {config.dropout}")
    print()
    print("学習設定:")
    print(f"  バッチサイズ: {config.batch_size}")
    print(f"  学習率: {config.learning_rate}")
    print(f"  エポック数: {config.num_epochs}")
    print(f"  早期終了patience: {config.early_stopping_patience}")
    print("=" * 60)

    # 学習実行
    trainer = PolicyValueTrainer(config=config)

    history = trainer.train(
        dataset_path=args.dataset,
        output_dir=args.output,
        max_actions=args.max_actions,
    )

    # 結果表示
    print()
    print("=" * 60)
    print("学習完了!")
    print("=" * 60)
    print(f"最終Train Loss: {history['train_loss'][-1]:.4f}")
    print(f"最終Val Loss: {history['val_loss'][-1]:.4f}")
    print(f"最良Val Loss: {min(history['val_loss']):.4f}")
    print(f"モデル保存先: {args.output}")


if __name__ == "__main__":
    main()
