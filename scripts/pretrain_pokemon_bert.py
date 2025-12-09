#!/usr/bin/env python3
"""
Pokemon BERT 事前学習スクリプト

パーティデータからMLMタスクでポケモンの埋め込みを学習する。

使用例:
    # 基本的な実行
    uv run python scripts/pretrain_pokemon_bert.py

    # パラメータ指定
    uv run python scripts/pretrain_pokemon_bert.py \
        --seasons 34 35 36 \
        --output models/pokemon_bert \
        --epochs 100 \
        --batch-size 32

    # GPU使用
    uv run python scripts/pretrain_pokemon_bert.py --device cuda
"""

import argparse
import json
import random
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, random_split

from src.selection_bert import (
    PokemonBertConfig,
    PokemonBertForMLM,
    PokemonMLMDataset,
    PokemonVocab,
    load_team_data,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pokemon BERT 事前学習")
    parser.add_argument(
        "--seasons",
        type=int,
        nargs="+",
        default=[34, 35, 36],
        help="使用するシーズン",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("src/battle_data"),
        help="パーティデータのディレクトリ",
    )
    parser.add_argument(
        "--zukan-path",
        type=Path,
        default=Path("data/zukan.txt"),
        help="図鑑データのパス",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("models/pokemon_bert"),
        help="出力ディレクトリ",
    )

    # モデル設定
    parser.add_argument("--hidden-size", type=int, default=256, help="隠れ層サイズ")
    parser.add_argument("--num-layers", type=int, default=4, help="Transformerレイヤー数")
    parser.add_argument("--num-heads", type=int, default=4, help="Attention Head数")
    parser.add_argument("--dropout", type=float, default=0.1, help="ドロップアウト率")

    # 学習設定
    parser.add_argument("--epochs", type=int, default=100, help="エポック数")
    parser.add_argument("--batch-size", type=int, default=32, help="バッチサイズ")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="学習率")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--mask-prob", type=float, default=0.15, help="マスク確率")
    parser.add_argument(
        "--augment-factor", type=int, default=20, help="データ拡張倍率（シャッフル回数）"
    )
    parser.add_argument("--val-ratio", type=float, default=0.1, help="検証データ割合")
    parser.add_argument("--seed", type=int, default=42, help="ランダムシード")

    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="デバイス",
    )
    parser.add_argument(
        "--save-every", type=int, default=10, help="チェックポイント保存間隔"
    )

    return parser.parse_args()


def train_epoch(
    model: PokemonBertForMLM,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
) -> float:
    """1エポックの学習"""
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        output = model(input_ids, attention_mask, labels=labels)
        loss = output["loss"]
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches


@torch.no_grad()
def evaluate(
    model: PokemonBertForMLM,
    dataloader: DataLoader,
    device: str,
) -> dict[str, float]:
    """評価"""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_masked = 0
    num_batches = 0

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        output = model(input_ids, attention_mask, labels=labels)
        loss = output["loss"]
        logits = output["logits"]

        total_loss += loss.item()
        num_batches += 1

        # マスクされた位置の正解率
        mask = labels != -100
        if mask.any():
            predictions = logits.argmax(dim=-1)
            total_correct += (predictions[mask] == labels[mask]).sum().item()
            total_masked += mask.sum().item()

    return {
        "loss": total_loss / num_batches,
        "accuracy": total_correct / total_masked if total_masked > 0 else 0.0,
    }


def main():
    args = parse_args()

    # シード設定
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    print("=" * 60)
    print("Pokemon BERT 事前学習")
    print("=" * 60)

    # データ読み込み
    print(f"\nデータ読み込み中... (シーズン: {args.seasons})")
    teams = load_team_data(args.data_dir, args.seasons)
    print(f"  パーティ数: {len(teams)}")

    # 語彙作成（図鑑から）
    print(f"\n語彙構築中... ({args.zukan_path})")
    vocab = PokemonVocab.from_zukan(args.zukan_path)
    print(f"  語彙サイズ: {len(vocab)}")

    # 未知ポケモンのチェック
    unknown_pokemon = set()
    for team in teams:
        for pokemon in team:
            if vocab.encode(pokemon) == vocab.unk_id:
                unknown_pokemon.add(pokemon)
    if unknown_pokemon:
        print(f"  警告: 未知のポケモン {len(unknown_pokemon)}種:")
        for p in sorted(unknown_pokemon)[:10]:
            print(f"    - {p}")
        if len(unknown_pokemon) > 10:
            print(f"    ... 他{len(unknown_pokemon) - 10}種")

    # データセット作成
    print(f"\nデータセット作成中... (augment_factor={args.augment_factor})")
    dataset = PokemonMLMDataset(
        teams, vocab, mask_prob=args.mask_prob, augment_factor=args.augment_factor
    )
    print(f"  サンプル数: {len(dataset)}")

    # Train/Val分割
    val_size = int(len(dataset) * args.val_ratio)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    print(f"  Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=0)

    # モデル作成
    print(f"\nモデル作成中...")
    config = PokemonBertConfig(
        vocab_size=len(vocab),
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_layers,
        num_attention_heads=args.num_heads,
        intermediate_size=args.hidden_size * 2,
        hidden_dropout_prob=args.dropout,
        attention_probs_dropout_prob=args.dropout,
    )
    model = PokemonBertForMLM(config).to(args.device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"  パラメータ数: {num_params:,}")
    print(f"  デバイス: {args.device}")

    # Optimizer & Scheduler
    optimizer = AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # 出力ディレクトリ
    args.output.mkdir(parents=True, exist_ok=True)

    # 語彙保存
    vocab.save(args.output / "vocab.json")

    # 設定保存
    config_dict = {
        "vocab_size": config.vocab_size,
        "hidden_size": config.hidden_size,
        "num_hidden_layers": config.num_hidden_layers,
        "num_attention_heads": config.num_attention_heads,
        "intermediate_size": config.intermediate_size,
        "hidden_dropout_prob": config.hidden_dropout_prob,
        "attention_probs_dropout_prob": config.attention_probs_dropout_prob,
        "max_position_embeddings": config.max_position_embeddings,
        "type_vocab_size": config.type_vocab_size,
    }
    with open(args.output / "config.json", "w") as f:
        json.dump(config_dict, f, indent=2)

    # 学習履歴
    history = {"train_loss": [], "val_loss": [], "val_accuracy": []}

    # 学習ループ
    print(f"\n学習開始 (epochs={args.epochs})")
    print("-" * 60)

    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, args.device)
        val_metrics = evaluate(model, val_loader, args.device)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_metrics["loss"])
        history["val_accuracy"].append(val_metrics["accuracy"])

        print(
            f"Epoch {epoch:3d}/{args.epochs}: "
            f"train_loss={train_loss:.4f}, "
            f"val_loss={val_metrics['loss']:.4f}, "
            f"val_acc={val_metrics['accuracy']:.4f}"
        )

        # ベストモデル保存
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            torch.save(model.state_dict(), args.output / "best_model.pt")
            print(f"  -> Best model saved (val_loss={best_val_loss:.4f})")

        # チェックポイント
        if epoch % args.save_every == 0:
            torch.save(model.state_dict(), args.output / f"checkpoint_epoch{epoch}.pt")

    # 最終モデル保存
    torch.save(model.state_dict(), args.output / "final_model.pt")

    # 履歴保存
    with open(args.output / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    print("-" * 60)
    print(f"\n学習完了!")
    print(f"  Best val_loss: {best_val_loss:.4f}")
    print(f"  出力: {args.output}")


if __name__ == "__main__":
    main()
