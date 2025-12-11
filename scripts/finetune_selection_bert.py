#!/usr/bin/env python3
"""
Pokemon BERT 選出予測ファインチューニング

事前学習済みPokemon BERTを使って、選出予測（Token Classification）を学習する。

使用例:
    # 基本的な実行（ReBeL自己対戦データから学習）
    PYTHONPATH=. uv run python scripts/finetune_selection_bert.py \
        --pretrained models/pokemon_bert \
        --output models/pokemon_bert_selection

    # 対戦データを生成しながら学習
    PYTHONPATH=. uv run python scripts/finetune_selection_bert.py \
        --pretrained models/pokemon_bert \
        --trainer-json data/top_rankers/season_27.json \
        --generate-data \
        --num-games 100 \
        --output models/pokemon_bert_selection
"""

import argparse
import json
import random
from pathlib import Path
from typing import Optional

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, random_split

from src.selection_bert import (
    PokemonBertConfig,
    PokemonBertForMLM,
    PokemonBertForTokenClassification,
    PokemonSelectionDataset,
    PokemonVocab,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pokemon BERT 選出予測ファインチューニング")

    parser.add_argument(
        "--pretrained",
        type=Path,
        default=Path("models/pokemon_bert"),
        help="事前学習済みモデルディレクトリ",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("models/pokemon_bert_selection"),
        help="出力ディレクトリ",
    )

    # データ生成オプション
    parser.add_argument(
        "--selection-data",
        type=Path,
        help="選出データのJSONLファイル（既存データを使う場合）",
    )
    parser.add_argument(
        "--generate-data",
        action="store_true",
        help="ReBeL自己対戦でデータを生成する",
    )
    parser.add_argument(
        "--trainer-json",
        type=Path,
        default=Path("data/top_rankers/season_27.json"),
        help="トレーナーデータ（データ生成用）",
    )
    parser.add_argument(
        "--num-games",
        type=int,
        default=200,
        help="生成するゲーム数",
    )
    parser.add_argument(
        "--mcts-iterations",
        type=int,
        default=50,
        help="MCTS探索回数（データ生成時）",
    )

    # 学習設定
    parser.add_argument("--epochs", type=int, default=50, help="エポック数")
    parser.add_argument("--batch-size", type=int, default=32, help="バッチサイズ")
    parser.add_argument("--learning-rate", type=float, default=2e-5, help="学習率")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="検証データ割合")
    parser.add_argument("--seed", type=int, default=42, help="ランダムシード")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="デバイス",
    )
    parser.add_argument(
        "--winners-only",
        action="store_true",
        default=True,
        help="勝者の選出のみを学習（デフォルト: True）",
    )

    return parser.parse_args()


def load_trainers(path: Path) -> list[dict]:
    """トレーナーデータを読み込み"""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def generate_selection_data(
    trainers: list[dict],
    num_games: int,
    mcts_iterations: int,
    winners_only: bool = True,
    seed: int = 42,
) -> list[dict]:
    """
    MCTS自己対戦から選出データを生成

    Returns:
        list[dict]: 選出データのリスト
            {
                "my_team": [6体のポケモン名],
                "opp_team": [6体のポケモン名],
                "my_selection": [選出した3体のインデックス],  # 先発が最初
                "opp_selection": [相手の選出3体のインデックス],
                "winner": 0 or 1
            }
    """
    from src.pokemon_battle_sim.pokemon import Pokemon

    Pokemon.init()

    # 動的インポート（重いので必要な時だけ）
    from src.hypothesis.hypothesis_mcts import HypothesisMCTSBattle
    from src.models import Trainer

    random.seed(seed)

    # ポケモンインスタンス化用のヘルパー
    def create_pokemon_instance(poke_data: dict) -> Pokemon:
        pokemon = Pokemon(poke_data["name"])
        pokemon.item = poke_data.get("item", "")
        pokemon.ability = poke_data.get("ability", "")
        pokemon.tera_type = poke_data.get("Ttype", "")
        pokemon.moves = poke_data.get("moves", [])
        pokemon.nature = poke_data.get("nature", "")
        if "effort" in poke_data and len(poke_data["effort"]) == 6:
            pokemon.effort = poke_data["effort"]
        return pokemon

    # トレーナーオブジェクト作成
    trainer_objects = []
    for t in trainers:
        raw_pokemons = t["pokemons"]
        pokemon_instances = [create_pokemon_instance(p) for p in raw_pokemons]
        trainer = Trainer(
            name=t["name"],
            rank=t.get("rank", 0),
            rating=t.get("rating", 1500),
            pokemons=pokemon_instances,
            raw_pokemons=raw_pokemons,
        )
        trainer_objects.append(trainer)

    selection_data = []

    print(f"  ゲーム生成中... (0/{num_games})", end="", flush=True)

    for game_idx in range(num_games):
        if (game_idx + 1) % 10 == 0:
            print(f"\r  ゲーム生成中... ({game_idx + 1}/{num_games})", end="", flush=True)

        # ランダムに2人選択
        t1, t2 = random.sample(trainer_objects, 2)

        # 6体から3体を選出（ランダム、先発はインデックス0）
        t1_full_team = [p["name"] for p in t1.raw_pokemons]
        t2_full_team = [p["name"] for p in t2.raw_pokemons]

        t1_selection = random.sample(range(6), 3)
        t2_selection = random.sample(range(6), 3)

        # バトル実行
        try:
            battle = HypothesisMCTSBattle(
                iterations=mcts_iterations,
                hypothesis_samples=5,
            )

            # 選出されたポケモンでバトル
            t1_selected_pokemon = [t1.pokemons[i] for i in t1_selection]
            t2_selected_pokemon = [t2.pokemons[i] for i in t2_selection]

            winner = battle.run_battle(
                trainer1_pokemons=t1_selected_pokemon,
                trainer2_pokemons=t2_selected_pokemon,
            )

            # データ記録
            if winners_only:
                # 勝者視点のデータのみ記録
                if winner == 0:
                    selection_data.append(
                        {
                            "my_team": t1_full_team,
                            "opp_team": t2_full_team,
                            "my_selection": t1_selection,
                            "opp_selection": t2_selection,
                            "winner": 0,
                        }
                    )
                else:
                    selection_data.append(
                        {
                            "my_team": t2_full_team,
                            "opp_team": t1_full_team,
                            "my_selection": t2_selection,
                            "opp_selection": t1_selection,
                            "winner": 0,  # 自分視点では常に勝者
                        }
                    )
            else:
                # 両プレイヤー視点のデータを記録
                selection_data.append(
                    {
                        "my_team": t1_full_team,
                        "opp_team": t2_full_team,
                        "my_selection": t1_selection,
                        "opp_selection": t2_selection,
                        "winner": winner,
                    }
                )
                selection_data.append(
                    {
                        "my_team": t2_full_team,
                        "opp_team": t1_full_team,
                        "my_selection": t2_selection,
                        "opp_selection": t1_selection,
                        "winner": 1 - winner,
                    }
                )

        except Exception as e:
            print(f"\n  Warning: Game {game_idx} failed: {e}")
            continue

    print(f"\r  ゲーム生成完了: {len(selection_data)} samples")

    return selection_data


def load_selection_data(path: Path) -> list[dict]:
    """選出データを読み込み"""
    data = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def save_selection_data(data: list[dict], path: Path) -> None:
    """選出データを保存"""
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def train_epoch(
    model: PokemonBertForTokenClassification,
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
        token_type_ids = batch["token_type_ids"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        output = model(input_ids, attention_mask, token_type_ids, labels=labels)
        loss = output["loss"]
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches


@torch.no_grad()
def evaluate(
    model: PokemonBertForTokenClassification,
    dataloader: DataLoader,
    device: str,
) -> dict[str, float]:
    """評価"""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_valid = 0
    num_batches = 0

    # 詳細メトリクス
    my_correct = 0
    my_total = 0
    opp_correct = 0
    opp_total = 0

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        token_type_ids = batch["token_type_ids"].to(device)
        labels = batch["labels"].to(device)

        output = model(input_ids, attention_mask, token_type_ids, labels=labels)
        loss = output["loss"]
        logits = output["logits"]

        total_loss += loss.item()
        num_batches += 1

        # 正解率計算（-100以外）
        predictions = logits.argmax(dim=-1)
        valid_mask = labels != -100

        total_correct += (predictions[valid_mask] == labels[valid_mask]).sum().item()
        total_valid += valid_mask.sum().item()

        # 自分/相手チーム別の正解率
        # 位置: [CLS]=0, my1-6=1-6, [SEP]=7, opp1-6=8-13, [SEP]=14
        batch_size = input_ids.size(0)
        for i in range(batch_size):
            my_labels = labels[i, 1:7]
            my_preds = predictions[i, 1:7]
            my_mask = my_labels != -100
            my_correct += (my_preds[my_mask] == my_labels[my_mask]).sum().item()
            my_total += my_mask.sum().item()

            opp_labels = labels[i, 8:14]
            opp_preds = predictions[i, 8:14]
            opp_mask = opp_labels != -100
            opp_correct += (opp_preds[opp_mask] == opp_labels[opp_mask]).sum().item()
            opp_total += opp_mask.sum().item()

    return {
        "loss": total_loss / num_batches,
        "accuracy": total_correct / total_valid if total_valid > 0 else 0.0,
        "my_accuracy": my_correct / my_total if my_total > 0 else 0.0,
        "opp_accuracy": opp_correct / opp_total if opp_total > 0 else 0.0,
    }


def main():
    args = parse_args()

    # シード設定
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    print("=" * 60)
    print("Pokemon BERT 選出予測ファインチューニング")
    print("=" * 60)

    # 語彙読み込み
    print(f"\n事前学習モデル読み込み中... ({args.pretrained})")
    vocab = PokemonVocab.load(args.pretrained / "vocab.json")
    print(f"  語彙サイズ: {len(vocab)}")

    # 設定読み込み
    with open(args.pretrained / "config.json") as f:
        config_dict = json.load(f)
    config = PokemonBertConfig(**config_dict)

    # 選出データ取得
    if args.selection_data:
        print(f"\n選出データ読み込み中... ({args.selection_data})")
        selection_data = load_selection_data(args.selection_data)
    elif args.generate_data:
        print(f"\n選出データ生成中...")
        trainers = load_trainers(args.trainer_json)
        print(f"  トレーナー数: {len(trainers)}")
        selection_data = generate_selection_data(
            trainers,
            args.num_games,
            args.mcts_iterations,
            winners_only=args.winners_only,
            seed=args.seed,
        )
    else:
        print("Error: --selection-data または --generate-data を指定してください")
        return

    print(f"  選出データ数: {len(selection_data)}")

    # 出力ディレクトリ作成
    args.output.mkdir(parents=True, exist_ok=True)

    # 生成データを保存
    if args.generate_data:
        save_selection_data(selection_data, args.output / "selection_data.jsonl")
        print(f"  データ保存: {args.output / 'selection_data.jsonl'}")

    # データセット作成
    print("\nデータセット作成中...")
    dataset = PokemonSelectionDataset(selection_data, vocab)
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
    print("\nモデル作成中...")

    # まず MLM モデルを読み込み
    mlm_model = PokemonBertForMLM(config)
    mlm_model.load_state_dict(torch.load(args.pretrained / "best_model.pt", map_location="cpu"))

    # Token Classification モデルに変換
    model = PokemonBertForTokenClassification.from_pretrained_mlm(mlm_model, config)
    model = model.to(args.device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"  パラメータ数: {num_params:,}")
    print(f"  デバイス: {args.device}")

    # Optimizer & Scheduler
    optimizer = AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-7)

    # 語彙保存（参照用）
    vocab.save(args.output / "vocab.json")

    # 設定保存
    with open(args.output / "config.json", "w") as f:
        json.dump(config_dict, f, indent=2)

    # 学習履歴
    history = {
        "train_loss": [],
        "val_loss": [],
        "val_accuracy": [],
        "val_my_accuracy": [],
        "val_opp_accuracy": [],
    }

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
        history["val_my_accuracy"].append(val_metrics["my_accuracy"])
        history["val_opp_accuracy"].append(val_metrics["opp_accuracy"])

        print(
            f"Epoch {epoch:3d}/{args.epochs}: "
            f"train_loss={train_loss:.4f}, "
            f"val_loss={val_metrics['loss']:.4f}, "
            f"acc={val_metrics['accuracy']:.4f} "
            f"(my={val_metrics['my_accuracy']:.4f}, opp={val_metrics['opp_accuracy']:.4f})"
        )

        # ベストモデル保存
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            torch.save(model.state_dict(), args.output / "best_model.pt")
            print(f"  -> Best model saved")

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
