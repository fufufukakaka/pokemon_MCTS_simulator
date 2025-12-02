#!/usr/bin/env python3
"""
Self-Play データ生成スクリプト

仮説ベースMCTS同士を対戦させ、Policy-Value Networkの学習用データを生成する。

Usage:
    poetry run python scripts/generate_selfplay_dataset.py \
        --trainer-json data/top_rankers/season_27.json \
        --output data/selfplay_dataset.jsonl \
        --num-games 100 \
        --n-hypotheses 15 \
        --mcts-iterations 100
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

from tqdm import tqdm

from src.hypothesis import (
    ItemPriorDatabase,
    SelfPlayGenerator,
    save_records_to_jsonl,
)
from src.pokemon_battle_sim.pokemon import Pokemon


def load_trainers(json_path: str) -> list[dict]:
    """トレーナーデータを読み込み"""
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def select_random_matchup(trainers: list[dict]) -> tuple[dict, dict]:
    """ランダムに2人のトレーナーを選択"""
    trainer0, trainer1 = random.sample(trainers, 2)
    return trainer0, trainer1


def main():
    parser = argparse.ArgumentParser(
        description="Self-Play データ生成"
    )
    parser.add_argument(
        "--trainer-json",
        type=str,
        default="data/top_rankers/season_27.json",
        help="トレーナーデータのJSONファイル",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/selfplay_dataset.jsonl",
        help="出力ファイルパス",
    )
    parser.add_argument(
        "--num-games",
        type=int,
        default=100,
        help="生成する試合数",
    )
    parser.add_argument(
        "--n-hypotheses",
        type=int,
        default=15,
        help="仮説サンプリング数",
    )
    parser.add_argument(
        "--mcts-iterations",
        type=int,
        default=100,
        help="MCTS イテレーション数",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=100,
        help="最大ターン数",
    )
    parser.add_argument(
        "--record-every-n-turns",
        type=int,
        default=1,
        help="何ターンごとに記録するか",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="ランダムシード",
    )

    args = parser.parse_args()

    # シード設定
    if args.seed is not None:
        random.seed(args.seed)

    print("=" * 60)
    print("Self-Play データ生成")
    print("=" * 60)
    print(f"トレーナーデータ: {args.trainer_json}")
    print(f"出力ファイル: {args.output}")
    print(f"試合数: {args.num_games}")
    print(f"仮説数: {args.n_hypotheses}")
    print(f"MCTSイテレーション: {args.mcts_iterations}")
    print("=" * 60)

    # Pokemon初期化
    print("\nPokemonデータを初期化中...")
    Pokemon.init()

    # トレーナーデータ読み込み
    print("トレーナーデータを読み込み中...")
    trainers = load_trainers(args.trainer_json)
    print(f"  {len(trainers)} 人のトレーナーを読み込みました")

    # 事前確率データベース構築
    print("持ち物事前確率データベースを構築中...")
    prior_db = ItemPriorDatabase.from_trainer_json(args.trainer_json)
    print(f"  {len(prior_db.get_all_pokemon_names())} 種のポケモンの持ち物分布を構築しました")

    # Self-Playジェネレーター
    generator = SelfPlayGenerator(
        prior_db=prior_db,
        n_hypotheses=args.n_hypotheses,
        mcts_iterations=args.mcts_iterations,
    )

    # データ生成
    print(f"\n{args.num_games} 試合を生成中...")
    records = []
    total_turns = 0
    wins = {0: 0, 1: 0, None: 0}

    start_time = time.time()

    for game_idx in tqdm(range(args.num_games), desc="Generating games"):
        # ランダムにマッチアップを選択
        trainer0, trainer1 = select_random_matchup(trainers)

        # 各トレーナーから3体ずつ選出（先頭3体）
        team0 = trainer0["pokemons"][:3]
        team1 = trainer1["pokemons"][:3]

        # 試合生成
        game_record = generator.generate_game(
            trainer0_pokemons=team0,
            trainer1_pokemons=team1,
            trainer0_name=trainer0["name"],
            trainer1_name=trainer1["name"],
            game_id=f"game_{game_idx:05d}",
            max_turns=args.max_turns,
            record_every_n_turns=args.record_every_n_turns,
        )

        records.append(game_record)
        total_turns += game_record.total_turns
        wins[game_record.winner] += 1

    elapsed = time.time() - start_time

    # 統計情報
    print("\n" + "=" * 60)
    print("生成完了!")
    print("=" * 60)
    print(f"試合数: {len(records)}")
    print(f"総ターン数: {total_turns}")
    print(f"平均ターン数: {total_turns / len(records):.1f}")
    print(f"勝率: Player0={wins[0]}, Player1={wins[1]}, 引き分け={wins[None]}")
    print(f"生成時間: {elapsed:.1f}秒 ({elapsed / len(records):.2f}秒/試合)")

    # ターン記録数
    total_turn_records = sum(len(r.turns) for r in records)
    print(f"総ターン記録数: {total_turn_records}")

    # 保存
    print(f"\n{args.output} に保存中...")
    save_records_to_jsonl(records, args.output)
    print("保存完了!")

    # サンプル出力
    if records:
        sample = records[0]
        print("\n" + "=" * 60)
        print("サンプル記録 (最初の試合)")
        print("=" * 60)
        print(f"Game ID: {sample.game_id}")
        print(f"{sample.player0_trainer} vs {sample.player1_trainer}")
        print(f"チーム0: {sample.player0_team}")
        print(f"チーム1: {sample.player1_team}")
        print(f"勝者: Player{sample.winner}")
        print(f"ターン数: {sample.total_turns}")

        if sample.turns:
            t = sample.turns[0]
            print(f"\n最初のターン記録:")
            print(f"  ターン: {t.turn}, プレイヤー: {t.player}")
            print(f"  場: {t.my_pokemon} vs {t.opp_pokemon}")
            print(f"  Value: {t.value:.3f}")
            print(f"  Policy (上位3):")
            sorted_policy = sorted(t.policy.items(), key=lambda x: -x[1])[:3]
            for action, prob in sorted_policy:
                print(f"    {action}: {prob:.2%}")
            print(f"  選択した行動: {t.action}")


if __name__ == "__main__":
    main()
