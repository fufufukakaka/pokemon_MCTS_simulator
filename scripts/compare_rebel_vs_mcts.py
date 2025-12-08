#!/usr/bin/env python3
"""
ReBeL vs HypothesisMCTS 性能比較スクリプト

両方の AI を対戦させて性能を比較する。
"""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from src.hypothesis.hypothesis_mcts import HypothesisMCTS
from src.hypothesis.item_belief_state import ItemBeliefState
from src.hypothesis.item_prior_database import ItemPriorDatabase
from src.hypothesis.pokemon_usage_database import (
    ItemPriorDatabaseAdapter,
    PokemonUsageDatabase,
)
from src.pokemon_battle_sim.battle import Battle
from src.pokemon_battle_sim.pokemon import Pokemon
from src.rebel import (
    PokemonBeliefState,
    PublicBeliefState,
    ReBeLSolver,
    CFRConfig,
)


@dataclass
class MatchResult:
    """対戦結果"""

    winner: Optional[int]  # 0: ReBeL, 1: MCTS, None: draw
    total_turns: int
    rebel_time: float  # ReBeL の累計思考時間
    mcts_time: float  # MCTS の累計思考時間


def create_battle_from_trainer(trainer_data: dict) -> tuple[Battle, list[dict]]:
    """トレーナーデータからバトルを作成"""
    Pokemon.init()
    battle = Battle()
    battle.reset_game()

    pokemons = trainer_data.get("pokemons", [])[:6]

    for i, pokemon_data in enumerate(pokemons):
        pokemon = Pokemon(pokemon_data.get("name", "ピカチュウ"))
        pokemon.item = pokemon_data.get("item", "")
        pokemon.ability = pokemon_data.get("ability", "")
        pokemon.moves = pokemon_data.get("moves", [])[:4]
        pokemon.tera_type = pokemon_data.get("tera_type", "ノーマル")

        if "evs" in pokemon_data:
            pokemon.effort_value = pokemon_data["evs"]
        if "nature" in pokemon_data:
            pokemon.nature = pokemon_data["nature"]

        battle.member[0][i] = pokemon

    return battle, pokemons


def run_match(
    trainer0_data: dict,
    trainer1_data: dict,
    usage_db: PokemonUsageDatabase,
    rebel_cfr_iterations: int = 50,
    rebel_world_samples: int = 20,
    mcts_iterations: int = 200,
    mcts_hypotheses: int = 30,
    max_turns: int = 100,
) -> MatchResult:
    """
    1試合を実行

    Player 0: ReBeL
    Player 1: HypothesisMCTS
    """
    Pokemon.init()
    battle = Battle()
    battle.reset_game()

    # 各プレイヤーのポケモンを作成
    all_pokemon: list[list[Pokemon]] = [[], []]
    for player, trainer_data in enumerate([trainer0_data, trainer1_data]):
        pokemons = trainer_data.get("pokemons", [])[:6]
        for pokemon_data in pokemons:
            pokemon = Pokemon(pokemon_data.get("name", "ピカチュウ"))
            pokemon.item = pokemon_data.get("item", "")
            pokemon.ability = pokemon_data.get("ability", "")
            pokemon.moves = pokemon_data.get("moves", [])[:4]
            pokemon.tera_type = pokemon_data.get("tera_type", "ノーマル")
            all_pokemon[player].append(pokemon)

    # ランダム選出（各プレイヤーから3体）
    for player in [0, 1]:
        available = len(all_pokemon[player])
        num_select = min(3, available)
        if num_select == 0:
            # ポケモンがない場合はダミーを追加
            dummy = Pokemon("ピカチュウ")
            battle.selected[player].append(dummy)
        else:
            indices = random.sample(range(available), num_select)
            for idx in indices:
                battle.selected[player].append(all_pokemon[player][idx])

    # ターン0で先頭のポケモンを場に出す（proceed が自動で行う）
    battle.proceed(commands=[Battle.SKIP, Battle.SKIP])

    # AI 初期化
    # ReBeL (Player 0)
    cfr_config = CFRConfig(
        num_iterations=rebel_cfr_iterations,
        num_world_samples=rebel_world_samples,
    )
    rebel_solver = ReBeLSolver(
        value_network=None,
        cfr_config=cfr_config,
        use_simplified=True,
    )
    rebel_belief = PokemonBeliefState(
        [p.name for p in battle.selected[1]],
        usage_db,
    )

    # HypothesisMCTS (Player 1)
    item_prior_adapter = ItemPriorDatabaseAdapter(usage_db)
    mcts = HypothesisMCTS(
        prior_db=item_prior_adapter,
        n_hypotheses=mcts_hypotheses,
        mcts_iterations=mcts_iterations,
    )
    mcts_belief = ItemBeliefState(
        opponent_pokemon_names=[p.name for p in battle.selected[0]],
        prior_db=item_prior_adapter,
    )

    rebel_time = 0.0
    mcts_time = 0.0
    turn = 0

    while battle.winner() is None and turn < max_turns:
        turn += 1

        # ReBeL の行動選択
        start_time = time.time()
        pbs = PublicBeliefState.from_battle(battle, 0, rebel_belief)
        rebel_action = rebel_solver.get_action(pbs, battle, explore=False)
        rebel_time += time.time() - start_time

        # MCTS の行動選択
        start_time = time.time()
        mcts_action = mcts.get_best_action(battle, 1, mcts_belief)
        mcts_time += time.time() - start_time

        # ターン実行
        battle.proceed(commands=[rebel_action, mcts_action])

    return MatchResult(
        winner=battle.winner(),
        total_turns=turn,
        rebel_time=rebel_time,
        mcts_time=mcts_time,
    )


def main():
    parser = argparse.ArgumentParser(description="ReBeL vs HypothesisMCTS 比較")
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
        "--num-matches",
        type=int,
        default=10,
        help="対戦数",
    )
    parser.add_argument(
        "--rebel-cfr-iterations",
        type=int,
        default=50,
        help="ReBeL の CFR イテレーション数",
    )
    parser.add_argument(
        "--rebel-world-samples",
        type=int,
        default=20,
        help="ReBeL のワールドサンプル数",
    )
    parser.add_argument(
        "--mcts-iterations",
        type=int,
        default=200,
        help="MCTS のイテレーション数",
    )
    parser.add_argument(
        "--mcts-hypotheses",
        type=int,
        default=30,
        help="MCTS の仮説サンプル数",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="結果の出力先",
    )
    args = parser.parse_args()

    # データ読み込み
    print("Loading data...")
    with open(args.trainer_json, "r", encoding="utf-8") as f:
        trainers = json.load(f)

    usage_db = PokemonUsageDatabase.from_json(args.usage_db)

    print(f"Loaded {len(trainers)} trainers")
    print(f"Usage DB: {usage_db}")

    # 対戦実行
    results = []
    rebel_wins = 0
    mcts_wins = 0
    draws = 0

    print(f"\nRunning {args.num_matches} matches...")
    print("=" * 60)

    for i in range(args.num_matches):
        # ランダムに2人のトレーナーを選択
        trainer0, trainer1 = random.sample(trainers, 2)

        result = run_match(
            trainer0,
            trainer1,
            usage_db,
            rebel_cfr_iterations=args.rebel_cfr_iterations,
            rebel_world_samples=args.rebel_world_samples,
            mcts_iterations=args.mcts_iterations,
            mcts_hypotheses=args.mcts_hypotheses,
        )
        results.append(result)

        if result.winner == 0:
            rebel_wins += 1
            winner_str = "ReBeL"
        elif result.winner == 1:
            mcts_wins += 1
            winner_str = "MCTS"
        else:
            draws += 1
            winner_str = "Draw"

        print(
            f"Match {i + 1:3d}: {winner_str:6s} | "
            f"Turns: {result.total_turns:3d} | "
            f"ReBeL: {result.rebel_time:.2f}s | "
            f"MCTS: {result.mcts_time:.2f}s"
        )

    # 結果サマリー
    print("=" * 60)
    print("\nResults Summary")
    print("-" * 40)
    print(f"ReBeL wins:  {rebel_wins:3d} ({100 * rebel_wins / args.num_matches:.1f}%)")
    print(f"MCTS wins:   {mcts_wins:3d} ({100 * mcts_wins / args.num_matches:.1f}%)")
    print(f"Draws:       {draws:3d} ({100 * draws / args.num_matches:.1f}%)")
    print("-" * 40)

    avg_rebel_time = sum(r.rebel_time for r in results) / len(results)
    avg_mcts_time = sum(r.mcts_time for r in results) / len(results)
    avg_turns = sum(r.total_turns for r in results) / len(results)

    print(f"Avg turns:       {avg_turns:.1f}")
    print(f"Avg ReBeL time:  {avg_rebel_time:.2f}s/game")
    print(f"Avg MCTS time:   {avg_mcts_time:.2f}s/game")

    # 結果を保存
    if args.output:
        output_data = {
            "config": {
                "num_matches": args.num_matches,
                "rebel_cfr_iterations": args.rebel_cfr_iterations,
                "rebel_world_samples": args.rebel_world_samples,
                "mcts_iterations": args.mcts_iterations,
                "mcts_hypotheses": args.mcts_hypotheses,
            },
            "results": {
                "rebel_wins": rebel_wins,
                "mcts_wins": mcts_wins,
                "draws": draws,
                "avg_turns": avg_turns,
                "avg_rebel_time": avg_rebel_time,
                "avg_mcts_time": avg_mcts_time,
            },
            "matches": [
                {
                    "winner": r.winner,
                    "total_turns": r.total_turns,
                    "rebel_time": r.rebel_time,
                    "mcts_time": r.mcts_time,
                }
                for r in results
            ],
        }
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
