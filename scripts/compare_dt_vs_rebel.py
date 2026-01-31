#!/usr/bin/env python
"""
Decision Transformer vs ReBeL+Selection BERT 対決スクリプト

2つのモデルを対戦させて勝率を比較する。
バトルログはbattle_ui互換のJSON形式で保存し、リプレイ可能。

Usage:
    # 10試合
    uv run python scripts/compare_dt_vs_rebel.py \
        --dt-checkpoint models/decision_transformer_full \
        --rebel-checkpoint models/revel_full_state_selection_BERT_move_effective/checkpoint_iter100 \
        --num-matches 10

    # 1000試合（本番）
    uv run python scripts/compare_dt_vs_rebel.py \
        --dt-checkpoint models/decision_transformer_full \
        --rebel-checkpoint models/revel_full_state_selection_BERT_move_effective/checkpoint_iter100 \
        --num-matches 1000 \
        --output results/dt_vs_rebel_1000
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

# プロジェクトルートをパスに追加
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.battle_ui.services.rebel_ai_service import RebelAI, RebelAIConfig

# AI サービス
from src.decision_transformer.ai_service import (
    DecisionTransformerAI,
    DecisionTransformerAIConfig,
)
from src.pokemon_battle_sim.battle import Battle
from src.pokemon_battle_sim.pokemon import Pokemon

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class MatchResult:
    """1試合の結果"""

    match_id: int
    winner: str  # "dt" or "rebel" or "draw"
    total_turns: int
    dt_team_name: str
    rebel_team_name: str
    dt_remaining: int  # DTの残りポケモン数
    rebel_remaining: int  # ReBeLの残りポケモン数
    dt_think_time: float  # DT合計思考時間
    rebel_think_time: float  # ReBeL合計思考時間
    surrendered_by: Optional[str] = None  # 降参した側


@dataclass
class BattleLogEntry:
    """バトルログエントリ"""

    turn: int
    player: str  # "dt" or "rebel"
    message: str


@dataclass
class MatchLog:
    """1試合の完全なログ（リプレイ用）"""

    match_id: int
    timestamp: str
    result: MatchResult
    teams: dict[str, Any]  # 両チームの情報
    final_state: dict[str, Any]  # 最終状態
    battle_log: list[dict[str, Any]]  # ログエントリ


def load_trainer_pools(trainer_json_paths: list[str]) -> list[dict]:
    """複数のトレーナーJSONからパーティプールを読み込む"""
    pool = []

    for path in trainer_json_paths:
        path = Path(path)
        if not path.exists():
            logger.warning(f"Trainer file not found: {path}")
            continue

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # season_XX.json形式（リスト）
        if isinstance(data, list):
            for trainer in data:
                if "pokemons" in trainer:
                    pool.append(
                        {
                            "name": trainer.get("name", f"Trainer_{len(pool)}"),
                            "pokemons": trainer["pokemons"],
                        }
                    )
        # my_fixed_party.json形式（単一パーティ）
        elif "pokemons" in data:
            pool.append(
                {
                    "name": data.get("name", path.stem),
                    "pokemons": data["pokemons"],
                }
            )

    logger.info(f"Loaded {len(pool)} parties from {len(trainer_json_paths)} files")
    return pool


def create_pokemon_from_data(data: dict) -> Pokemon:
    """辞書データからPokemonインスタンスを作成"""
    pokemon = Pokemon(data["name"])
    pokemon.item = data.get("item", "")
    pokemon.nature = data.get("nature", "まじめ")
    pokemon.ability = data.get("ability", "")
    pokemon.Ttype = data.get("Ttype", data.get("tera_type", ""))
    pokemon.moves = data.get("moves", [])[:4]

    # 努力値
    effort = data.get("effort", [0, 0, 0, 0, 0, 0])
    if isinstance(effort, list) and len(effort) == 6:
        pokemon.effort = effort

    # 個体値
    indiv = data.get("individual", [31, 31, 31, 31, 31, 31])
    if isinstance(indiv, list) and len(indiv) == 6:
        pokemon.indiv = indiv

    # ステータス再計算
    pokemon.update_status()
    return pokemon


def setup_battle(
    dt_party_data: list[dict],
    rebel_party_data: list[dict],
    dt_selection: list[int],
    rebel_selection: list[int],
) -> Battle:
    """バトルをセットアップ"""
    battle = Battle()
    battle.reset_game()

    # DTのチーム（player 0）
    for i in dt_selection:
        pokemon = create_pokemon_from_data(dt_party_data[i])
        battle.selected[0].append(pokemon)

    # ReBeLのチーム（player 1）
    for i in rebel_selection:
        pokemon = create_pokemon_from_data(rebel_party_data[i])
        battle.selected[1].append(pokemon)

    return battle


def run_single_match(
    match_id: int,
    dt_ai: DecisionTransformerAI,
    rebel_ai: RebelAI,
    dt_party: dict,
    rebel_party: dict,
    max_turns: int = 100,
) -> tuple[MatchResult, MatchLog]:
    """1試合を実行"""
    dt_party_data = dt_party["pokemons"]
    rebel_party_data = rebel_party["pokemons"]

    dt_team_names = [p["name"] for p in dt_party_data]
    rebel_team_names = [p["name"] for p in rebel_party_data]

    # AIをリセット
    dt_ai.reset()
    rebel_ai.reset()

    dt_think_time = 0.0
    rebel_think_time = 0.0
    battle_log: list[dict[str, Any]] = []
    surrendered_by: Optional[str] = None

    # 選出フェーズ
    logger.info(f"Match {match_id}: Selection phase")
    logger.info(f"  DT team: {dt_party['name']}")
    logger.info(f"  ReBeL team: {rebel_party['name']}")

    # DT選出
    start = time.time()
    dt_selection = dt_ai.get_selection(dt_team_names, rebel_team_names)
    dt_think_time += time.time() - start

    # ReBeL選出
    start = time.time()
    rebel_selection = rebel_ai.get_selection(rebel_team_names, dt_team_names)
    rebel_think_time += time.time() - start

    logger.info(f"  DT selection: {[dt_team_names[i] for i in dt_selection[:3]]}")
    logger.info(
        f"  ReBeL selection: {[rebel_team_names[i] for i in rebel_selection[:3]]}"
    )

    # バトルセットアップ
    battle = setup_battle(
        dt_party_data,
        rebel_party_data,
        dt_selection[:3],
        rebel_selection[:3],
    )

    # 信念状態を初期化
    rebel_ai.init_belief_state(
        player=1,
        opponent_pokemon_names=[dt_team_names[i] for i in dt_selection[:3]],
    )

    # バトルログに選出を記録
    battle_log.append(
        {
            "turn": 0,
            "player": "dt",
            "message": f"選出: {', '.join([dt_team_names[i] for i in dt_selection[:3]])}",
        }
    )
    battle_log.append(
        {
            "turn": 0,
            "player": "rebel",
            "message": f"選出: {', '.join([rebel_team_names[i] for i in rebel_selection[:3]])}",
        }
    )

    # バトルループ
    turn = 0
    while turn < max_turns:
        turn += 1
        winner = battle.winner()
        if winner is not None:
            break

        # 降参チェック
        if dt_ai.should_surrender(battle, 0):
            surrendered_by = "dt"
            battle_log.append(
                {
                    "turn": turn,
                    "player": "dt",
                    "message": "Decision Transformer が降参した！",
                }
            )
            break

        if rebel_ai.should_surrender(battle, 1):
            surrendered_by = "rebel"
            battle_log.append(
                {
                    "turn": turn,
                    "player": "rebel",
                    "message": "ReBeL が降参した！",
                }
            )
            break

        # フェーズ判定
        phase = "battle"
        if battle.pokemon[0] is None or battle.pokemon[0].hp <= 0:
            phase = "change"
        if battle.pokemon[1] is None or battle.pokemon[1].hp <= 0:
            phase = "change"

        # ターン開始時の状態を記録
        dt_pokemon = battle.pokemon[0]
        rebel_pokemon = battle.pokemon[1]

        turn_start_state = {
            "dt_active": dt_pokemon.name if dt_pokemon else None,
            "dt_hp": f"{dt_pokemon.hp}/{dt_pokemon.status[0]}" if dt_pokemon else "N/A",
            "dt_hp_pct": int(dt_pokemon.hp / dt_pokemon.status[0] * 100)
            if dt_pokemon and dt_pokemon.status[0] > 0
            else 0,
            "rebel_active": rebel_pokemon.name if rebel_pokemon else None,
            "rebel_hp": f"{rebel_pokemon.hp}/{rebel_pokemon.status[0]}"
            if rebel_pokemon
            else "N/A",
            "rebel_hp_pct": int(rebel_pokemon.hp / rebel_pokemon.status[0] * 100)
            if rebel_pokemon and rebel_pokemon.status[0] > 0
            else 0,
        }

        # コマンド取得
        if phase == "battle":
            # DTのコマンド
            start = time.time()
            dt_cmd = dt_ai.get_battle_command(battle, 0)
            dt_think_time += time.time() - start

            # ReBeLのコマンド
            start = time.time()
            rebel_cmd = rebel_ai.get_battle_command(battle, 1)
            rebel_think_time += time.time() - start
        else:
            # 交代フェーズ
            dt_cmd = Battle.SKIP
            rebel_cmd = Battle.SKIP

            if battle.pokemon[0] is None or battle.pokemon[0].hp <= 0:
                start = time.time()
                dt_cmd = dt_ai.get_change_command(battle, 0)
                dt_think_time += time.time() - start

            if battle.pokemon[1] is None or battle.pokemon[1].hp <= 0:
                start = time.time()
                rebel_cmd = rebel_ai.get_change_command(battle, 1)
                rebel_think_time += time.time() - start

        # コマンドを人間が読める形式に変換
        def cmd_to_str(cmd: int, player: int, battle: Battle) -> str:
            pokemon = battle.pokemon[player]
            if cmd == Battle.SKIP:
                return "SKIP"
            elif cmd == Battle.STRUGGLE:
                return "わるあがき"
            elif 0 <= cmd <= 3:
                if pokemon and cmd < len(pokemon.moves):
                    return f"技: {pokemon.moves[cmd]}"
                return f"技{cmd}"
            elif 10 <= cmd <= 13:
                move_idx = cmd - 10
                if pokemon and move_idx < len(pokemon.moves):
                    return f"テラス+{pokemon.moves[move_idx]}"
                return f"テラス+技{move_idx}"
            elif 20 <= cmd <= 25:
                idx = cmd - 20
                if idx < len(battle.selected[player]):
                    target = battle.selected[player][idx]
                    if target:
                        return f"交代→{target.name}"
                return f"交代{idx}"
            return f"cmd={cmd}"

        dt_cmd_str = cmd_to_str(dt_cmd, 0, battle)
        rebel_cmd_str = cmd_to_str(rebel_cmd, 1, battle)

        # ターン実行
        battle.proceed(commands=[dt_cmd, rebel_cmd])

        # ターン終了時の状態
        dt_pokemon_after = battle.pokemon[0]
        rebel_pokemon_after = battle.pokemon[1]

        turn_end_state = {
            "dt_active": dt_pokemon_after.name if dt_pokemon_after else None,
            "dt_hp_pct": int(dt_pokemon_after.hp / dt_pokemon_after.status[0] * 100)
            if dt_pokemon_after and dt_pokemon_after.status[0] > 0
            else 0,
            "rebel_active": rebel_pokemon_after.name if rebel_pokemon_after else None,
            "rebel_hp_pct": int(
                rebel_pokemon_after.hp / rebel_pokemon_after.status[0] * 100
            )
            if rebel_pokemon_after and rebel_pokemon_after.status[0] > 0
            else 0,
        }

        # 詳細ログを記録
        battle_log.append(
            {
                "turn": turn,
                "phase": phase,
                "before": turn_start_state,
                "dt_action": dt_cmd_str,
                "rebel_action": rebel_cmd_str,
                "after": turn_end_state,
            }
        )

        # 倒れたポケモンがいれば記録
        if dt_pokemon and (
            dt_pokemon_after is None
            or dt_pokemon_after.hp <= 0
            or dt_pokemon_after.name != dt_pokemon.name
        ):
            if dt_pokemon_after is None or dt_pokemon_after.hp <= 0:
                battle_log.append(
                    {
                        "turn": turn,
                        "event": "faint",
                        "player": "dt",
                        "pokemon": dt_pokemon.name,
                    }
                )
        if rebel_pokemon and (
            rebel_pokemon_after is None
            or rebel_pokemon_after.hp <= 0
            or rebel_pokemon_after.name != rebel_pokemon.name
        ):
            if rebel_pokemon_after is None or rebel_pokemon_after.hp <= 0:
                battle_log.append(
                    {
                        "turn": turn,
                        "event": "faint",
                        "player": "rebel",
                        "pokemon": rebel_pokemon.name,
                    }
                )

        # ReBeLに観測情報を伝える
        rebel_ai.observe_battle_state(battle, ai_player=1)

    # 結果判定
    winner = battle.winner()
    if surrendered_by == "dt":
        winner_str = "rebel"
    elif surrendered_by == "rebel":
        winner_str = "dt"
    elif winner == 0:
        winner_str = "dt"
    elif winner == 1:
        winner_str = "rebel"
    else:
        winner_str = "draw"

    # 残りポケモン数をカウント
    dt_remaining = sum(1 for p in battle.selected[0] if p is not None and p.hp > 0)
    rebel_remaining = sum(1 for p in battle.selected[1] if p is not None and p.hp > 0)

    result = MatchResult(
        match_id=match_id,
        winner=winner_str,
        total_turns=turn,
        dt_team_name=dt_party["name"],
        rebel_team_name=rebel_party["name"],
        dt_remaining=dt_remaining,
        rebel_remaining=rebel_remaining,
        dt_think_time=dt_think_time,
        rebel_think_time=rebel_think_time,
        surrendered_by=surrendered_by,
    )

    # リプレイ用ログを作成
    match_log = MatchLog(
        match_id=match_id,
        timestamp=datetime.now().isoformat(),
        result=result,
        teams={
            "dt": {
                "name": dt_party["name"],
                "selection": dt_selection[:3],
                "pokemon": [
                    {
                        "name": dt_party_data[i]["name"],
                        "item": dt_party_data[i].get("item", ""),
                        "ability": dt_party_data[i].get("ability", ""),
                        "tera_type": dt_party_data[i].get("Ttype", ""),
                        "moves": dt_party_data[i].get("moves", []),
                        "nature": dt_party_data[i].get("nature", ""),
                    }
                    for i in dt_selection[:3]
                ],
            },
            "rebel": {
                "name": rebel_party["name"],
                "selection": rebel_selection[:3],
                "pokemon": [
                    {
                        "name": rebel_party_data[i]["name"],
                        "item": rebel_party_data[i].get("item", ""),
                        "ability": rebel_party_data[i].get("ability", ""),
                        "tera_type": rebel_party_data[i].get("Ttype", ""),
                        "moves": rebel_party_data[i].get("moves", []),
                        "nature": rebel_party_data[i].get("nature", ""),
                    }
                    for i in rebel_selection[:3]
                ],
            },
        },
        final_state={
            "dt_pokemon": [
                {
                    "name": p.name if p else "???",
                    "hp_percent": int(p.hp / p.status[0] * 100)
                    if p and p.status[0] > 0
                    else 0,
                    "status": "active" if p == battle.pokemon[0] else "bench",
                }
                for p in battle.selected[0]
                if p is not None
            ],
            "rebel_pokemon": [
                {
                    "name": p.name if p else "???",
                    "hp_percent": int(p.hp / p.status[0] * 100)
                    if p and p.status[0] > 0
                    else 0,
                    "status": "active" if p == battle.pokemon[1] else "bench",
                }
                for p in battle.selected[1]
                if p is not None
            ],
        },
        battle_log=battle_log,
    )

    logger.info(
        f"Match {match_id}: {winner_str.upper()} wins! "
        f"(turns={turn}, DT remaining={dt_remaining}, ReBeL remaining={rebel_remaining})"
    )

    return result, match_log


def run_comparison(
    dt_checkpoint: str,
    rebel_checkpoint: str,
    trainer_jsons: list[str],
    num_matches: int,
    output_dir: str,
    usage_db_path: str,
    max_turns: int = 100,
    dt_use_mcts: bool = False,
    dt_mcts_simulations: int = 200,
    device: str = "cpu",
) -> dict[str, Any]:
    """対決を実行"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # パーティプールを読み込み
    party_pool = load_trainer_pools(trainer_jsons)
    if len(party_pool) == 0:
        raise ValueError("No parties found in trainer files")

    # Pokemonデータを初期化
    Pokemon.init()

    # AIをロード
    logger.info(f"Loading Decision Transformer from {dt_checkpoint}")
    dt_config = DecisionTransformerAIConfig(
        checkpoint_path=dt_checkpoint,
        device=device,
        use_mcts=dt_use_mcts,
        mcts_simulations=dt_mcts_simulations,
    )
    dt_ai = DecisionTransformerAI(dt_config)

    logger.info(f"Loading ReBeL from {rebel_checkpoint}")
    rebel_config = RebelAIConfig(
        checkpoint_path=rebel_checkpoint,
        usage_db_path=usage_db_path,
        device=device,
    )
    rebel_ai = RebelAI(rebel_config)

    # 結果を格納
    results: list[MatchResult] = []
    match_logs: list[MatchLog] = []

    # 対戦ループ
    for match_id in range(1, num_matches + 1):
        # 両モデルがランダムにパーティを選択（重複可）
        dt_party = random.choice(party_pool)
        rebel_party = random.choice(party_pool)

        try:
            result, match_log = run_single_match(
                match_id=match_id,
                dt_ai=dt_ai,
                rebel_ai=rebel_ai,
                dt_party=dt_party,
                rebel_party=rebel_party,
                max_turns=max_turns,
            )
            results.append(result)
            match_logs.append(match_log)

            # 進捗表示
            dt_wins = sum(1 for r in results if r.winner == "dt")
            rebel_wins = sum(1 for r in results if r.winner == "rebel")
            draws = sum(1 for r in results if r.winner == "draw")
            logger.info(
                f"Progress: {match_id}/{num_matches} | "
                f"DT: {dt_wins} ({dt_wins / match_id * 100:.1f}%) | "
                f"ReBeL: {rebel_wins} ({rebel_wins / match_id * 100:.1f}%) | "
                f"Draw: {draws}"
            )

            # 中間結果を保存（10試合ごと）
            if match_id % 10 == 0:
                _save_results(output_path, results, match_logs)

        except Exception as e:
            logger.error(f"Match {match_id} failed: {e}")
            import traceback

            traceback.print_exc()
            continue

    # 最終結果を保存
    _save_results(output_path, results, match_logs)

    # 統計を計算
    stats = _calculate_stats(results)
    stats_path = output_path / "stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    logger.info(f"Results saved to {output_path}")
    logger.info(f"Final stats: {json.dumps(stats, ensure_ascii=False, indent=2)}")

    return stats


def _save_results(
    output_path: Path,
    results: list[MatchResult],
    match_logs: list[MatchLog],
) -> None:
    """結果を保存"""
    # 結果サマリー
    results_path = output_path / "results.jsonl"
    with open(results_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")

    # 個別のバトルログ（リプレイ用）
    logs_dir = output_path / "battle_logs"
    logs_dir.mkdir(exist_ok=True)
    for log in match_logs:
        log_path = logs_dir / f"match_{log.match_id:04d}.json"
        log_data = {
            "match_id": log.match_id,
            "timestamp": log.timestamp,
            "result": asdict(log.result),
            "teams": log.teams,
            "final_state": log.final_state,
            "battle_log": log.battle_log,
        }
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(log_data, f, ensure_ascii=False, indent=2)


def _calculate_stats(results: list[MatchResult]) -> dict[str, Any]:
    """統計を計算"""
    if not results:
        return {"error": "No results"}

    total = len(results)
    dt_wins = sum(1 for r in results if r.winner == "dt")
    rebel_wins = sum(1 for r in results if r.winner == "rebel")
    draws = sum(1 for r in results if r.winner == "draw")

    dt_surrenders = sum(1 for r in results if r.surrendered_by == "dt")
    rebel_surrenders = sum(1 for r in results if r.surrendered_by == "rebel")

    avg_turns = sum(r.total_turns for r in results) / total
    avg_dt_remaining = sum(r.dt_remaining for r in results) / total
    avg_rebel_remaining = sum(r.rebel_remaining for r in results) / total

    avg_dt_think_time = sum(r.dt_think_time for r in results) / total
    avg_rebel_think_time = sum(r.rebel_think_time for r in results) / total

    return {
        "total_matches": total,
        "dt_wins": dt_wins,
        "dt_win_rate": dt_wins / total,
        "rebel_wins": rebel_wins,
        "rebel_win_rate": rebel_wins / total,
        "draws": draws,
        "draw_rate": draws / total,
        "dt_surrenders": dt_surrenders,
        "rebel_surrenders": rebel_surrenders,
        "avg_turns": avg_turns,
        "avg_dt_remaining": avg_dt_remaining,
        "avg_rebel_remaining": avg_rebel_remaining,
        "avg_dt_think_time_sec": avg_dt_think_time,
        "avg_rebel_think_time_sec": avg_rebel_think_time,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Decision Transformer vs ReBeL+Selection BERT 対決"
    )
    parser.add_argument(
        "--dt-checkpoint",
        type=str,
        required=True,
        help="Decision Transformerのチェックポイントパス",
    )
    parser.add_argument(
        "--rebel-checkpoint",
        type=str,
        required=True,
        help="ReBeLのチェックポイントパス",
    )
    parser.add_argument(
        "--trainer-jsons",
        type=str,
        nargs="+",
        default=[
            "data/top_rankers/season_35.json",
            "data/top_rankers/season_36.json",
            "data/my_fixed_party.json",
        ],
        help="トレーナーJSONファイル（複数指定可）",
    )
    parser.add_argument(
        "--num-matches",
        type=int,
        default=10,
        help="対戦数（デフォルト: 10）",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/dt_vs_rebel",
        help="出力ディレクトリ",
    )
    parser.add_argument(
        "--usage-db",
        type=str,
        default="data/pokedb_usage/season_37_top150.json",
        help="使用率データベース（ReBeL用）",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=100,
        help="最大ターン数（デフォルト: 100）",
    )
    parser.add_argument(
        "--dt-use-mcts",
        action="store_true",
        help="Decision TransformerでMCTSを使用",
    )
    parser.add_argument(
        "--dt-mcts-simulations",
        type=int,
        default=200,
        help="MCTS シミュレーション回数（--dt-use-mcts時）",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="デバイス（cpu/cuda）",
    )

    args = parser.parse_args()

    stats = run_comparison(
        dt_checkpoint=args.dt_checkpoint,
        rebel_checkpoint=args.rebel_checkpoint,
        trainer_jsons=args.trainer_jsons,
        num_matches=args.num_matches,
        output_dir=args.output,
        usage_db_path=args.usage_db,
        max_turns=args.max_turns,
        dt_use_mcts=args.dt_use_mcts,
        dt_mcts_simulations=args.dt_mcts_simulations,
        device=args.device,
    )

    # 最終結果を表示
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"Total matches: {stats['total_matches']}")
    print(
        f"Decision Transformer wins: {stats['dt_wins']} ({stats['dt_win_rate'] * 100:.1f}%)"
    )
    print(f"ReBeL wins: {stats['rebel_wins']} ({stats['rebel_win_rate'] * 100:.1f}%)")
    print(f"Draws: {stats['draws']} ({stats['draw_rate'] * 100:.1f}%)")
    print(f"Average turns: {stats['avg_turns']:.1f}")
    print(f"Average DT think time: {stats['avg_dt_think_time_sec']:.2f}s")
    print(f"Average ReBeL think time: {stats['avg_rebel_think_time_sec']:.2f}s")


if __name__ == "__main__":
    main()
