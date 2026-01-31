#!/usr/bin/env python
"""
DT vs ReBeL 対戦ログ分析スクリプト

交代行動、テラスタル使用、ダメージ効率などを分析する。

Usage:
    uv run python scripts/analyze_battle_logs.py results/dt_vs_rebel/battle_logs
    uv run python scripts/analyze_battle_logs.py results/dt_vs_rebel/battle_logs --output results/analysis.json
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class PlayerStats:
    """プレイヤーの統計"""

    total_actions: int = 0
    switches: int = 0  # 交代回数（battleフェーズ）
    forced_switches: int = 0  # 強制交代回数（changeフェーズ）
    tera_uses: int = 0  # テラスタル使用回数
    moves: int = 0  # 技使用回数

    # 交代のタイミング
    switch_when_low_hp: int = 0  # HP50%以下で交代
    switch_when_high_hp: int = 0  # HP50%超で交代
    switch_vs_low_hp_opp: int = 0  # 相手HP50%以下で交代
    switch_vs_high_hp_opp: int = 0  # 相手HP50%超で交代

    # ダメージ統計
    total_damage_dealt: int = 0  # 与えたダメージ（HP%換算）
    total_damage_taken: int = 0  # 受けたダメージ（HP%換算）

    # 勝敗
    wins: int = 0
    losses: int = 0
    draws: int = 0

    # KO統計
    pokemon_koed: int = 0  # 倒されたポケモン数
    pokemon_kos: int = 0  # 倒したポケモン数

    # 連続行動
    same_move_streaks: list = field(default_factory=list)  # 同じ技を連続で使った回数


@dataclass
class MatchAnalysis:
    """1試合の分析結果"""

    match_id: str
    winner: str
    turns: int
    dt_switches: int = 0
    rebel_switches: int = 0
    dt_tera: bool = False
    rebel_tera: bool = False
    dt_damage_dealt: int = 0
    rebel_damage_dealt: int = 0


def analyze_match(match_data: dict) -> tuple[PlayerStats, PlayerStats, MatchAnalysis]:
    """1試合を分析"""
    dt_stats = PlayerStats()
    rebel_stats = PlayerStats()

    match_id = match_data.get("match_id", "unknown")
    result = match_data.get("result", {})
    winner = result.get("winner", "unknown")
    total_turns = result.get("total_turns", 0)

    # 勝敗記録
    if winner == "dt":
        dt_stats.wins += 1
        rebel_stats.losses += 1
    elif winner == "rebel":
        dt_stats.losses += 1
        rebel_stats.wins += 1
    else:
        dt_stats.draws += 1
        rebel_stats.draws += 1

    # 残りポケモン数からKO数を計算
    dt_remaining = result.get("dt_remaining", 0)
    rebel_remaining = result.get("rebel_remaining", 0)
    dt_stats.pokemon_koed = 3 - dt_remaining
    rebel_stats.pokemon_koed = 3 - rebel_remaining
    dt_stats.pokemon_kos = 3 - rebel_remaining
    rebel_stats.pokemon_kos = 3 - dt_remaining

    # 前の行動を追跡（連続使用検出用）
    dt_prev_action = None
    rebel_prev_action = None
    dt_streak = 0
    rebel_streak = 0

    analysis = MatchAnalysis(
        match_id=str(match_id),
        winner=winner,
        turns=total_turns,
    )

    battle_log = match_data.get("battle_log", [])

    for i, entry in enumerate(battle_log):
        # 新形式のログ
        if "dt_action" in entry and "rebel_action" in entry:
            phase = entry.get("phase", "battle")
            dt_action = entry["dt_action"]
            rebel_action = entry["rebel_action"]
            before = entry.get("before", {})
            after = entry.get("after", {})

            dt_hp_before = before.get("dt_hp_pct", 100)
            rebel_hp_before = before.get("rebel_hp_pct", 100)
            dt_hp_after = after.get("dt_hp_pct", 100)
            rebel_hp_after = after.get("rebel_hp_pct", 100)

            # ダメージ計算
            dt_damage = max(0, rebel_hp_before - rebel_hp_after)
            rebel_damage = max(0, dt_hp_before - dt_hp_after)
            dt_stats.total_damage_dealt += dt_damage
            dt_stats.total_damage_taken += rebel_damage
            rebel_stats.total_damage_dealt += rebel_damage
            rebel_stats.total_damage_taken += dt_damage
            analysis.dt_damage_dealt += dt_damage
            analysis.rebel_damage_dealt += rebel_damage

            if phase == "battle":
                dt_stats.total_actions += 1
                rebel_stats.total_actions += 1

                # DT の行動分析
                if "交代→" in dt_action:
                    dt_stats.switches += 1
                    analysis.dt_switches += 1
                    if dt_hp_before <= 50:
                        dt_stats.switch_when_low_hp += 1
                    else:
                        dt_stats.switch_when_high_hp += 1
                    if rebel_hp_before <= 50:
                        dt_stats.switch_vs_low_hp_opp += 1
                    else:
                        dt_stats.switch_vs_high_hp_opp += 1
                elif "テラス+" in dt_action:
                    dt_stats.tera_uses += 1
                    dt_stats.moves += 1
                    analysis.dt_tera = True
                else:
                    dt_stats.moves += 1

                # ReBeL の行動分析
                if "交代→" in rebel_action:
                    rebel_stats.switches += 1
                    analysis.rebel_switches += 1
                    if rebel_hp_before <= 50:
                        rebel_stats.switch_when_low_hp += 1
                    else:
                        rebel_stats.switch_when_high_hp += 1
                    if dt_hp_before <= 50:
                        rebel_stats.switch_vs_low_hp_opp += 1
                    else:
                        rebel_stats.switch_vs_high_hp_opp += 1
                elif "テラス+" in rebel_action:
                    rebel_stats.tera_uses += 1
                    rebel_stats.moves += 1
                    analysis.rebel_tera = True
                else:
                    rebel_stats.moves += 1

                # 連続行動検出
                if dt_action == dt_prev_action and "技:" in dt_action:
                    dt_streak += 1
                else:
                    if dt_streak >= 3:
                        dt_stats.same_move_streaks.append(dt_streak)
                    dt_streak = 1
                dt_prev_action = dt_action

                if rebel_action == rebel_prev_action and "技:" in rebel_action:
                    rebel_streak += 1
                else:
                    if rebel_streak >= 3:
                        rebel_stats.same_move_streaks.append(rebel_streak)
                    rebel_streak = 1
                rebel_prev_action = rebel_action

            elif phase == "change":
                # 強制交代
                if "交代→" in dt_action:
                    dt_stats.forced_switches += 1
                if "交代→" in rebel_action:
                    rebel_stats.forced_switches += 1

        # 倒れたイベント
        elif entry.get("event") == "faint":
            pass  # 既にresultから計算済み

    # 最後のストリーク
    if dt_streak >= 3:
        dt_stats.same_move_streaks.append(dt_streak)
    if rebel_streak >= 3:
        rebel_stats.same_move_streaks.append(rebel_streak)

    return dt_stats, rebel_stats, analysis


def analyze_all_matches(log_dir: Path) -> dict[str, Any]:
    """全試合を分析"""
    dt_total = PlayerStats()
    rebel_total = PlayerStats()
    match_analyses = []

    log_files = sorted(log_dir.glob("match_*.json"))

    for log_file in log_files:
        with open(log_file, "r", encoding="utf-8") as f:
            match_data = json.load(f)

        dt_stats, rebel_stats, analysis = analyze_match(match_data)
        match_analyses.append(analysis)

        # 累積
        for attr in [
            "total_actions",
            "switches",
            "forced_switches",
            "tera_uses",
            "moves",
            "switch_when_low_hp",
            "switch_when_high_hp",
            "switch_vs_low_hp_opp",
            "switch_vs_high_hp_opp",
            "total_damage_dealt",
            "total_damage_taken",
            "wins",
            "losses",
            "draws",
            "pokemon_koed",
            "pokemon_kos",
        ]:
            setattr(dt_total, attr, getattr(dt_total, attr) + getattr(dt_stats, attr))
            setattr(
                rebel_total,
                attr,
                getattr(rebel_total, attr) + getattr(rebel_stats, attr),
            )

        dt_total.same_move_streaks.extend(dt_stats.same_move_streaks)
        rebel_total.same_move_streaks.extend(rebel_stats.same_move_streaks)

    return {
        "total_matches": len(log_files),
        "dt_stats": dt_total,
        "rebel_stats": rebel_total,
        "match_analyses": match_analyses,
    }


def print_report(results: dict[str, Any]) -> None:
    """分析レポートを出力"""
    total = results["total_matches"]
    dt = results["dt_stats"]
    rebel = results["rebel_stats"]

    print("=" * 70)
    print("DT vs ReBeL 詳細分析レポート")
    print("=" * 70)
    print(f"分析試合数: {total}")
    print()

    # 勝敗
    print("【勝敗】")
    print(
        f"  DT:    {dt.wins}勝 {dt.losses}敗 {dt.draws}分 (勝率: {dt.wins / total * 100:.1f}%)"
    )
    print(
        f"  ReBeL: {rebel.wins}勝 {rebel.losses}敗 {rebel.draws}分 (勝率: {rebel.wins / total * 100:.1f}%)"
    )
    print()

    # 交代統計
    print("【交代行動（battleフェーズ）】")
    dt_switch_rate = dt.switches / dt.total_actions * 100 if dt.total_actions > 0 else 0
    rebel_switch_rate = (
        rebel.switches / rebel.total_actions * 100 if rebel.total_actions > 0 else 0
    )
    print(
        f"  DT:    {dt.switches}回 / {dt.total_actions}ターン ({dt_switch_rate:.1f}%)"
    )
    print(
        f"  ReBeL: {rebel.switches}回 / {rebel.total_actions}ターン ({rebel_switch_rate:.1f}%)"
    )
    print()

    print("【交代タイミング分析】")
    print("  ＜自分のHP状況＞")
    print(
        f"    DT    - HP≤50%で交代: {dt.switch_when_low_hp}回, HP>50%で交代: {dt.switch_when_high_hp}回"
    )
    print(
        f"    ReBeL - HP≤50%で交代: {rebel.switch_when_low_hp}回, HP>50%で交代: {rebel.switch_when_high_hp}回"
    )
    print("  ＜相手のHP状況＞")
    print(
        f"    DT    - 相手HP≤50%で交代: {dt.switch_vs_low_hp_opp}回, 相手HP>50%で交代: {dt.switch_vs_high_hp_opp}回"
    )
    print(
        f"    ReBeL - 相手HP≤50%で交代: {rebel.switch_vs_low_hp_opp}回, 相手HP>50%で交代: {rebel.switch_vs_high_hp_opp}回"
    )
    print()

    # 強制交代
    print("【強制交代（ポケモンが倒れた後）】")
    print(f"  DT:    {dt.forced_switches}回")
    print(f"  ReBeL: {rebel.forced_switches}回")
    print()

    # テラスタル
    print("【テラスタル使用】")
    print(f"  DT:    {dt.tera_uses}回 ({dt.tera_uses / total:.1f}回/試合)")
    print(f"  ReBeL: {rebel.tera_uses}回 ({rebel.tera_uses / total:.1f}回/試合)")
    print()

    # ダメージ効率
    print("【ダメージ効率（HP%換算）】")
    print(
        f"  DT    - 与ダメージ: {dt.total_damage_dealt}%, 被ダメージ: {dt.total_damage_taken}%"
    )
    print(f"          効率: {dt.total_damage_dealt - dt.total_damage_taken:+}%")
    print(
        f"  ReBeL - 与ダメージ: {rebel.total_damage_dealt}%, 被ダメージ: {rebel.total_damage_taken}%"
    )
    print(f"          効率: {rebel.total_damage_dealt - rebel.total_damage_taken:+}%")
    print()

    # KO統計
    print("【KO統計】")
    print(
        f"  DT    - 倒したポケモン: {dt.pokemon_kos}, 倒されたポケモン: {dt.pokemon_koed}"
    )
    print(
        f"          KO比率: {dt.pokemon_kos / dt.pokemon_koed:.2f}"
        if dt.pokemon_koed > 0
        else "          KO比率: N/A"
    )
    print(
        f"  ReBeL - 倒したポケモン: {rebel.pokemon_kos}, 倒されたポケモン: {rebel.pokemon_koed}"
    )
    print(
        f"          KO比率: {rebel.pokemon_kos / rebel.pokemon_koed:.2f}"
        if rebel.pokemon_koed > 0
        else "          KO比率: N/A"
    )
    print()

    # 連続行動
    print("【同じ技の連続使用（3回以上）】")
    dt_streaks = dt.same_move_streaks
    rebel_streaks = rebel.same_move_streaks
    print(
        f"  DT:    {len(dt_streaks)}回発生, 最長: {max(dt_streaks) if dt_streaks else 0}連続"
    )
    print(
        f"  ReBeL: {len(rebel_streaks)}回発生, 最長: {max(rebel_streaks) if rebel_streaks else 0}連続"
    )
    if dt_streaks:
        print(f"         DT連続回数分布: {sorted(dt_streaks, reverse=True)[:10]}")
    print()

    # 交代と勝率の相関
    print("【交代回数と勝率の相関】")
    analyses = results["match_analyses"]

    # DT交代あり vs なし
    dt_switch_wins = sum(1 for a in analyses if a.dt_switches > 0 and a.winner == "dt")
    dt_switch_total = sum(1 for a in analyses if a.dt_switches > 0)
    dt_no_switch_wins = sum(
        1 for a in analyses if a.dt_switches == 0 and a.winner == "dt"
    )
    dt_no_switch_total = sum(1 for a in analyses if a.dt_switches == 0)

    print(
        f"  DT が交代した試合:   {dt_switch_wins}/{dt_switch_total} 勝 ({dt_switch_wins / dt_switch_total * 100:.1f}%)"
        if dt_switch_total > 0
        else "  DT が交代した試合:   データなし"
    )
    print(
        f"  DT が交代しない試合: {dt_no_switch_wins}/{dt_no_switch_total} 勝 ({dt_no_switch_wins / dt_no_switch_total * 100:.1f}%)"
        if dt_no_switch_total > 0
        else "  DT が交代しない試合: データなし"
    )

    # ReBeL交代回数別
    rebel_many_switch = [(a.rebel_switches, a.winner == "rebel") for a in analyses]
    high_switch = [w for s, w in rebel_many_switch if s >= 3]
    low_switch = [w for s, w in rebel_many_switch if s < 3]
    print(
        f"  ReBeL 3回以上交代: {sum(high_switch)}/{len(high_switch)} 勝 ({sum(high_switch) / len(high_switch) * 100:.1f}%)"
        if high_switch
        else "  ReBeL 3回以上交代: データなし"
    )
    print(
        f"  ReBeL 3回未満交代: {sum(low_switch)}/{len(low_switch)} 勝 ({sum(low_switch) / len(low_switch) * 100:.1f}%)"
        if low_switch
        else "  ReBeL 3回未満交代: データなし"
    )
    print()

    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="DT vs ReBeL 対戦ログ分析")
    parser.add_argument("log_dir", type=str, help="battle_logs ディレクトリのパス")
    parser.add_argument(
        "--output", "-o", type=str, help="JSON出力先（省略時は標準出力のみ）"
    )

    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    if not log_dir.exists():
        print(f"Error: {log_dir} が存在しません")
        return

    results = analyze_all_matches(log_dir)
    print_report(results)

    if args.output:
        output_path = Path(args.output)
        # dataclass を dict に変換
        output_data = {
            "total_matches": results["total_matches"],
            "dt_stats": asdict(results["dt_stats"]),
            "rebel_stats": asdict(results["rebel_stats"]),
            "match_analyses": [asdict(a) for a in results["match_analyses"]],
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        print(f"\n結果を {output_path} に保存しました")


if __name__ == "__main__":
    main()
