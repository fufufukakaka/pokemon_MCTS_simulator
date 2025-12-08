#!/usr/bin/env python3
"""
CSVファイルをトレーナーJSON形式に変換するスクリプト

Usage:
    uv run python scripts/convert_csv_to_trainer_json.py \
        --input data/season_36_pokemon_data.csv \
        --output data/top_rankers/season_36.json
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_effort(effort_str: str) -> list[int]:
    """努力値文字列をリストに変換"""
    if not effort_str or effort_str.strip() == "":
        return [0, 0, 0, 0, 0, 0]
    try:
        values = [int(x.strip()) for x in effort_str.split(",")]
        if len(values) != 6:
            return [0, 0, 0, 0, 0, 0]
        return values
    except ValueError:
        return [0, 0, 0, 0, 0, 0]


def parse_moves(moves_str: str) -> list[str]:
    """技文字列をリストに変換"""
    if not moves_str or moves_str.strip() == "":
        return []
    return [m.strip() for m in moves_str.split(",") if m.strip()]


def convert_csv_to_trainer_json(input_path: Path, output_path: Path) -> None:
    """CSVをトレーナーJSON形式に変換"""
    trainers = []

    with open(input_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            trainer = {
                "rank": int(row["rank"]) if row.get("rank") else 0,
                "rating": float(row["rating"]) if row.get("rating") else 0.0,
                "name": row.get("trainer_name", ""),
                "pokemons": [],
            }

            # 6体のポケモンを処理
            for i in range(1, 7):
                prefix = f"pokemon{i}_"

                name = row.get(f"{prefix}name", "")
                if not name or name.strip() == "":
                    continue

                pokemon = {
                    "name": name.strip(),
                    "item": row.get(f"{prefix}item", "").strip(),
                    "nature": row.get(f"{prefix}nature", "").strip(),
                    "ability": row.get(f"{prefix}ability", "").strip(),
                    "Ttype": row.get(f"{prefix}Ttype", "").strip(),
                    "moves": parse_moves(row.get(f"{prefix}moves", "")),
                    "effort": parse_effort(row.get(f"{prefix}effort", "")),
                }

                trainer["pokemons"].append(pokemon)

            if trainer["pokemons"]:
                trainers.append(trainer)

    # 出力ディレクトリがなければ作成
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(trainers, f, ensure_ascii=False, indent=4)

    print(f"変換完了: {len(trainers)}人のトレーナーデータ")
    print(f"出力: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="CSVをトレーナーJSON形式に変換")
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        required=True,
        help="入力CSVファイル",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        required=True,
        help="出力JSONファイル",
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        print(f"エラー: 入力ファイルが見つかりません: {input_path}")
        return

    convert_csv_to_trainer_json(input_path, output_path)


if __name__ == "__main__":
    main()
