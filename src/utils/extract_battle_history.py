import glob
import re

import pandas as pd


def extract_battle_info(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        lines = file.readlines()

    # 情報を格納する辞書
    battle_info = {
        "player_0_name": None,
        "player_1_name": None,
        "player_0_team": None,
        "player_1_team": None,
        "player_0_selected": [],
        "player_1_selected": [],
        "player_0_first": None,
        "player_1_first": None,
        "winner_name": None,
    }

    # プレイヤー名とチームを抽出
    for line in lines:
        if " vs" in line:
            match = re.match(r"Player 0: (.+?) vs Player 1: (.+)", line)
            if match:
                battle_info["player_0_name"] = match.group(1)
                battle_info["player_1_name"] = match.group(2)

        if line.startswith("Player 0 team:"):
            battle_info["player_0_team"] = list(
                set(eval(line.split(": ", 1)[1].strip()))
            )
        elif line.startswith("Player 1 team:"):
            battle_info["player_1_team"] = list(
                set(eval(line.split(": ", 1)[1].strip()))
            )

        # 先発ポケモンを抽出
        if line.startswith("Player 0: ['交代"):
            match = re.search(r"Player 0: \['交代 -> (.+?)'", line)
            if match:
                battle_info["player_0_first"] = match.group(1)
        elif line.startswith("Player 1: ['交代"):
            match = re.search(r"Player 1: \['交代 -> (.+?)'", line)
            if match:
                battle_info["player_1_first"] = match.group(1)

        if "交代 ->" in line:
            if line.startswith("Player 0:"):
                selected = re.search(r"交代 -> (.+?)'", line)
                if selected:
                    battle_info["player_0_selected"].append(selected.group(1))
            elif line.startswith("Player 1:"):
                selected = re.search(r"交代 -> (.+?)'", line)
                if selected:
                    battle_info["player_1_selected"].append(selected.group(1))

        if line.startswith("勝者:"):
            match = re.match(r"勝者: Player \d, (.+)", line)
            if match:
                battle_info["winner_name"] = match.group(1)

    # set で重複を削除
    battle_info["player_0_selected"] = list(set(battle_info["player_0_selected"]))
    battle_info["player_1_selected"] = list(set(battle_info["player_1_selected"]))

    return battle_info


def main():
    # フォルダ内の全てのファイルを取得
    folder_path = "logs"
    file_paths = glob.glob(f"{folder_path}/*.txt")

    all_battle_info = []

    for file_path in file_paths:
        battle_info = extract_battle_info(file_path)
        all_battle_info.append(battle_info)

    # DataFrameに変換
    df = pd.DataFrame(all_battle_info)
    # CSVファイルとして保存
    output_file = "data/all_battle_history_season_27.csv"
    df.to_csv(output_file, index=False)


if __name__ == "__main__":
    main()
