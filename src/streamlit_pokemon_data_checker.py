"""抽出されたポケモンデータをチェックするStreamlitアプリ."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

# --- Constants ---
DATA_DIR = Path(__file__).parent.parent / "data"
POKEMON_SLOTS = 6
DEFAULT_PAGE_SIZE = 10

VALID_TYPES = [
    "ノーマル",
    "ほのお",
    "みず",
    "でんき",
    "くさ",
    "こおり",
    "かくとう",
    "どく",
    "じめん",
    "ひこう",
    "エスパー",
    "むし",
    "いわ",
    "ゴースト",
    "ドラゴン",
    "あく",
    "はがね",
    "フェアリー",
    "ステラ",
]


# --- Data Loading ---
@st.cache_data
def load_master_pokemon_names() -> list[str]:
    """ポケモン名のマスターデータを読み込む."""
    df = pd.read_csv(DATA_DIR / "zukan.txt", sep="\t")
    return sorted(df["Name"].tolist())


@st.cache_data
def load_master_move_names() -> list[str]:
    """技名のマスターデータを読み込む."""
    df = pd.read_csv(DATA_DIR / "move.txt", sep="\t")
    return sorted(df["Name"].tolist())


@st.cache_data
def load_master_item_names() -> list[str]:
    """アイテム名のマスターデータを読み込む."""
    df = pd.read_csv(DATA_DIR / "item.txt", sep="\t")
    return sorted(df["Name"].tolist())


@st.cache_data
def load_master_nature_names() -> list[str]:
    """性格名のマスターデータを読み込む."""
    df = pd.read_csv(DATA_DIR / "nature.txt", sep=" ", header=None)
    df.columns = ["Name", "1", "2", "3", "4", "5", "6"]
    return sorted(df["Name"].tolist())


@st.cache_data
def load_master_ability_names() -> list[str]:
    """特性名のマスターデータを読み込む."""
    df = pd.read_csv(DATA_DIR / "zukan.txt", sep="\t")
    abilities = set()
    for col in ["Ability1", "Ability2", "Ability3", "Ability4"]:
        abilities.update(df[col].dropna().tolist())
    abilities.discard("-")
    return sorted(abilities)


@st.cache_data
def load_pokemon_to_abilities() -> dict[str, list[str]]:
    """ポケモンごとの特性対応を取得."""
    df = pd.read_csv(DATA_DIR / "zukan.txt", sep="\t")
    result = {}
    for _, row in df.iterrows():
        name = row["Name"]
        abilities = []
        for col in ["Ability1", "Ability2", "Ability3", "Ability4"]:
            if pd.notna(row[col]) and row[col] != "-":
                abilities.append(row[col])
        result[name] = abilities
    return result


@st.cache_data
def get_available_csv_files() -> list[Path]:
    """利用可能なCSVファイルを取得."""
    return sorted(DATA_DIR.glob("season_*_pokemon_data.csv"))


def load_extracted_data(file_path: Path) -> pd.DataFrame:
    """抽出されたCSVデータを読み込む."""
    return pd.read_csv(file_path)


# --- Validation Functions ---
def validate_pokemon_name(name: str, master_names: set[str]) -> tuple[bool, str]:
    """ポケモン名を検証."""
    if pd.isna(name) or name == "" or name == "不明":
        return False, "欠損または不明"
    if name not in master_names:
        return False, f"マスターに存在しない: {name}"
    return True, ""


def validate_move_name(move: str, master_moves: set[str]) -> tuple[bool, str]:
    """技名を検証."""
    if pd.isna(move) or move == "" or move == "-":
        return True, ""  # 空は許容
    if move not in master_moves:
        return False, f"マスターに存在しない: {move}"
    return True, ""


def validate_item_name(item: str, master_items: set[str]) -> tuple[bool, str]:
    """アイテム名を検証."""
    if pd.isna(item) or item == "" or item == "-" or item == "不明":
        return False, "欠損または不明"
    if item not in master_items:
        return False, f"マスターに存在しない: {item}"
    return True, ""


def validate_nature_name(nature: str, master_natures: set[str]) -> tuple[bool, str]:
    """性格名を検証."""
    if pd.isna(nature) or nature == "" or nature == "不明":
        return False, "欠損または不明"
    if nature not in master_natures:
        return False, f"マスターに存在しない: {nature}"
    return True, ""


def validate_ability(
    ability: str,
    pokemon_name: str,
    master_abilities: set[str],
    pokemon_to_abilities: dict[str, list[str]],
) -> tuple[bool, str]:
    """特性を検証."""
    if pd.isna(ability) or ability == "" or ability == "不明":
        return False, "欠損または不明"
    if ability not in master_abilities:
        return False, f"マスターに存在しない: {ability}"
    # ポケモンとの対応チェック
    if pokemon_name in pokemon_to_abilities:
        valid_abilities = pokemon_to_abilities[pokemon_name]
        if ability not in valid_abilities:
            return False, f"{pokemon_name}の特性ではない (候補: {valid_abilities})"
    return True, ""


def validate_effort(effort_str: str) -> tuple[bool, str]:
    """努力値を検証."""
    if pd.isna(effort_str) or effort_str == "":
        return False, "欠損"
    try:
        values = [int(v.strip()) for v in effort_str.split(",")]
        if len(values) != 6:
            return False, f"6個ではない: {len(values)}個"
        total = sum(values)
        if total > 512:
            return False, f"合計512超過: {total}"
        for v in values:
            if v < 0 or v > 252:
                return False, f"0-252の範囲外: {v}"
        return True, ""
    except ValueError:
        return False, f"パースエラー: {effort_str}"


def validate_ttype(ttype: str) -> tuple[bool, str]:
    """テラスタルタイプを検証."""
    if pd.isna(ttype) or ttype == "" or ttype == "不明":
        return False, "欠損または不明"
    if ttype not in VALID_TYPES:
        return False, f"無効なタイプ: {ttype}"
    return True, ""


def validate_row(
    row: pd.Series,
    master_pokemon: set[str],
    master_moves: set[str],
    master_items: set[str],
    master_natures: set[str],
    master_abilities: set[str],
    pokemon_to_abilities: dict[str, list[str]],
) -> dict:
    """1行（1トレーナー）のデータを検証."""
    errors = []
    warnings = []

    for i in range(1, POKEMON_SLOTS + 1):
        prefix = f"pokemon{i}"
        pokemon_name = row.get(f"{prefix}_name", "")

        # ポケモン名
        is_valid, msg = validate_pokemon_name(pokemon_name, master_pokemon)
        if not is_valid:
            errors.append(f"P{i} 名前: {msg}")

        # アイテム
        item = row.get(f"{prefix}_item", "")
        is_valid, msg = validate_item_name(item, master_items)
        if not is_valid:
            errors.append(f"P{i} アイテム: {msg}")

        # 性格
        nature = row.get(f"{prefix}_nature", "")
        is_valid, msg = validate_nature_name(nature, master_natures)
        if not is_valid:
            errors.append(f"P{i} 性格: {msg}")

        # 特性
        ability = row.get(f"{prefix}_ability", "")
        is_valid, msg = validate_ability(
            ability, pokemon_name, master_abilities, pokemon_to_abilities
        )
        if not is_valid:
            errors.append(f"P{i} 特性: {msg}")

        # テラスタルタイプ
        ttype = row.get(f"{prefix}_Ttype", "")
        is_valid, msg = validate_ttype(ttype)
        if not is_valid:
            errors.append(f"P{i} テラスタル: {msg}")

        # 技
        moves_str = row.get(f"{prefix}_moves", "")
        if pd.notna(moves_str) and moves_str != "":
            moves = [m.strip() for m in str(moves_str).split(",")]
            for j, move in enumerate(moves, 1):
                is_valid, msg = validate_move_name(move, master_moves)
                if not is_valid:
                    errors.append(f"P{i} 技{j}: {msg}")
            if len(moves) != 4:
                warnings.append(f"P{i} 技の数: {len(moves)}個 (通常4個)")
        else:
            errors.append(f"P{i} 技: 欠損")

        # 努力値
        effort = row.get(f"{prefix}_effort", "")
        is_valid, msg = validate_effort(str(effort) if pd.notna(effort) else "")
        if not is_valid:
            errors.append(f"P{i} 努力値: {msg}")

    return {"errors": errors, "warnings": warnings}


def get_field_status(value: str, field_type: str, masters: dict, pokemon_name: str = "") -> str:
    """フィールドの検証状態を取得."""
    if field_type == "pokemon":
        is_valid, _ = validate_pokemon_name(value, masters["pokemon"])
        return "" if is_valid else "error"
    elif field_type == "item":
        is_valid, _ = validate_item_name(value, masters["items"])
        return "" if is_valid else "error"
    elif field_type == "nature":
        is_valid, _ = validate_nature_name(value, masters["natures"])
        return "" if is_valid else "error"
    elif field_type == "ability":
        is_valid, _ = validate_ability(
            value, pokemon_name, masters["abilities"], masters["pokemon_to_abilities"]
        )
        return "" if is_valid else "error"
    elif field_type == "ttype":
        is_valid, _ = validate_ttype(value)
        return "" if is_valid else "error"
    elif field_type == "moves":
        if pd.isna(value) or value == "":
            return "error"
        moves = [m.strip() for m in str(value).split(",")]
        for move in moves:
            is_valid, _ = validate_move_name(move, masters["moves"])
            if not is_valid:
                return "error"
        return ""
    elif field_type == "effort":
        is_valid, _ = validate_effort(str(value) if pd.notna(value) else "")
        return "" if is_valid else "error"
    return ""


# --- Streamlit UI ---
def main():
    st.set_page_config(
        page_title="ポケモンデータチェッカー",
        page_icon="🔍",
        layout="wide",
    )

    st.title("ポケモン構築記事 抽出データチェッカー")
    st.markdown("抽出されたポケモン名、技名、アイテム名などがマスターデータと一致しているか確認・編集します。")

    # マスターデータ読み込み
    with st.spinner("マスターデータを読み込み中..."):
        master_pokemon = load_master_pokemon_names()
        master_moves = load_master_move_names()
        master_items = load_master_item_names()
        master_natures = load_master_nature_names()
        master_abilities = load_master_ability_names()
        pokemon_to_abilities = load_pokemon_to_abilities()

    masters = {
        "pokemon": set(master_pokemon),
        "moves": set(master_moves),
        "items": set(master_items),
        "natures": set(master_natures),
        "abilities": set(master_abilities),
        "pokemon_to_abilities": pokemon_to_abilities,
    }

    st.sidebar.header("設定")
    st.sidebar.markdown(f"- ポケモン数: {len(master_pokemon)}")
    st.sidebar.markdown(f"- 技数: {len(master_moves)}")
    st.sidebar.markdown(f"- アイテム数: {len(master_items)}")
    st.sidebar.markdown(f"- 性格数: {len(master_natures)}")
    st.sidebar.markdown(f"- 特性数: {len(master_abilities)}")

    # CSVファイル選択
    csv_files = get_available_csv_files()
    if not csv_files:
        st.error("data/ ディレクトリに season_*_pokemon_data.csv ファイルが見つかりません。")
        return

    selected_file = st.sidebar.selectbox(
        "CSVファイルを選択",
        csv_files,
        format_func=lambda x: x.name,
        key="selected_file",
    )

    # session_stateでデータを管理
    file_key = str(selected_file)
    if "loaded_file" not in st.session_state or st.session_state.loaded_file != file_key:
        st.session_state.df = load_extracted_data(selected_file)
        st.session_state.loaded_file = file_key
        st.session_state.has_changes = False

    df = st.session_state.df

    # 保存ボタン
    st.sidebar.subheader("保存")
    if st.session_state.get("has_changes", False):
        st.sidebar.warning("未保存の変更があります")
    if st.sidebar.button("CSVを保存", type="primary"):
        st.session_state.df.to_csv(selected_file, index=False)
        st.session_state.has_changes = False
        st.sidebar.success(f"保存しました: {selected_file.name}")
        st.rerun()

    # フィルターオプション
    st.sidebar.subheader("フィルター")
    show_only_errors = st.sidebar.checkbox("エラーのある行のみ表示", value=False)
    filter_error_type = st.sidebar.multiselect(
        "エラータイプでフィルター",
        ["名前", "アイテム", "性格", "特性", "テラスタル", "技", "努力値"],
        default=[],
    )

    # 編集モード
    st.sidebar.subheader("編集")
    edit_mode = st.sidebar.checkbox("編集モードを有効化", value=False)

    st.info(f"読み込んだレコード数: {len(df)}")

    # 全体統計
    validation_results = []
    for idx, row in df.iterrows():
        result = validate_row(
            row,
            masters["pokemon"],
            masters["moves"],
            masters["items"],
            masters["natures"],
            masters["abilities"],
            pokemon_to_abilities,
        )
        result["index"] = idx
        result["rank"] = row.get("rank", "")
        result["trainer_name"] = row.get("trainer_name", "")
        result["blog_url"] = row.get("blog_url", "")
        validation_results.append(result)

    # 統計表示
    total_errors = sum(len(r["errors"]) for r in validation_results)
    total_warnings = sum(len(r["warnings"]) for r in validation_results)
    rows_with_errors = sum(1 for r in validation_results if r["errors"])

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("総エラー数", total_errors)
    with col2:
        st.metric("エラーのある行", f"{rows_with_errors} / {len(df)}")
    with col3:
        st.metric("警告数", total_warnings)

    # エラー種別の集計
    st.subheader("エラー種別サマリー")
    error_categories = {}
    for result in validation_results:
        for error in result["errors"]:
            if ": " in error:
                parts = error.split(": ", 1)
                if " " in parts[0]:
                    category = parts[0].split(" ", 1)[1]
                    if category not in error_categories:
                        error_categories[category] = []
                    error_categories[category].append(parts[1])

    if error_categories:
        for category, errors in sorted(error_categories.items()):
            with st.expander(f"{category} ({len(errors)}件)"):
                error_counts = {}
                for e in errors:
                    error_counts[e] = error_counts.get(e, 0) + 1
                for error, count in sorted(
                    error_counts.items(), key=lambda x: -x[1]
                )[:20]:
                    st.text(f"  ({count}) {error}")

    # 詳細表示
    st.subheader("詳細チェック")

    # フィルタリング
    filtered_results = validation_results
    if show_only_errors:
        filtered_results = [r for r in filtered_results if r["errors"]]
    if filter_error_type:
        filtered_results = [
            r
            for r in filtered_results
            if any(
                any(t in e for t in filter_error_type)
                for e in r["errors"]
            )
        ]

    total_filtered = len(filtered_results)
    st.info(f"表示対象レコード数: {total_filtered}")

    # ページネーション設定
    st.sidebar.subheader("ページネーション")
    page_size = st.sidebar.selectbox(
        "1ページあたりの表示件数",
        [5, 10, 20, 50],
        index=1,
    )
    total_pages = max(1, (total_filtered + page_size - 1) // page_size)

    current_page = st.number_input(
        f"ページ (1-{total_pages})",
        min_value=1,
        max_value=total_pages,
        value=1,
        step=1,
    )

    start_idx = (current_page - 1) * page_size
    end_idx = min(start_idx + page_size, total_filtered)

    st.caption(f"{start_idx + 1} - {end_idx} / {total_filtered} 件を表示中")

    # ページ内のデータのみ表示
    for result in filtered_results[start_idx:end_idx]:
        row_idx = result["index"]
        row = df.iloc[row_idx]
        has_errors = bool(result["errors"])
        has_warnings = bool(result["warnings"])

        status_icon = "❌" if has_errors else ("⚠️" if has_warnings else "✅")
        header = f"{status_icon} Rank {result['rank']}: {result['trainer_name']}"

        with st.expander(header, expanded=has_errors):
            st.markdown(f"**URL**: [{result['blog_url']}]({result['blog_url']})")

            if result["errors"]:
                st.error("**エラー:**")
                for error in result["errors"]:
                    st.markdown(f"- {error}")

            if result["warnings"]:
                st.warning("**警告:**")
                for warning in result["warnings"]:
                    st.markdown(f"- {warning}")

            st.markdown("---")

            if edit_mode:
                # 編集モード: 入力フィールドを表示
                for i in range(1, POKEMON_SLOTS + 1):
                    prefix = f"pokemon{i}"
                    st.markdown(f"### ポケモン {i}")

                    col1, col2, col3 = st.columns(3)

                    with col1:
                        # ポケモン名
                        current_name = str(row.get(f"{prefix}_name", ""))
                        name_status = get_field_status(current_name, "pokemon", masters)
                        name_label = f"名前 {'❌' if name_status else ''}"
                        options = [""] + master_pokemon
                        try:
                            idx = options.index(current_name)
                        except ValueError:
                            idx = 0
                            options = [current_name] + options
                        new_name = st.selectbox(
                            name_label,
                            options,
                            index=idx,
                            key=f"name_{row_idx}_{i}",
                        )
                        if new_name != current_name:
                            st.session_state.df.at[row_idx, f"{prefix}_name"] = new_name
                            st.session_state.has_changes = True

                        # アイテム
                        current_item = str(row.get(f"{prefix}_item", ""))
                        item_status = get_field_status(current_item, "item", masters)
                        item_label = f"アイテム {'❌' if item_status else ''}"
                        options = [""] + master_items
                        try:
                            idx = options.index(current_item)
                        except ValueError:
                            idx = 0
                            options = [current_item] + options
                        new_item = st.selectbox(
                            item_label,
                            options,
                            index=idx,
                            key=f"item_{row_idx}_{i}",
                        )
                        if new_item != current_item:
                            st.session_state.df.at[row_idx, f"{prefix}_item"] = new_item
                            st.session_state.has_changes = True

                    with col2:
                        # 性格
                        current_nature = str(row.get(f"{prefix}_nature", ""))
                        nature_status = get_field_status(current_nature, "nature", masters)
                        nature_label = f"性格 {'❌' if nature_status else ''}"
                        options = [""] + master_natures
                        try:
                            idx = options.index(current_nature)
                        except ValueError:
                            idx = 0
                            options = [current_nature] + options
                        new_nature = st.selectbox(
                            nature_label,
                            options,
                            index=idx,
                            key=f"nature_{row_idx}_{i}",
                        )
                        if new_nature != current_nature:
                            st.session_state.df.at[row_idx, f"{prefix}_nature"] = new_nature
                            st.session_state.has_changes = True

                        # 特性
                        current_ability = str(row.get(f"{prefix}_ability", ""))
                        pokemon_name = str(row.get(f"{prefix}_name", ""))
                        ability_status = get_field_status(current_ability, "ability", masters, pokemon_name)
                        ability_label = f"特性 {'❌' if ability_status else ''}"
                        # このポケモンの特性候補を優先表示
                        pokemon_abilities = pokemon_to_abilities.get(pokemon_name, [])
                        other_abilities = [a for a in master_abilities if a not in pokemon_abilities]
                        options = [""] + pokemon_abilities + ["---"] + other_abilities
                        try:
                            idx = options.index(current_ability)
                        except ValueError:
                            idx = 0
                            options = [current_ability] + options
                        new_ability = st.selectbox(
                            ability_label,
                            options,
                            index=idx,
                            key=f"ability_{row_idx}_{i}",
                        )
                        if new_ability != current_ability and new_ability != "---":
                            st.session_state.df.at[row_idx, f"{prefix}_ability"] = new_ability
                            st.session_state.has_changes = True

                    with col3:
                        # テラスタルタイプ
                        current_ttype = str(row.get(f"{prefix}_Ttype", ""))
                        ttype_status = get_field_status(current_ttype, "ttype", masters)
                        ttype_label = f"テラスタル {'❌' if ttype_status else ''}"
                        options = [""] + VALID_TYPES
                        try:
                            idx = options.index(current_ttype)
                        except ValueError:
                            idx = 0
                            options = [current_ttype] + options
                        new_ttype = st.selectbox(
                            ttype_label,
                            options,
                            index=idx,
                            key=f"ttype_{row_idx}_{i}",
                        )
                        if new_ttype != current_ttype:
                            st.session_state.df.at[row_idx, f"{prefix}_Ttype"] = new_ttype
                            st.session_state.has_changes = True

                        # 努力値
                        current_effort = str(row.get(f"{prefix}_effort", ""))
                        effort_status = get_field_status(current_effort, "effort", masters)
                        effort_label = f"努力値 {'❌' if effort_status else ''}"
                        new_effort = st.text_input(
                            effort_label,
                            value=current_effort if current_effort != "nan" else "",
                            key=f"effort_{row_idx}_{i}",
                        )
                        if new_effort != current_effort:
                            st.session_state.df.at[row_idx, f"{prefix}_effort"] = new_effort
                            st.session_state.has_changes = True

                    # 技（全幅で表示）
                    current_moves = str(row.get(f"{prefix}_moves", ""))
                    moves_status = get_field_status(current_moves, "moves", masters)
                    moves_label = f"技 (カンマ区切り) {'❌' if moves_status else ''}"
                    new_moves = st.text_input(
                        moves_label,
                        value=current_moves if current_moves != "nan" else "",
                        key=f"moves_{row_idx}_{i}",
                    )
                    if new_moves != current_moves:
                        st.session_state.df.at[row_idx, f"{prefix}_moves"] = new_moves
                        st.session_state.has_changes = True

                    st.markdown("---")
            else:
                # 表示モード
                cols = st.columns(3)
                for i in range(1, POKEMON_SLOTS + 1):
                    prefix = f"pokemon{i}"
                    col_idx = (i - 1) % 3
                    with cols[col_idx]:
                        pokemon_name = row.get(f"{prefix}_name", "")
                        item = row.get(f"{prefix}_item", "")
                        nature = row.get(f"{prefix}_nature", "")
                        ability = row.get(f"{prefix}_ability", "")
                        ttype = row.get(f"{prefix}_Ttype", "")
                        moves = row.get(f"{prefix}_moves", "")
                        effort = row.get(f"{prefix}_effort", "")

                        # エラー状態をチェック
                        name_err = "❌ " if get_field_status(str(pokemon_name), "pokemon", masters) else ""
                        item_err = "❌ " if get_field_status(str(item), "item", masters) else ""
                        nature_err = "❌ " if get_field_status(str(nature), "nature", masters) else ""
                        ability_err = "❌ " if get_field_status(str(ability), "ability", masters, str(pokemon_name)) else ""
                        ttype_err = "❌ " if get_field_status(str(ttype), "ttype", masters) else ""
                        moves_err = "❌ " if get_field_status(str(moves), "moves", masters) else ""
                        effort_err = "❌ " if get_field_status(str(effort), "effort", masters) else ""

                        st.markdown(f"**{i}. {name_err}{pokemon_name}**")
                        st.markdown(f"- {item_err}アイテム: {item}")
                        st.markdown(f"- {nature_err}性格: {nature}")
                        st.markdown(f"- {ability_err}特性: {ability}")
                        st.markdown(f"- {ttype_err}テラス: {ttype}")
                        st.markdown(f"- {moves_err}技: {moves}")
                        st.markdown(f"- {effort_err}努力値: {effort}")


if __name__ == "__main__":
    main()
