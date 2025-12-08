#!/usr/bin/env python3
"""
pokedb.tokyo からポケモンの使用率データを取得するスクリプト

使用例:
    # 特定のポケモンのデータを取得
    uv run python scripts/fetch_pokedb_usage.py --pokemon-id 1003-00 --season 37

    # 複数のポケモンを取得（上位使用率ポケモン）
    uv run python scripts/fetch_pokedb_usage.py --top 50 --season 37

    # 既存のtrainer.jsonからポケモンリストを取得
    uv run python scripts/fetch_pokedb_usage.py --from-trainer data/top_rankers/season_36.json --season 37
"""

from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

# =============================================================================
# 図鑑番号から正規名へのマッピング（data/zukan.txt から自動生成）
# =============================================================================

# グローバルキャッシュ
_ZUKAN_ID_TO_NAME: dict[str, str] | None = None


def _get_zukan_data_path() -> Path:
    """zukan.txt のパスを取得"""
    # このスクリプトから相対的に data/zukan.txt を見つける
    script_dir = Path(__file__).parent
    return script_dir.parent / "data" / "zukan.txt"


def load_zukan_mapping() -> dict[str, str]:
    """
    zukan.txt から {図鑑ID: 正規名} のマッピングを作成

    zukan.txt 形式: "XXX" or "XXX-Y" (例: "25", "916-1")
    pokedb 形式: "XXXX-XX" (例: "0025-00", "0916-01")

    Returns:
        {pokedb形式ID: 正規名} の辞書
    """
    global _ZUKAN_ID_TO_NAME

    if _ZUKAN_ID_TO_NAME is not None:
        return _ZUKAN_ID_TO_NAME

    zukan_path = _get_zukan_data_path()
    if not zukan_path.exists():
        print(f"Warning: zukan.txt not found at {zukan_path}")
        _ZUKAN_ID_TO_NAME = {}
        return _ZUKAN_ID_TO_NAME

    mapping = {}

    with open(zukan_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # ヘッダーをスキップ
    for line in lines[1:]:
        parts = line.strip().split("\t")
        if len(parts) < 2:
            continue

        num = parts[0]
        name = parts[1]

        # zukan形式 -> pokedb形式に変換
        if "-" in num:
            base, form = num.split("-")
            pokedb_id = f"{int(base):04d}-{int(form):02d}"
        else:
            pokedb_id = f"{int(num):04d}-00"

        mapping[pokedb_id] = name

    _ZUKAN_ID_TO_NAME = mapping
    return _ZUKAN_ID_TO_NAME


def normalize_pokemon_name(pokemon_id: str, fallback_name: str) -> str:
    """
    pokemon_idを使って正規名を取得

    Args:
        pokemon_id: pokedb形式のID (例: "0898-02")
        fallback_name: 正規名が見つからない場合に使用する名前

    Returns:
        正規化されたポケモン名
    """
    mapping = load_zukan_mapping()
    return mapping.get(pokemon_id, fallback_name)


# =============================================================================
# ポケモン名から図鑑番号へのマッピング（主要なポケモンのみ）
# 完全なマッピングは別途データファイルから読み込むことを推奨
# =============================================================================
POKEMON_NAME_TO_ID: dict[str, str] = {
    # 伝説・幻
    "ミュウツー": "0150-00",
    "ルギア": "0249-00",
    "ホウオウ": "0250-00",
    "カイオーガ": "0382-00",
    "グラードン": "0383-00",
    "レックウザ": "0384-00",
    "ディアルガ": "0483-00",
    "パルキア": "0484-00",
    "ギラティナ": "0487-00",
    "ギラティナ(オリジン)": "0487-01",
    "アルセウス": "0493-00",
    "ゼクロム": "0644-00",
    "レシラム": "0643-00",
    "キュレム": "0646-00",
    "ブラックキュレム": "0646-02",
    "ホワイトキュレム": "0646-01",
    "ゼルネアス": "0716-00",
    "イベルタル": "0717-00",
    "ソルガレオ": "0791-00",
    "ルナアーラ": "0792-00",
    "ネクロズマ": "0800-00",
    "ネクロズマ(たそがれ)": "0800-01",
    "ネクロズマ(あかつき)": "0800-02",
    "ザシアン": "0888-00",
    "ザシアン(けんのおう)": "0888-01",
    "ザマゼンタ": "0889-00",
    "ザマゼンタ(たてのおう)": "0889-01",
    "ムゲンダイナ": "0890-00",
    "バドレックス": "0898-00",
    "バドレックス(こくば)": "0898-01",
    "バドレックス(はくば)": "0898-02",
    "バドレックス(こくばじょう)": "0898-01",
    "バドレックス(はくばじょう)": "0898-02",
    "コライドン": "1007-00",
    "ミライドン": "1008-00",
    "テラパゴス": "1024-00",
    "テラパゴス(テラスタル)": "1024-01",
    # SV新ポケモン
    "ディンルー": "1003-00",
    "チオンジェン": "1001-00",
    "パオジアン": "1002-00",
    "イーユイ": "1004-00",
    "ハバタクカミ": "0987-00",
    "テツノツツミ": "0990-00",
    "テツノブジン": "0988-00",
    "テツノカイナ": "0992-00",
    "テツノドクガ": "0994-00",
    "テツノコウベ": "0993-00",
    "テツノワダチ": "0991-00",
    "テツノイバラ": "0995-00",
    "テツノカシラ": "1006-00",
    "サケブシッポ": "0985-00",
    "アラブルタケ": "0986-00",
    "ハラバリー": "0939-00",
    "キラフロル": "0970-00",
    "オーガポン": "1017-00",
    "オーガポン(かまど)": "1017-01",
    "オーガポン(いど)": "1017-02",
    "オーガポン(いしずえ)": "1017-03",
    "ブリジュラス": "1018-00",
    "カミツオロチ": "1020-00",
    # 一般ポケモン（使用率上位）
    "カイリュー": "0149-00",
    "ガブリアス": "0445-00",
    "ウーラオス": "0892-00",
    "ウーラオス(れんげき)": "0892-01",
    "ウーラオス(いちげき)": "0892-00",
    "ランドロス": "0645-00",
    "ランドロス(れいじゅう)": "0645-01",
    "ランドロス(けしん)": "0645-00",
    "ヒードラン": "0485-00",
    "サーフゴー": "1000-00",
    "ドラパルト": "0887-00",
    "キョジオーン": "0954-00",
    "ドオー": "0980-00",
    "ヘイラッシャ": "0977-00",
    "シャリタツ": "0978-00",
    "ミミッキュ": "0778-00",
    "ガチグマ": "0901-00",
    "ガチグマ(アカツキ)": "0901-01",
    "グライオン": "0472-00",
    "クレセリア": "0488-00",
    "ポリゴン2": "0233-00",
    "ラッキー": "0113-00",
    "ドヒドイデ": "0748-00",
    "カバルドン": "0450-00",
    "エルフーン": "0547-00",
    "ラティオス": "0381-00",
    "ラティアス": "0380-00",
    "ボーマンダ": "0373-00",
    "メタグロス": "0376-00",
    "ロトム": "0479-00",
    "ヒートロトム": "0479-01",
    "ウォッシュロトム": "0479-02",
    "カットロトム": "0479-03",
    "フロストロトム": "0479-04",
    "スピンロトム": "0479-05",
    "キラーメ": "0969-00",
    "イルカマン": "0964-00",
    "イルカマン(マイティ)": "0964-01",
    "セグレイブ": "0998-00",
    "リザードン": "0006-00",
    "ゲンガー": "0094-00",
    "ギャラドス": "0130-00",
    "ラプラス": "0131-00",
    "カビゴン": "0143-00",
    "バンギラス": "0248-00",
    "キノガッサ": "0286-00",
    "ジバコイル": "0462-00",
    "エーフィ": "0196-00",
    "ブラッキー": "0197-00",
    "グレイシア": "0471-00",
    "ウルガモス": "0637-00",
    "キュウコン": "0038-00",
    "キュウコン(アローラ)": "0038-01",
    "ママンボウ": "0594-00",
    "アーマーガア": "0823-00",
    "モロバレル": "0591-00",
    "コノヨザル": "0979-00",
    "イダイナキバ": "0984-00",
    # 追加ポケモン
    "アローラベトベトン": "0089-01",
    "ベトベトン": "0089-00",
    "アローラペルシアン": "0053-01",
    "イエッサン": "0876-00",
    "イエッサン(メス)": "0876-01",
    "エルレイド": "0475-00",
    "エンニュート": "0758-00",
    "オオニューラ": "0903-00",
    "オーロンゲ": "0861-00",
    "カラミンゴ": "0973-00",
    "ガオガエン": "0727-00",
    "ガラルマタドガス": "0110-01",
    "ゴチルゼル": "0576-00",
    "サザンドラ": "0635-00",
    "ストリンダー": "0849-00",
    "ストリンダー(ハイ)": "0849-00",
    "ストリンダー(ロー)": "0849-01",
    "ソウブレイズ": "0937-00",
    "デカヌチャン": "0959-00",
    "ドデカバシ": "0733-00",
    "ハッサム": "0212-00",
    "ヒスイヌメルゴン": "0706-01",
    "フシギバナ": "0003-00",
    "ブリムオン": "0858-00",
    "ベラカス": "0955-00",
    "ミミズズ": "0968-00",
    "メタモン": "0132-00",
    "メレシー": "0703-00",
    "ラウドボーン": "0911-00",
    "ラグラージ": "0260-00",
    "レジエレキ": "0894-00",
}


@dataclass
class PokemonUsageData:
    """ポケモンの使用率データ"""

    pokemon_id: str
    pokemon_name: str
    season: int
    rule: int  # 0=シングル, 1=ダブル

    # 各種採用率データ {名前: 採用率}
    moves: dict[str, float] = field(default_factory=dict)
    items: dict[str, float] = field(default_factory=dict)
    abilities: dict[str, float] = field(default_factory=dict)
    tera_types: dict[str, float] = field(default_factory=dict)
    natures: dict[str, float] = field(default_factory=dict)

    # メタ情報
    fetched_at: str = ""
    source_url: str = ""


def fetch_pokemon_usage(
    pokemon_id: str,
    season: int = 37,
    rule: int = 0,
    delay: float = 1.0,
) -> Optional[PokemonUsageData]:
    """
    pokedb.tokyoから特定ポケモンの使用率データを取得

    Args:
        pokemon_id: ポケモンID（例: "1003-00"）
        season: シーズン番号
        rule: 0=シングル, 1=ダブル
        delay: リクエスト間の待機時間（秒）

    Returns:
        PokemonUsageData or None（取得失敗時）
    """
    url = (
        f"https://sv.pokedb.tokyo/pokemon/show/{pokemon_id}?season={season}&rule={rule}"
    )

    try:
        req = Request(url, headers={"User-Agent": "Pokemon-MCTS-Simulator/1.0"})
        with urlopen(req, timeout=30) as response:
            html = response.read().decode("utf-8")
    except (HTTPError, URLError) as e:
        print(f"  Error fetching {pokemon_id}: {e}")
        return None

    # ポケモン名を抽出（「の使用率」より前の部分）
    name_match = re.search(r"<title>([^<]+?)の使用率", html)
    if not name_match:
        # フォールバック: 最初の | より前
        name_match = re.search(r"<title>([^<|]+)", html)
    raw_name = name_match.group(1).strip() if name_match else pokemon_id

    # zukan.txt の正規名に変換（見つからない場合は取得した名前を使用）
    pokemon_name = normalize_pokemon_name(pokemon_id, raw_name)

    # JavaScriptのデータを抽出
    data = PokemonUsageData(
        pokemon_id=pokemon_id,
        pokemon_name=pokemon_name,
        season=season,
        rule=rule,
        source_url=url,
    )

    # trendItems: 持ち物の採用率（パーセント形式）
    items_match = re.search(r"trendItems:\s*(\{[^}]+\})", html)
    if items_match:
        data.items = _parse_trend_data(items_match.group(1))

    # trendAbilities: 特性の採用率
    abilities_match = re.search(r"trendAbilities:\s*(\{[^}]+\})", html)
    if abilities_match:
        data.abilities = _parse_trend_data(abilities_match.group(1))

    # trendTeraTypes: テラスタイプの採用率
    tera_match = re.search(r"trendTeraTypes:\s*(\{[^}]+\})", html)
    if tera_match:
        data.tera_types = _parse_trend_data(tera_match.group(1))

    # trendNatures: 性格の採用率（存在する場合）
    natures_match = re.search(r"trendNatures:\s*(\{[^}]+\})", html)
    if natures_match:
        data.natures = _parse_trend_data(natures_match.group(1))

    # trendMoves: 技の採用率（存在する場合）
    moves_match = re.search(r"trendMoves:\s*(\{[^}]+\})", html)
    if moves_match:
        data.moves = _parse_trend_data(moves_match.group(1))

    # 技データがなければHTMLから抽出
    if not data.moves:
        data.moves = _parse_moves_from_html(html)

    # 性格データもHTMLから抽出を試みる
    if not data.natures:
        data.natures = _parse_natures_from_html(html)

    # analysisTeamsから指定シーズンのデータを補完
    analysis_match = re.search(
        r"window\.pokedbChart\.charts\.analysisTeams\s*=\s*(\{.+?\});", html
    )
    if analysis_match:
        try:
            analysis_data = json.loads(analysis_match.group(1))
            season_key = str(season)
            if season_key in analysis_data:
                season_data = analysis_data[season_key]
                # 持ち物データがなければ補完
                if not data.items and "items" in season_data:
                    data.items = _parse_analysis_chart(season_data["items"])
                # テラスタイプデータがなければ補完
                if not data.tera_types and "teraTypes" in season_data:
                    data.tera_types = _parse_analysis_chart(season_data["teraTypes"])
        except json.JSONDecodeError:
            pass

    # 取得時刻
    from datetime import datetime

    data.fetched_at = datetime.now().isoformat()

    time.sleep(delay)  # サーバー負荷軽減
    return data


def _parse_trend_data(json_str: str) -> dict[str, float]:
    """
    trendXXX形式のデータをパース

    形式: {"data":["45.1","36.2",...], "labels":["アイテム1","アイテム2",...]}
    """
    try:
        # シングルクォートをダブルクォートに変換
        json_str = json_str.replace("'", '"')
        obj = json.loads(json_str)

        labels = obj.get("labels", [])
        values = obj.get("data", [])

        result = {}
        for label, value in zip(labels, values):
            # 値は文字列の場合があるので float に変換
            try:
                prob = float(value) / 100.0  # パーセントを0-1に変換
                result[label] = prob
            except (ValueError, TypeError):
                continue

        return result
    except json.JSONDecodeError:
        return {}


def _parse_analysis_chart(chart_data: dict) -> dict[str, float]:
    """
    analysisTeams内のチャートデータをパース

    形式: {"data":[19,15,9,...], "labels":["アイテム1","アイテム2",...]}
    """
    labels = chart_data.get("labels", [])
    values = chart_data.get("data", [])

    total = sum(values) if values else 0
    if total == 0:
        return {}

    result = {}
    for label, value in zip(labels, values):
        result[label] = value / total

    return result


def _parse_moves_from_html(html: str) -> dict[str, float]:
    """
    HTMLから技の採用率を抽出

    形式:
    <span class="pokemon-move-name">じしん</span>
    ...
    <span class="pokemon-move-rate">92.3<small>%</small></span>
    """
    result = {}

    # 技名と採用率のペアを抽出
    pattern = r'<span class="pokemon-move-name">([^<]+)</span>.*?<span class="pokemon-move-rate">(\d+\.?\d*)'
    matches = re.findall(pattern, html, re.DOTALL)

    for move_name, rate_str in matches:
        try:
            rate = float(rate_str) / 100.0
            result[move_name.strip()] = rate
        except ValueError:
            continue

    return result


def _parse_natures_from_html(html: str) -> dict[str, float]:
    """
    HTMLから性格の採用率を抽出

    性格データは chart-trend-natures セクションにある
    """
    result = {}

    # trendNatures の JavaScript オブジェクトを探す（より広いパターン）
    pattern = (
        r'trendNatures:\s*\{[^}]*"data":\s*\[([^\]]+)\][^}]*"labels":\s*\[([^\]]+)\]'
    )
    match = re.search(pattern, html)
    if match:
        try:
            values_str = match.group(1)
            labels_str = match.group(2)

            # 値をパース
            values = [float(v.strip().strip('"')) for v in values_str.split(",")]
            # ラベルをパース
            labels = [
                l.strip().strip('"').encode().decode("unicode_escape")
                for l in labels_str.split(",")
            ]

            for label, value in zip(labels, values):
                result[label] = value / 100.0
        except (ValueError, IndexError):
            pass

    return result


def _parse_chart_data(chart_section: dict) -> dict[str, float]:
    """
    チャートデータから {名前: 採用率} を抽出

    pokedb.tokyoのデータ形式:
    {
        "labels": ["技1", "技2", ...],
        "datasets": [{"data": [45.1, 36.2, ...], ...}]
    }
    """
    result = {}

    labels = chart_section.get("labels", [])
    datasets = chart_section.get("datasets", [])

    if datasets and "data" in datasets[0]:
        values = datasets[0]["data"]
        for label, value in zip(labels, values):
            if isinstance(value, (int, float)):
                result[label] = float(value) / 100.0  # パーセントを0-1に変換
            else:
                result[label] = 0.0

    return result


def get_pokemon_id(pokemon_name: str) -> Optional[str]:
    """ポケモン名から図鑑IDを取得"""
    return POKEMON_NAME_TO_ID.get(pokemon_name)


def fetch_top_pokemon_ids(
    top_n: int = 150,
    season: int = 37,
    rule: int = 0,
) -> list[str]:
    """
    使用率ランキングページから上位N位のポケモンIDを取得

    Args:
        top_n: 取得する上位ポケモン数
        season: シーズン番号
        rule: 0=シングル, 1=ダブル

    Returns:
        ポケモンIDのリスト（例: ["1003-00", "0898-02", ...]）
    """
    url = f"https://sv.pokedb.tokyo/pokemon/list?season={season}&rule={rule}"

    try:
        req = Request(url, headers={"User-Agent": "Pokemon-MCTS-Simulator/1.0"})
        with urlopen(req, timeout=30) as response:
            html = response.read().decode("utf-8")
    except (HTTPError, URLError) as e:
        print(f"Error fetching ranking page: {e}")
        return []

    # /pokemon/show/XXXX-XX 形式のリンクを抽出
    pattern = r"/pokemon/show/(\d{4}-\d{2})"
    matches = re.findall(pattern, html)

    # 重複を除去しつつ順序を保持
    seen = set()
    unique_ids = []
    for pokemon_id in matches:
        if pokemon_id not in seen:
            seen.add(pokemon_id)
            unique_ids.append(pokemon_id)
            if len(unique_ids) >= top_n:
                break

    return unique_ids


def fetch_multiple_pokemon(
    pokemon_ids: list[str],
    season: int = 37,
    rule: int = 0,
    delay: float = 1.0,
) -> list[PokemonUsageData]:
    """複数のポケモンの使用率データを取得"""
    results = []
    total = len(pokemon_ids)

    for i, pokemon_id in enumerate(pokemon_ids, 1):
        print(f"[{i}/{total}] Fetching {pokemon_id}...")
        data = fetch_pokemon_usage(pokemon_id, season, rule, delay)
        if data:
            results.append(data)
            print(
                f"  -> {data.pokemon_name}: {len(data.moves)} moves, {len(data.items)} items"
            )

    return results


def load_pokemon_names_from_trainer_json(json_path: str | Path) -> list[str]:
    """trainer.jsonからユニークなポケモン名リストを取得"""
    path = Path(json_path)
    with path.open("r", encoding="utf-8") as f:
        trainers = json.load(f)

    pokemon_names = set()
    for trainer in trainers:
        for pokemon in trainer.get("pokemons", []):
            name = pokemon.get("name")
            if name:
                pokemon_names.add(name)

    return sorted(pokemon_names)


def save_usage_data(
    data_list: list[PokemonUsageData],
    output_path: str | Path,
) -> None:
    """使用率データをJSONファイルに保存"""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # dataclassをdictに変換
    output = [asdict(d) for d in data_list]

    with path.open("w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(data_list)} pokemon data to {path}")


def normalize_existing_json(
    input_path: str | Path, output_path: str | Path | None = None
) -> None:
    """
    既存のJSONファイルのポケモン名をzukan.txtに基づいて正規化

    Args:
        input_path: 入力JSONファイルパス
        output_path: 出力JSONファイルパス（Noneの場合は入力ファイルを上書き）
    """
    input_path = Path(input_path)
    if output_path is None:
        output_path = input_path
    else:
        output_path = Path(output_path)

    with input_path.open("r", encoding="utf-8") as f:
        data_list = json.load(f)

    normalized_count = 0
    for entry in data_list:
        pokemon_id = entry.get("pokemon_id", "")
        old_name = entry.get("pokemon_name", "")
        new_name = normalize_pokemon_name(pokemon_id, old_name)

        if new_name != old_name:
            print(f"  {pokemon_id}: '{old_name}' -> '{new_name}'")
            entry["pokemon_name"] = new_name
            normalized_count += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(data_list, f, ensure_ascii=False, indent=2)

    print(f"\nNormalized {normalized_count} / {len(data_list)} pokemon names")
    print(f"Saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Fetch Pokemon usage data from pokedb.tokyo"
    )
    parser.add_argument(
        "--pokemon-id",
        type=str,
        help="Single Pokemon ID to fetch (e.g., 1003-00)",
    )
    parser.add_argument(
        "--pokemon-name",
        type=str,
        help="Single Pokemon name to fetch (e.g., ディンルー)",
    )
    parser.add_argument(
        "--from-trainer",
        type=str,
        help="Path to trainer.json to extract Pokemon names",
    )
    parser.add_argument(
        "--top",
        type=int,
        help="Fetch top N Pokemon from usage ranking (e.g., --top 150)",
    )
    parser.add_argument(
        "--season",
        type=int,
        default=37,
        help="Season number (default: 37)",
    )
    parser.add_argument(
        "--rule",
        type=int,
        default=0,
        choices=[0, 1],
        help="Rule: 0=Singles, 1=Doubles (default: 0)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/pokedb_usage/usage_data.json",
        help="Output JSON file path",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="Delay between requests in seconds (default: 1.0)",
    )
    parser.add_argument(
        "--normalize",
        type=str,
        metavar="JSON_FILE",
        help="Normalize pokemon names in existing JSON file using zukan.txt",
    )

    args = parser.parse_args()

    # 既存JSONの正規化モード
    if args.normalize:
        print(f"Normalizing pokemon names in {args.normalize}...")
        normalize_existing_json(
            args.normalize,
            args.output if args.output != "data/pokedb_usage/usage_data.json" else None,
        )
        return

    pokemon_ids = []

    # 単一ポケモンID指定
    if args.pokemon_id:
        pokemon_ids = [args.pokemon_id]

    # ポケモン名指定
    elif args.pokemon_name:
        pid = get_pokemon_id(args.pokemon_name)
        if pid:
            pokemon_ids = [pid]
        else:
            print(f"Unknown pokemon name: {args.pokemon_name}")
            print("Available names:", list(POKEMON_NAME_TO_ID.keys())[:10], "...")
            return

    # trainer.jsonから取得
    elif args.from_trainer:
        pokemon_names = load_pokemon_names_from_trainer_json(args.from_trainer)
        print(f"Found {len(pokemon_names)} unique pokemon in {args.from_trainer}")

        for name in pokemon_names:
            pid = get_pokemon_id(name)
            if pid:
                pokemon_ids.append(pid)
            else:
                print(f"  Warning: No ID mapping for '{name}'")

        print(f"Will fetch {len(pokemon_ids)} pokemon with known IDs")

    # 使用率上位Nポケモンを取得
    elif args.top:
        print(f"Fetching top {args.top} Pokemon IDs from ranking page...")
        pokemon_ids = fetch_top_pokemon_ids(
            top_n=args.top,
            season=args.season,
            rule=args.rule,
        )
        print(f"Found {len(pokemon_ids)} Pokemon IDs")

    else:
        parser.print_help()
        return

    if not pokemon_ids:
        print("No pokemon to fetch")
        return

    # データ取得
    print(
        f"\nFetching data for season {args.season}, rule {'Singles' if args.rule == 0 else 'Doubles'}"
    )
    results = fetch_multiple_pokemon(
        pokemon_ids,
        season=args.season,
        rule=args.rule,
        delay=args.delay,
    )

    # 保存
    if results:
        save_usage_data(results, args.output)

        # サマリー表示
        print("\n" + "=" * 50)
        print("Summary:")
        for data in results[:5]:
            print(f"\n{data.pokemon_name}:")
            if data.moves:
                top_moves = sorted(data.moves.items(), key=lambda x: -x[1])[:3]
                print(
                    f"  Top moves: {', '.join(f'{m}({v:.1%})' for m, v in top_moves)}"
                )
            if data.items:
                top_items = sorted(data.items.items(), key=lambda x: -x[1])[:3]
                print(
                    f"  Top items: {', '.join(f'{i}({v:.1%})' for i, v in top_items)}"
                )
        if len(results) > 5:
            print(f"\n... and {len(results) - 5} more pokemon")


if __name__ == "__main__":
    main()
