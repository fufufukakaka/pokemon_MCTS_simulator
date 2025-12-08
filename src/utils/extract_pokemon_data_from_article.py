"""ポケモン構築記事からポケモン情報を抽出するスクリプト."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any  # noqa: F401

import click
import pandas as pd
import requests
from bs4 import BeautifulSoup
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from tqdm.auto import tqdm

# --- Logging Setup ---
logger = logging.getLogger(__name__)


def setup_logging(verbose: bool = False) -> None:
    """ロギングの設定を行う."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


# --- Constants ---
DATA_DIR = Path("data")
DEFAULT_MODEL = "gpt-5.1"
SKIP_DOMAINS = ("yakkun", "x.com", "twitter.com")
MAX_POKEMON_PER_TEAM = 6


# --- Data Loading ---
@lru_cache(maxsize=1)
def load_pokemon_names_and_abilities() -> list[dict]:
    """ポケモン名と特性の対応データを読み込む."""
    logger.debug("ポケモン名と特性データを読み込み中...")
    df = pd.read_csv(DATA_DIR / "zukan.txt", sep="\t")
    data = df[["Name", "Ability1", "Ability2", "Ability3", "Ability4"]].to_dict(
        orient="records"
    )
    logger.debug(f"ポケモンデータ {len(data)} 件を読み込みました")
    return data


@lru_cache(maxsize=1)
def load_move_names() -> list[str]:
    """技名リストを読み込む."""
    logger.debug("技名データを読み込み中...")
    names = pd.read_csv(DATA_DIR / "move.txt", sep="\t")["Name"].tolist()
    logger.debug(f"技名 {len(names)} 件を読み込みました")
    return names


@lru_cache(maxsize=1)
def load_item_names() -> list[str]:
    """アイテム名リストを読み込む."""
    logger.debug("アイテム名データを読み込み中...")
    names = pd.read_csv(DATA_DIR / "item.txt", sep="\t")["Name"].tolist()
    logger.debug(f"アイテム名 {len(names)} 件を読み込みました")
    return names


@lru_cache(maxsize=1)
def load_nature_names() -> list[str]:
    """性格名リストを読み込む."""
    logger.debug("性格名データを読み込み中...")
    df = pd.read_csv(
        DATA_DIR / "nature.txt",
        sep=" ",
        header=None,
    )
    df.columns = ["Name", "1", "2", "3", "4", "5", "6"]
    names = df["Name"].tolist()
    logger.debug(f"性格名 {len(names)} 件を読み込みました")
    return names


# --- Pydantic Models ---
class PredictExtractabilityPokemonInfo(BaseModel):
    """ポケモン情報が抽出可能かどうかを表すモデル."""

    is_extractable: bool = Field(description="情報抽出可能かどうか")


class PokemonInfo(BaseModel):
    """個別ポケモンの情報."""

    name: str = Field(description="ポケモン名")
    item: str = Field(description="アイテム")
    nature: str = Field(description="性格")
    ability: str = Field(description="特性")
    Ttype: str = Field(description="テラスタルタイプ")
    moves: list[str] = Field(description="技")
    effort: list[int] = Field(description="努力値")


class ExtractedPokemons(BaseModel):
    """抽出されたポケモン情報のリスト."""

    internal_thinking_process: str = Field(description="思考過程")
    pokemons: list[PokemonInfo] = Field(description="ポケモンの情報")


# --- Prompt Templates ---
EXTRACTABILITY_CHECK_PROMPT = ChatPromptTemplate.from_template(
    "あなたはポケモンに詳しいポケモンマスターです。"
    "以下のブログ記事の中にはポケモンに関する次の情報が含まれています。"
    "ポケモン名、タイプ、特性、技4つ、アイテム、性格、努力値、テラスタルタイプ"
    "この情報が全部で6体分存在します。"
    "ポケモン名: pokemon_name, "
    "特性: ability, "
    "技: moves, "
    "アイテム: item, "
    "性格: nature, "
    "努力値: evs(H,A,B,C,D,S), "
    "テラスタルタイプ: terastal_type(基本的にひらがな、カタカナで記載してください)"
    "重要！これらの情報が含まれているかどうかを確認してください。"
    "もし情報が抽出できそうなら YES, できなそうなら NO と答えてください。"
    "対象のブログ記事: {blog_article_content}"
)

EXTRACTION_PROMPT = ChatPromptTemplate.from_template(
    """
    # あなたの役割
    あなたはポケモンに詳しいポケモンマスターです。
    以下のブログ記事の中にはポケモンに関する次の情報が含まれています。
    ポケモン名、タイプ、特性、技4つ、アイテム、性格、努力値、テラスタルタイプ
    この情報が全部で6体分存在します。それを列挙してください。
    もし情報が不足している場合は、わかる範囲で記載してください。

    # 出力すべき情報
    ポケモン名: pokemon_name,
    特性: ability,
    技: moves,
    アイテム: item,
    性格: nature,
    努力値: evs(H,A,B,C,D,S)
    テラスタルタイプ: terastal_type

    # 注意点
    技名は以下に示す一覧表記に従ってください。
    {move_names_list}

    性格名は以下に示す一覧表記に従ってください。
    {nature_names_list}

    アイテム名は以下に示す一覧表記に従ってください。
    {item_names_list}

    特性名はポケモンごとに対応するものが決まっています。ブログ記事内で特性名が曖昧な場合は、ポケモン名から特性名を特定してください。
    ポケモン名と特性名の対応は以下に従ってください。
    {pokemon_names_and_ability}

    ポケモン名について。
    - ポケモン名と特性名の対応に書いてあるポケモン名表記に揃えてください。
        - ウーラオスは「ウーラオス(れんげき)」または「ウーラオス(いちげき)」
        - バドレックスは「バドレックス(こくば)」または「バドレックス(はくば)」
        - アルセウスは常に「アルセウス」としてください。

    努力値ですが、努力値ではなく実数値で表記されている場合があります。例えば、131(4)-x-75-187(252)-155-205(252+) のような形です。
    この場合、努力値は [4, 0, 0, 252, 0, 252] としてください。

    テラスタルのタイプは漢字を使わずにひらがなまたはカタカナで記載してください。
    そのポケモンのタイプではなく、テラスタルタイプを記載してください。記事中にはテラスタイプ・またはテラスタルタイプと書かれています。
    たとえば、ディンルー@オボンの実
    特性：災いの器
    性格：腕白
    テラスタイプ：水
    261(244)-130-178(132)-×-117(132)-65
    と書かれていた場合、テラスタルタイプは「みず」となります。

    # 対象のブログ記事

    {blog_article_content}
    """
)


# --- Data Classes ---
@dataclass
class TrainerData:
    """トレーナーデータを表すクラス."""

    rank: int
    rating: float
    trainer_name: str
    article_url: str


@dataclass
class ExtractedTrainerData:
    """抽出されたトレーナーデータを表すクラス."""

    rank: int
    rating: float
    trainer_name: str
    blog_url: str
    pokemon_info: list[PokemonInfo]


# --- LLM Model Factory ---
def create_extractability_model() -> ChatOpenAI:
    """抽出可能性判定用のモデルを作成."""
    return ChatOpenAI(model=DEFAULT_MODEL).with_structured_output(
        PredictExtractabilityPokemonInfo
    )


def create_extraction_model() -> ChatOpenAI:
    """ポケモン情報抽出用のモデルを作成."""
    return ChatOpenAI(model=DEFAULT_MODEL).with_structured_output(ExtractedPokemons)


# --- Article Link Extraction ---
def build_search_url(season: int, page: int) -> str:
    """記事検索URLを構築する."""
    base_url = "https://sv.pokedb.tokyo/article/search"
    params = {
        "rule": 0,
        "season_start": season,
        "season_end": season,
        "sort": "default",
        "per_page": 30,
        "page": page,
    }
    # 各ポケモンスロットのパラメータを追加
    for i in range(1, 7):
        params.update(
            {
                f"pokemon_detail_type_{i}": "",
                f"pokemon_detail_category_{i}": "all",
                f"pokemon_detail_stat_name_{i}": "",
                f"pokemon_detail_stat_value_{i}": "",
                f"pokemon_{i}": "",
                f"item_{i}": "",
                f"terastal_{i}": "",
            }
        )
    params.update(
        {
            "regulation": "",
            "trainer_name": "",
            "trainer_mode": "and",
            "article_title": "",
            "title_mode": "and",
            "min_rank": "",
            "min_rate": "",
        }
    )
    query_string = "&".join(f"{k}={v}" for k, v in params.items())
    return f"{base_url}?{query_string}#search-results"


def parse_trainer_card(trainer_element: BeautifulSoup) -> TrainerData | None:
    """トレーナーカード要素からデータを抽出する."""
    header = trainer_element.find("div", class_="card-header-title is-flex-wrap-wrap")
    if header is None:
        return None

    spans = header.find_all("span")
    if len(spans) < 5:
        return None

    footer = trainer_element.find("footer")
    if footer is None:
        return None

    link_element = footer.find("a")
    if link_element is None or link_element.get("href") is None:
        return None

    rank_text = spans[2].text.strip().replace("位", "")
    rating_text = spans[4].text.strip()
    trainer_name = header.find("p").text.strip() if header.find("p") else ""

    return TrainerData(
        rank=int(rank_text),
        rating=float(rating_text),
        trainer_name=trainer_name,
        article_url=link_element["href"],
    )


def extract_article_links(season: int, max_pages: int = 4) -> list[TrainerData]:
    """指定シーズンの構築記事リンクを抽出する."""
    logger.info(f"シーズン {season} の構築記事を検索中... (最大 {max_pages} ページ)")
    trainers_data: list[TrainerData] = []

    for page in range(1, max_pages + 1):
        logger.debug(f"ページ {page}/{max_pages} を取得中...")
        url = build_search_url(season, page)
        response = requests.get(url, timeout=30)
        response.encoding = response.apparent_encoding

        soup = BeautifulSoup(response.text, "html.parser")
        trainer_elements = soup.select("div.column.is-half-desktop")
        page_count = 0

        for element in trainer_elements:
            trainer = parse_trainer_card(element)
            if trainer is not None:
                trainers_data.append(trainer)
                page_count += 1

        logger.debug(f"ページ {page}: {page_count} 件のトレーナーを取得")

    logger.info(f"合計 {len(trainers_data)} 件のトレーナー記事を発見")
    return trainers_data


# --- Pokemon Data Extraction ---
def should_skip_url(url: str) -> tuple[bool, str]:
    """URLがスキップ対象かどうかを判定する."""
    for domain in SKIP_DOMAINS:
        if domain in url:
            if domain == "yakkun":
                return True, "ポケ徹記事のためスキップします。"
            return True, "Twitter記事のためスキップします。"
    return False, ""


def load_article_content(url: str) -> Any | None:
    """記事コンテンツを読み込む."""
    logger.debug(f"記事を読み込み中: {url}")
    loader = WebBaseLoader(web_paths=[url])
    docs = list(loader.lazy_load())
    if docs:
        logger.debug(f"記事の読み込み完了 (文字数: {len(docs[0].page_content)})")
    return docs[0] if docs else None


def check_extractability(content: str, model: ChatOpenAI) -> bool:
    """記事からポケモン情報が抽出可能か判定する."""
    logger.debug("抽出可能性をLLMで判定中...")
    chain = EXTRACTABILITY_CHECK_PROMPT | model
    result: PredictExtractabilityPokemonInfo = chain.invoke(
        {"blog_article_content": content}
    )
    logger.debug(f"抽出可能性判定結果: {result.is_extractable}")
    return result.is_extractable


def extract_pokemon_info(content: str, model: ChatOpenAI) -> list[PokemonInfo]:
    """記事からポケモン情報を抽出する."""
    logger.debug("ポケモン情報をLLMで抽出中...")
    chain = EXTRACTION_PROMPT | model
    result: ExtractedPokemons = chain.invoke(
        {
            "pokemon_names_and_ability": load_pokemon_names_and_abilities(),
            "item_names_list": load_item_names(),
            "nature_names_list": load_nature_names(),
            "move_names_list": load_move_names(),
            "blog_article_content": content,
        }
    )
    logger.debug(f"抽出完了: {len(result.pokemons)} 体のポケモン")
    return result.pokemons


def extract_pokemon_data_from_article(blog_url: str) -> list[PokemonInfo] | None:
    """ブログ記事からポケモンデータを抽出する."""
    logger.info(f"記事を処理中: {blog_url}")

    should_skip, message = should_skip_url(blog_url)
    if should_skip:
        logger.warning(message)
        return None

    doc = load_article_content(blog_url)
    if doc is None:
        logger.error(f"記事の読み込みに失敗しました: {blog_url}")
        return None

    extractability_model = create_extractability_model()
    try:
        if not check_extractability(doc.page_content, extractability_model):
            logger.warning("情報を抽出できませんでした。")
            return None
    except Exception as e:
        logger.error(f"抽出可能性チェック中にエラー: {e}")
        return None

    extraction_model = create_extraction_model()
    try:
        pokemons = extract_pokemon_info(doc.page_content, extraction_model)
        logger.info(f"抽出成功: {len(pokemons)} 体のポケモン")
        for i, pokemon in enumerate(pokemons, 1):
            logger.debug(f"  {i}. {pokemon.name} (@{pokemon.item}) - {pokemon.ability}")
        return pokemons
    except Exception as e:
        logger.error(f"ポケモン情報抽出中にエラー: {e}")
        return None


# --- Output Formatting ---
def flatten_pokemon_data(pokemon: PokemonInfo, index: int) -> dict:
    """ポケモン情報をフラット化する."""
    prefix = f"pokemon{index}"
    return {
        f"{prefix}_name": pokemon.name,
        f"{prefix}_item": pokemon.item,
        f"{prefix}_nature": pokemon.nature,
        f"{prefix}_ability": pokemon.ability,
        f"{prefix}_Ttype": pokemon.Ttype,
        f"{prefix}_moves": ", ".join(pokemon.moves),
        f"{prefix}_effort": (
            ", ".join(map(str, pokemon.effort)) if pokemon.effort else None
        ),
    }


def trainer_data_to_flat_dict(trainer: ExtractedTrainerData) -> dict:
    """トレーナーデータをフラットな辞書に変換する."""
    flat_data = {
        "rank": trainer.rank,
        "rating": trainer.rating,
        "trainer_name": trainer.trainer_name,
        "blog_url": trainer.blog_url,
    }

    for idx, pokemon in enumerate(trainer.pokemon_info[:MAX_POKEMON_PER_TEAM], start=1):
        flat_data.update(flatten_pokemon_data(pokemon, idx))

    return flat_data


# --- Main Entry Point ---
@click.command()
@click.option("--season", type=int, default=36, help="シーズン番号")
@click.option("--max-pages", type=int, default=4, help="検索ページ数")
@click.option("--verbose", "-v", is_flag=True, help="詳細ログを出力")
def main(season: int, max_pages: int, verbose: bool) -> None:
    """構築記事からポケモンデータを抽出してCSVに保存する."""
    setup_logging(verbose=verbose)

    logger.info("=" * 60)
    logger.info(f"ポケモン構築記事抽出ツール - シーズン {season}")
    logger.info("=" * 60)
    logger.info(f"設定: max_pages={max_pages}, verbose={verbose}")

    trainers = extract_article_links(season=season, max_pages=max_pages)
    logger.info(f"処理対象: {len(trainers)} 件のトレーナー")

    extracted_trainers: list[ExtractedTrainerData] = []
    success_count = 0
    skip_count = 0

    for i, trainer in enumerate(tqdm(trainers, desc="Extracting pokemon data"), 1):
        logger.info(
            f"[{i}/{len(trainers)}] {trainer.trainer_name} (順位: {trainer.rank})"
        )
        pokemon_info = extract_pokemon_data_from_article(trainer.article_url)
        if pokemon_info is not None:
            extracted_trainers.append(
                ExtractedTrainerData(
                    rank=trainer.rank,
                    rating=trainer.rating,
                    trainer_name=trainer.trainer_name,
                    blog_url=trainer.article_url,
                    pokemon_info=pokemon_info,
                )
            )
            success_count += 1
        else:
            skip_count += 1

    logger.info("-" * 60)
    logger.info("処理結果サマリ:")
    logger.info(f"  成功: {success_count} 件")
    logger.info(f"  スキップ/エラー: {skip_count} 件")
    logger.info(f"  成功率: {success_count / len(trainers) * 100:.1f}%")

    flat_data_list = [trainer_data_to_flat_dict(t) for t in extracted_trainers]

    df = pd.DataFrame(flat_data_list)
    output_path = f"data/season_{season}_pokemon_data.csv"
    df.to_csv(output_path, index=False)
    logger.info(f"出力ファイル: {output_path}")
    logger.info("=" * 60)
    logger.info("処理完了")


if __name__ == "__main__":
    main()
