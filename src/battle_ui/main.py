"""
Pokemon Battle UI - FastAPI + HTMX

シンプルなバトルUIをFastAPI + HTMXで実装
"""

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from src.battle_ui.routers import api, pages
from src.pokemon_battle_sim.pokemon import Pokemon

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# パス設定
BASE_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"

# 環境変数から設定を取得
USAGE_DATA_PATH = os.environ.get("POKEMON_USAGE_DATA_PATH", None)
SEASON = os.environ.get("POKEMON_SEASON", None)
PLAYER_PARTY_PATH = os.environ.get("PLAYER_PARTY_PATH", None)
DEBUG_OPPONENT = os.environ.get("DEBUG_OPPONENT", None)  # "gliscor" 等で固定対戦相手を有効化


@asynccontextmanager
async def lifespan(app: FastAPI):
    """アプリケーションライフサイクル管理"""
    logger.info("Starting Pokemon Battle UI...")

    # Pokemon データ初期化
    try:
        season = int(SEASON) if SEASON else None
        Pokemon.init(season=season, usage_data_path=USAGE_DATA_PATH)
        if USAGE_DATA_PATH:
            logger.info(f"Pokemon data initialized with usage_data_path: {USAGE_DATA_PATH}")
        elif season:
            logger.info(f"Pokemon data initialized with season: {season}")
        else:
            logger.info("Pokemon data initialized with default settings")
    except Exception as e:
        logger.error(f"Failed to initialize Pokemon data: {e}")
        raise

    yield

    logger.info("Shutting down Pokemon Battle UI...")


# FastAPIアプリケーション作成
app = FastAPI(
    title="Pokemon Battle UI",
    description="FastAPI + HTMX によるポケモンバトルUI",
    version="1.0.0",
    lifespan=lifespan,
)

# テンプレート設定
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# 静的ファイル配信
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# ルーター登録
app.include_router(pages.router)
app.include_router(api.router, prefix="/api")


# ヘルスチェック
@app.get("/health")
async def health_check():
    """ヘルスチェックエンドポイント"""
    return {"status": "healthy", "service": "Pokemon Battle UI"}


# 開発用サーバー起動
if __name__ == "__main__":
    uvicorn.run(
        "src.battle_ui.main:app",
        host="0.0.0.0",
        port=8080,
        reload=True,
        log_level="info",
    )
