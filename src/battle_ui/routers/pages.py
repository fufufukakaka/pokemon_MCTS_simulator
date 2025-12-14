"""
ページルーター - HTMLページを返すエンドポイント
"""

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pathlib import Path

router = APIRouter()

TEMPLATES_DIR = Path(__file__).resolve().parent.parent / "templates"
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


@router.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """トップページ"""
    return templates.TemplateResponse(
        "pages/index.html",
        {"request": request}
    )


@router.get("/setup", response_class=HTMLResponse)
async def setup(request: Request):
    """チームセットアップページ"""
    return templates.TemplateResponse(
        "pages/setup.html",
        {"request": request}
    )


@router.get("/battle/{session_id}", response_class=HTMLResponse)
async def battle(request: Request, session_id: str):
    """バトルページ"""
    return templates.TemplateResponse(
        "pages/battle.html",
        {"request": request, "session_id": session_id}
    )
