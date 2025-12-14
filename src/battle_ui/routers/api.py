"""
APIルーター - HTMX用のHTMLフラグメントを返すエンドポイント
"""

from fastapi import APIRouter, Request, Form, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pathlib import Path
from typing import Optional
import json

from src.battle_ui.services.session_manager import session_manager
from src.battle_ui.services.battle_service import BattleService

router = APIRouter()

TEMPLATES_DIR = Path(__file__).resolve().parent.parent / "templates"
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


@router.get("/trainers", response_class=HTMLResponse)
async def get_trainers(request: Request):
    """トレーナー一覧を取得"""
    trainers = BattleService.get_trainers()
    return templates.TemplateResponse(
        "components/trainer_select.html",
        {"request": request, "trainers": trainers}
    )


@router.get("/player-party", response_class=HTMLResponse)
async def get_player_party(request: Request):
    """プレイヤーのパーティを取得"""
    party = BattleService.get_player_party()
    return templates.TemplateResponse(
        "components/party_display.html",
        {"request": request, "party": party}
    )


@router.get("/checkpoints", response_class=HTMLResponse)
async def get_checkpoints(request: Request):
    """チェックポイント一覧を取得"""
    checkpoints = BattleService.get_checkpoints()
    return templates.TemplateResponse(
        "components/checkpoint_select.html",
        {"request": request, "checkpoints": checkpoints}
    )


@router.post("/battle/create", response_class=HTMLResponse)
async def create_battle(
    request: Request,
    trainer_index: int = Form(...),
    ai_mode: str = Form("random"),
    checkpoint_path: Optional[str] = Form(None),
    ai_analysis_always_on: Optional[str] = Form(None),
):
    """バトルセッションを作成"""
    try:
        # プレイヤーパーティとAIトレーナーのパーティを取得
        player_party = BattleService.get_player_party()
        ai_party = BattleService.get_trainer_party(trainer_index)

        # チェックボックスの値を処理（"true" or None）
        analysis_always_on = ai_analysis_always_on == "true"

        # セッション作成
        session = session_manager.create_session(
            player_team=player_party,
            opponent_team=ai_party,
            opponent_trainer_index=trainer_index,
            ai_mode=ai_mode,
            checkpoint_path=checkpoint_path,
            ai_analysis_always_on=analysis_always_on,
        )

        # 選出画面へリダイレクト用のレスポンス
        return HTMLResponse(
            content="",
            headers={"HX-Redirect": f"/battle/{session.id}"}
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        return templates.TemplateResponse(
            "components/error.html",
            {"request": request, "error": str(e)},
            status_code=500
        )


@router.get("/battle/{session_id}/state", response_class=HTMLResponse)
async def get_battle_state(request: Request, session_id: str):
    """バトル状態を取得"""
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    state = BattleService.get_battle_state(session)

    if session.phase == "selection":
        return templates.TemplateResponse(
            "components/selection_phase.html",
            {"request": request, "session_id": session_id, "state": state}
        )
    elif session.phase == "battle" or session.phase == "change":
        return templates.TemplateResponse(
            "components/battle_field.html",
            {"request": request, "session_id": session_id, "state": state}
        )
    elif session.phase == "finished":
        return templates.TemplateResponse(
            "components/battle_result.html",
            {"request": request, "session_id": session_id, "state": state}
        )


@router.post("/battle/{session_id}/select", response_class=HTMLResponse)
async def select_pokemon(
    request: Request,
    session_id: str,
    pokemon_indices: str = Form(...),  # JSON形式の配列 "[0, 1, 2]"
):
    """選出を確定"""
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    try:
        print(f"[DEBUG] pokemon_indices raw: {pokemon_indices!r}")
        indices = json.loads(pokemon_indices)
        print(f"[DEBUG] indices parsed: {indices}")
        BattleService.select_pokemon(session, indices)

        state = BattleService.get_battle_state(session)
        return templates.TemplateResponse(
            "components/battle_field.html",
            {"request": request, "session_id": session_id, "state": state}
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        return templates.TemplateResponse(
            "components/error.html",
            {"request": request, "error": str(e)},
            status_code=400
        )


@router.post("/battle/{session_id}/action", response_class=HTMLResponse)
async def perform_action(
    request: Request,
    session_id: str,
    action_type: str = Form(...),
    action_index: int = Form(...),
    with_tera: bool = Form(False),
):
    """行動を実行"""
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    try:
        BattleService.perform_action(
            session,
            action_type=action_type,
            action_index=action_index,
            with_tera=with_tera,
        )

        state = BattleService.get_battle_state(session)

        if session.phase == "finished":
            return templates.TemplateResponse(
                "components/battle_result.html",
                {"request": request, "session_id": session_id, "state": state}
            )
        elif session.phase == "change":
            return templates.TemplateResponse(
                "components/change_pokemon.html",
                {"request": request, "session_id": session_id, "state": state}
            )
        else:
            return templates.TemplateResponse(
                "components/battle_field.html",
                {"request": request, "session_id": session_id, "state": state}
            )
    except Exception as e:
        import traceback
        traceback.print_exc()
        return templates.TemplateResponse(
            "components/error.html",
            {"request": request, "error": str(e)},
            status_code=400
        )


@router.post("/battle/{session_id}/surrender", response_class=HTMLResponse)
async def surrender(request: Request, session_id: str):
    """降参"""
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    BattleService.surrender(session)
    state = BattleService.get_battle_state(session)

    return templates.TemplateResponse(
        "components/battle_result.html",
        {"request": request, "session_id": session_id, "state": state}
    )


@router.get("/battle/{session_id}/ai-analysis", response_class=HTMLResponse)
async def get_ai_analysis(request: Request, session_id: str):
    """AI分析データを取得"""
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    state = BattleService.get_battle_state(session, include_ai_analysis=True)

    return templates.TemplateResponse(
        "components/ai_analysis.html",
        {"request": request, "session_id": session_id, "state": state}
    )
