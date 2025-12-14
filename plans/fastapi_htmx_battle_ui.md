# FastAPI + HTMX バトルUI実装計画

## 概要

既存のNext.jsフロントエンドを置き換える形で、FastAPI + HTMX + Jinja2でバトルUIを実装する。
Pythonのみで完結するシンプルな構成で、既存のBattleクラスやMCTSを直接呼び出せる。

## 現状分析

### 既存Next.jsフロントエンド構成
- `frontend/src/types/battle.ts`: 型定義（BattleState, Action等）
- `frontend/src/app/page.tsx`: メインページ（フェーズ管理）
- `frontend/src/components/battle-field.tsx`: バトル画面
- `frontend/src/components/action-panel.tsx`: 行動選択
- `frontend/src/components/pokemon-card.tsx`: ポケモン表示
- `frontend/src/components/battle-log.tsx`: ログ表示
- `frontend/src/lib/api.ts`: APIクライアント

### 実装すべき機能
1. チームセットアップ画面
2. 選出フェーズ（3匹選択）
3. バトル画面
   - ポケモン表示（HP、状態異常、能力変化）
   - 技選択 / 交代選択
   - テラスタル
   - フィールド状態（天気、地形、壁等）
4. バトルログ表示
5. 勝敗判定

## アーキテクチャ

```
src/battle_ui/
├── main.py                 # FastAPIアプリ（エントリーポイント）
├── routers/
│   ├── __init__.py
│   ├── pages.py            # HTMLページ用ルーター
│   └── api.py              # HTMX用APIルーター（HTMLフラグメント返却）
├── services/
│   ├── __init__.py
│   ├── battle_service.py   # バトルロジック（既存Battleクラスをラップ）
│   └── session_manager.py  # セッション管理（バトル状態保持）
├── templates/
│   ├── base.html           # ベーステンプレート（HTMX読み込み）
│   ├── pages/
│   │   ├── index.html      # トップページ
│   │   ├── setup.html      # チームセットアップ
│   │   ├── selection.html  # 選出画面
│   │   └── battle.html     # バトル画面
│   └── components/
│       ├── pokemon_card.html     # ポケモンカード
│       ├── action_panel.html     # 行動選択パネル
│       ├── battle_log.html       # バトルログ
│       ├── field_status.html     # フィールド状態
│       └── result_modal.html     # 結果表示
└── static/
    ├── css/
    │   └── styles.css      # Tailwind CSS（CDN）+ カスタムスタイル
    └── images/             # ポケモン画像等（オプション）
```

## 実装フェーズ

### Phase 1: 基盤構築
1. FastAPIアプリ骨格作成（main.py）
2. Jinja2テンプレート設定
3. 静的ファイル配信設定
4. ベーステンプレート作成（HTMX + Tailwind CDN）

### Phase 2: セッション・サービス層
1. SessionManager実装（辞書ベースのインメモリ管理）
2. BattleService実装（既存Battleクラスのラップ）
3. 状態変換関数（Battle → テンプレート用dict）

### Phase 3: ページルーター
1. トップページ（/）
2. セットアップページ（/setup）
3. 選出ページ（/battle/{session_id}/selection）
4. バトルページ（/battle/{session_id}）

### Phase 4: HTMXコンポーネント
1. ポケモンカード（HPバー、状態表示）
2. 行動選択パネル（技・交代・テラスタル）
3. バトルログ（自動スクロール）
4. フィールド状態表示

### Phase 5: HTMX API実装
1. POST /api/battle/create - バトル作成
2. POST /api/battle/{id}/select - 選出確定
3. POST /api/battle/{id}/action - 行動実行
4. GET /api/battle/{id}/state - 状態取得（ポーリング用）
5. POST /api/battle/{id}/surrender - 降参

### Phase 6: 仕上げ
1. エラーハンドリング
2. ローディング状態表示
3. アニメーション（CSS）
4. レスポンシブ対応

## 技術詳細

### HTMX属性の使用パターン

```html
<!-- 技選択ボタン -->
<button
  hx-post="/api/battle/{{session_id}}/action"
  hx-vals='{"action_type": "move", "index": 0}'
  hx-target="#battle-field"
  hx-swap="innerHTML"
  hx-indicator="#loading"
>
  10まんボルト
</button>

<!-- バトルログ（ポーリング更新） -->
<div
  id="battle-log"
  hx-get="/api/battle/{{session_id}}/log"
  hx-trigger="every 2s"
  hx-swap="innerHTML"
>
  ...
</div>

<!-- 交代ポケモン選択 -->
<button
  hx-post="/api/battle/{{session_id}}/action"
  hx-vals='{"action_type": "switch", "index": 1}'
  hx-target="#battle-field"
  hx-confirm="カイリューに交代しますか？"
>
  カイリュー
</button>
```

### セッション管理

```python
from dataclasses import dataclass
from typing import Dict, Optional
import uuid

@dataclass
class BattleSession:
    id: str
    battle: Battle
    player_team: list
    opponent_team: list
    phase: str  # "selection", "battle", "finished"

class SessionManager:
    def __init__(self):
        self._sessions: Dict[str, BattleSession] = {}

    def create_session(self, player_team, opponent_team) -> str:
        session_id = str(uuid.uuid4())
        # Battle初期化...
        return session_id

    def get_session(self, session_id: str) -> Optional[BattleSession]:
        return self._sessions.get(session_id)
```

### 状態変換

```python
def battle_to_view_state(battle: Battle, player: int = 0) -> dict:
    """BattleオブジェクトをHTMLレンダリング用のdictに変換"""
    return {
        "turn": battle.turn,
        "phase": "battle" if battle.winner() is None else "finished",
        "player_active": pokemon_to_dict(battle.pokemon[player][0]),
        "player_bench": [pokemon_to_dict(p) for p in battle.pokemon[player][1:3]],
        "opponent_active": pokemon_to_dict(battle.pokemon[1-player][0]),
        "opponent_bench": [pokemon_to_dict(p) for p in battle.pokemon[1-player][1:3]],
        "available_actions": get_available_actions(battle, player),
        "field": get_field_state(battle),
        "log": battle.log_lines[-10:],  # 直近10件
        "winner": battle.winner(),
    }
```

## 依存関係追加

```toml
# pyproject.toml に追加
jinja2 = "^3.1.0"
python-multipart = "^0.0.9"  # フォーム処理用
```

## 実行方法

```bash
# 開発サーバー起動
uv run uvicorn src.battle_ui.main:app --reload --port 8080

# ブラウザでアクセス
open http://localhost:8080
```

## 実装済み機能

- [x] トップページ（/）
- [x] セットアップページ（/setup）- トレーナー選択
- [x] 選出フェーズ - 3匹選択
- [x] バトル画面 - 技選択、交代、テラスタル
- [x] バトルログ表示
- [x] フィールド効果表示（天気、壁、設置技）
- [x] 勝敗結果画面
- [x] 強制交代フェーズ
- [x] エラーハンドリング

## 現在のAI

シンプルなランダム選択AIを使用しています。より強いAIに変更する場合は
`battle_service.py` の `perform_action` メソッドを修正してください。

## 参考URL
- HTMX公式: https://htmx.org/docs/
- FastAPI Templates: https://fastapi.tiangolo.com/advanced/templates/
- Tailwind CDN: https://tailwindcss.com/docs/installation/play-cdn
