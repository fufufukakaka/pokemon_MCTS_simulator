## プロジェクト概要（pokemon_MCTS_simulator）

このリポジトリは、ポケモン対戦のための **シミュレーション基盤** と **モンテカルロ木探索（MCTS）による対戦 AI**、および **ポケモン SV 対応のダメージ計算 API** をまとめたプロジェクトです。
実際のランクマッチデータや PostgreSQL データベース、Streamlit によるリーダーボード UI などを組み合わせて、対戦環境を再現・分析することを目的としています。

---

## 主なコンポーネント

- **バトルシミュレータ (`src/pokemon_battle_sim/`)**

  - `battle.py` : 場の状態・天候・フィールド効果・行動順などを含む対戦ロジックの中核クラス `Battle`
  - `pokemon.py` : 図鑑データ・種族値・技・特性・アイテムなどを扱う `Pokemon` クラス
  - `damage.py` / `utils.py` : ダメージ計算や各種補正、ユーティリティ関数群
  - `data/` 配下のテキスト・画像ファイルと連携して、現行世代の対戦ルールを詳細に再現します。

- **MCTS による対戦 AI (`src/mcts/mcts_battle.py`)**

  - `MCTSNode` / `MCTSNodeForChangeCommand` : UCT を用いた木探索ノード実装
  - `mcts` / `mcts_for_change_command` : ランダムプレイアウトによる評価とバックアップ
  - `MyMCTSBattle` : `Battle` を継承し、`battle_command` / `change_command` で MCTS を用いて行動選択を行う拡張クラス
  - 盤面評価関数 `score` で自他の内部評価値比を用いたスコアリングを行います。

- **シミュレーションとマッチング (`scripts/matching.py`, `src/models.py`)**

  - `Trainer` クラス: ランク・レーティング・ポケモン構成を持つトレーナーモデル
  - `SimulatedBattle` : 2 人のトレーナー間で `MyMCTSBattle` を用いた対戦をシミュレートし、ログを保存
  - `match_trainers` / `update_elo` / `pokemon_battle` :
    レーティング差に基づく対戦相手マッチングと Elo レーティング更新ロジックを提供
  - `scripts/matching.py` の CLI から、データベースと連携した対戦シミュレーションをバッチ実行します。

- **データベース連携とレーティング管理 (`src/database_handler.py`)**

  - PostgreSQL（環境変数 `POSTGRES_*`）を利用し、以下のテーブルを管理
    - `battle_history` : 対戦結果・参加トレーナー・レーティング・ログ保存時刻
    - `trainer_rating` : トレーナーごとのシミュレーションレーティング
  - 対戦履歴の保存、レーティングの初期化・更新、リーダーボード表示用データの生成を行います。

- **リーダーボード UI (`src/streamlit_leaderboard.py`)**

  - Streamlit を用いて、`trainer_rating` テーブルをもとにトレーナーのランキングを表示
  - 上位トレーナーのレーティング分布を Altair で可視化し、直近の対戦履歴もテーブル表示します。

- **ダメージ計算 API (`src/damage_calculator_api/`)**

  - FastAPI ベースの REST API サーバー
  - 主な要素
    - `main.py` : アプリケーションエントリポイント（CORS・例外ハンドラ・ルーティング設定）
    - `routers/damage.py` :
      - `/calculate` : 16 段階乱数・各種補正込みのダメージ計算
      - `/compare` : 複数技の比較とおすすめ技の提示
      - `/analyze` : ダメージ分布と確定数・KO 確率の詳細分析
    - `routers/pokemon.py` : ポケモン・技・道具・タイプ相性の参照 API 群
    - `calculators/` / `schemas/` / `models/` / `utils/` :  
      ドメインモデル、Pydantic スキーマ、実数値計算ロジック、データローダなど
  - `lifespan` フックで起動時にポケモン/技/道具データを事前ロードし、高速なレスポンスを実現します。

- **ユーティリティとその他**
  - `src/utils/` : シミュレーション結果の抽出 (`extract_battle_history.py`) や fastText 学習 (`train_fasttext.py`) など
  - `notebook/` : データ変換・シミュレーション検証・レーティング見直しなどの検証 Notebook 群
  - `data/` : ランクマッチデータ、図鑑・技・特性・アイテム情報、画面用テンプレート画像など

---

## 実行環境

- **Python**: `>=3.12,<4.0`
- **主要ライブラリ（抜粋）**
  - Web/API: FastAPI, Uvicorn
  - データ処理: pandas, numpy, matplotlib, scikit-learn, lightgbm
  - DB: SQLAlchemy, psycopg2-binary
  - アプリケーション / UI: click, streamlit, altair
  - 自然言語系: fasttext, gensim
  - テスト: pytest
  - LLM 連携: langchain, openai, langchain-openai, langchain-community

依存関係は `pyproject.toml` に Poetry 形式で定義されています。

---

## 代表的な使い方

### 1. 対戦シミュレーションの実行

PostgreSQL と Discord Webhook の環境変数を設定した上で、対戦シミュレーションを実行します。

```bash
DISCORD_WEBHOOK_URL={} \
POSTGRES_DB={} \
POSTGRES_PASSWORD={} \
POSTGRES_USER={} \
POSTGRES_HOST={} \
POSTGRES_PORT={} \
uv run python scripts/matching.py
```

- `--resume` オプションを付与すると、途中状態からの再開が可能です（README 参照）。
- 対戦履歴・レーティングは PostgreSQL に保存されます。

### 2. バトル履歴の抽出

シミュレーション結果からバトル履歴を抽出するユーティリティ。

```bash
uv run python src/utils/extract_battle_history.py
```

### 3. fastText による埋め込み学習

ポケモン・技などの埋め込み表現を学習するスクリプト。

```bash
uv run python src/utils/train_fasttext.py
```

### 4. リーダーボード UI（Streamlit）

PostgreSQL に保存された `trainer_rating` テーブルをもとに、ブラウザからランキングを閲覧できます。

```bash
uv run streamlit run src/streamlit_leaderboard.py
```

### 5. ダメージ計算 API サーバー

ポケモン SV 対応のダメージ計算 API を立ち上げます。

```bash
uv run uvicorn src.damage_calculator_api.main:app --reload
```

- 起動後、`/docs` から自動生成された Swagger UI でエンドポイントを確認できます。
- 代表的なエンドポイント
  - `POST /api/v1/damage/calculate`
  - `POST /api/v1/damage/compare`
  - `POST /api/v1/damage/analyze`
  - `GET /api/v1/pokemon/list`, `/api/v1/pokemon/info/{name}` など

---

## データと前提条件

- `data/top_rankers/season_27.json` : シミュレーション対象となるトレーナーとパーティ構成
- `data/battle_database/*.csv` : ランクマッチの使用率・構築データ
- `data/*.txt` : 図鑑・技・特性・アイテム・性格・タイプ相性などの定義
- これらのデータを前提として `Pokemon.init()` などで内部データ構造が初期化されます。

PostgreSQL のスキーマは、`DatabaseHandler` および `streamlit_leaderboard.py` 内の `BattleHistory` / `TrainerRating` モデル定義に依存します。

---

## 今後の拡張アイデア（例）

- MCTS パラメータ（探索回数・評価関数）のチューニングと A/B テスト
  \*- Damage Calculator API とシミュレーションエンジンの統合（同一データソース・ロジックの再利用）
- 追加世代やフォーマット（ダブルバトル等）への対応
- リーダーボード UI へのフィルタ・検索・リプレイ閲覧機能の追加

このドキュメントはプロジェクト全体の俯瞰的なサマリーです。  
より詳細なダメージ計算 API の設計や仕様については、`docs/damage_calculator_api_plan.md` などの個別ドキュメントを参照してください。
