# pokemon_MCTS_simulator

ポケモン対戦シミュレータ + MCTS + 強化学習

## 機能

- **ポケモン対戦シミュレーション**: Gen9 対応の対戦シミュレータ
- **MCTS (Monte Carlo Tree Search)**: 仮説ベースの MCTS で不完全情報ゲームに対応
- **Policy-Value Network**: AlphaZero スタイルの行動予測・勝率予測ネットワーク
- **Team Selection Network**: 相手チームを見て最適な 3 匹を選出するネットワーク
- **強化学習ループ**: Self-Play → 学習 → 評価のサイクルで継続的に強化
- **固定パーティモード**: 自分のパーティを固定して最適な戦い方を学習

---

## セットアップ

```bash
# 依存関係インストール
poetry install

# Pokemonデータ初期化（必須）
# コード内で Pokemon.init() を呼ぶ必要があります
```

---

## 強化学習パイプライン

### 1. Self-Play データ生成

```bash
# 純粋MCTSでSelf-Playデータを生成
uv run python scripts/generate_selfplay_dataset.py \
    --trainer-json data/top_rankers/season_27.json \
    --output data/selfplay_records.jsonl \
    --num-games 100 \
    --mcts-iterations 100
```

### 2. Policy-Value Network 学習

```bash
# Self-PlayデータでPolicy-Value Networkを学習
uv run python scripts/train_policy_value_network.py \
    --dataset data/selfplay_records.jsonl \
    --output models/policy_value \
    --hidden-dim 256 \
    --num-epochs 100
```

### 3. 強化学習ループ（Self-Play → 学習 → 評価 → 繰り返し）

```bash
# AlphaZeroスタイルの強化学習ループ（標準モード）
uv run python scripts/run_reinforcement_loop.py \
    --trainer-json data/top_rankers/season_27.json \
    --output models/reinforcement \
    --num-generations 10 \
    --games-per-generation 100 \
    --evaluation-games 50

# 軽量テスト
uv run python scripts/run_reinforcement_loop.py \
    --trainer-json data/top_rankers/season_27.json \
    --output models/reinforcement_test \
    --num-generations 3 \
    --games-per-generation 20 \
    --evaluation-games 10 \
    --training-epochs 10
```

### 4. 固定パーティモード

自分の考えたパーティを固定し、そのパーティで最も強く戦えるモデルを学習する。

```bash
# 固定パーティで強化学習
# - Player 0: 固定パーティ（自分）
# - Player 1: trainer-jsonからランダムに選出（対戦相手）
uv run python scripts/run_reinforcement_loop.py \
    --trainer-json data/top_rankers/season_36.json \
    --fixed-party data/my_fixed_party.json \
    --output models/my_party_rl \
    --num-generations 50 \
    --games-per-generation 100 \
    --evaluation-games 50

# Team Selection Networkも同時に学習
uv run python scripts/run_reinforcement_loop.py \
    --trainer-json data/top_rankers/season_36.json \
    --fixed-party data/my_fixed_party.json \
    --output models/my_party_rl_with_selection \
    --num-generations 50 \
    --games-per-generation 100 \
    --train-team-selection \
    --team-selection-epochs 20 \
    --team-selection-update-interval 1
```

#### 固定パーティ JSON の形式

```json
{
  "pokemons": [
    {
      "name": "ポケモン名",
      "item": "持ち物",
      "nature": "性格",
      "evs": "H,A,B,C,D,S",
      "moves": ["技1", "技2", "技3", "技4"],
      "tera_type": "テラスタイプ"
    }
  ]
}
```

サンプル: `data/sample_fixed_party.json`, `data/my_fixed_party.json`

---

## Team Selection Network

相手の 6 匹を見て、自分の 6 匹から最適な 3 匹を選出するネットワーク。

### 単独学習

```bash
# ランダムデータで初期学習
uv run python scripts/train_team_selection.py \
    --trainer-json data/top_rankers/season_27.json \
    --output models/team_selection \
    --num-samples 10000 \
    --num-epochs 50

# Self-Playデータで学習（より高品質）
uv run python scripts/train_team_selection.py \
    --trainer-json data/top_rankers/season_27.json \
    --selfplay-data data/selfplay_records.jsonl \
    --output models/team_selection \
    --num-epochs 100
```

### 強化学習ループ内で学習

強化学習ループ実行時に `--train-team-selection` オプションを指定すると、Self-Play 中に収集した選出データを使って Team Selection Network も同時に学習する。

```bash
uv run python scripts/run_reinforcement_loop.py \
    --trainer-json data/top_rankers/season_36.json \
    --output models/reinforcement_with_selection \
    --num-generations 20 \
    --games-per-generation 100 \
    --train-team-selection \
    --team-selection-epochs 20 \
    --team-selection-update-interval 1
```

**オプション:**

- `--train-team-selection`: Team Selection Network の学習を有効化
- `--team-selection-epochs`: 学習エポック数（デフォルト: 20）
- `--team-selection-update-interval`: 何世代ごとに更新するか（デフォルト: 1）

### 使い方（コード内）

```python
from src.policy_value_network import (
    load_team_selector,
    RandomTeamSelector,
    TopNTeamSelector,
)

# NNベースのセレクター
selector = load_team_selector("models/team_selection", device="cpu")

# 選出
my_team = [...]  # 6匹のポケモンデータ
opp_team = [...]  # 相手の6匹
selected = selector.select(my_team, opp_team, num_select=3)

# ランダムセレクター
random_selector = RandomTeamSelector()
selected = random_selector.select(my_team, opp_team, num_select=3)

# 先頭N匹セレクター（従来の動作）
top_n_selector = TopNTeamSelector()
selected = top_n_selector.select(my_team, opp_team, num_select=3)
```

---

## データ変換ツール

### CSV → トレーナー JSON 変換

ポケモンの CSV データをトレーナー JSON 形式に変換するスクリプト。

```bash
uv run python scripts/convert_csv_to_trainer_json.py \
    --input data/season_36_pokemon_data.csv \
    --output data/top_rankers/season_36.json
```

**CSV の形式:**

```csv
rank,rating,trainer_name,pokemon1_name,pokemon1_item,pokemon1_nature,...
```

各ポケモンに対して `pokemonN_name`, `pokemonN_item`, `pokemonN_nature`, `pokemonN_evs`, `pokemonN_moves`, `pokemonN_tera_type` のカラムが必要（N=1〜6）。

---

## トーナメント

```bash
# Eloレーティングベースのトーナメント（resume対応）
DISCORD_WEBHOOK_URL={} \
POSTGRES_DB={} \
POSTGRES_PASSWORD={} \
POSTGRES_USER={} \
POSTGRES_HOST={} \
POSTGRES_PORT={} \
uv run python scripts/matching.py --resume
```

---

## LLM データセット生成

```bash
# 静的データセット生成（ダメージ計算ベース）
uv run python scripts/generate_llm_static_dataset.py \
    --trainer-json data/top_rankers/season_27.json \
    --output data/llm_static_dataset.jsonl \
    --num-battles 10000

# チャット形式に変換（SFT用）
uv run python scripts/convert_llm_static_to_chat_sft.py \
    --input data/llm_static_dataset.jsonl \
    --output data/llm_sft_chat_dataset.jsonl
```

---

## ダメージ計算 API

```bash
# FastAPIサーバー起動
uv run python src/damage_calculator_api/main.py

# または uvicorn で起動
uv run uvicorn src.damage_calculator_api.main:app --reload --port 8000
```

API ドキュメント: http://localhost:8000/docs

---

## Human vs ReBeL AI Battle Interface

学習した ReBeL AI モデルと人間が対戦できる Web インターフェース。

### 必要なもの

- 学習済みモデル（例: `models/rebel_selection_vs_my_party`）
- 固定パーティ JSON（例: `data/my_fixed_party.json`）
- Node.js + pnpm（フロントエンド用）

### セットアップ

```bash
# フロントエンドの依存関係インストール（初回のみ）
cd frontend
pnpm install
cd ..
```

### 起動方法

**ターミナル 1: バックエンド（FastAPI）**

```bash
uv run python -m src.battle_api.main
```

バックエンドは http://localhost:8001 で起動します。

**ターミナル 2: フロントエンド（Next.js）**

```bash
cd frontend
pnpm dev
```

フロントエンドは http://localhost:3000 で起動します。

### 使い方

1. ブラウザで http://localhost:3000 にアクセス
2. **対戦相手選択**: トレーナー一覧から対戦相手を選ぶ
3. **選出フェーズ**:
   - 相手の 6 匹が表示される
   - 自分のパーティ（`my_fixed_party.json`）から 3 匹を選ぶ
   - AI はニューラルネットワークで最適な 3 匹を自動選出
4. **バトルフェーズ**:
   - 技を選択（MOVE_0〜MOVE_3）または交代（SWITCH）
   - テラスタル可能（1 回のみ）
   - ダメージログがリアルタイムで表示される
5. 勝敗が決まると結果が表示される

### スクリーンショット

```
┌─────────────────────────────────────────────────────┐
│  相手の6匹を見て、あなたの3匹を選んでください        │
├─────────────────────────────────────────────────────┤
│  [相手のポケモン6匹のカード]                         │
│                                                     │
│  あなたのパーティ:                                   │
│  [選択可能なポケモン6匹のカード]                     │
│                                                     │
│  選出: ① アルセウス  ② コライドン  ③ サーフゴー    │
│                                [選出を確定]          │
└─────────────────────────────────────────────────────┘
```

### カスタマイズ

#### 自分のパーティを変更する

`data/my_fixed_party.json` を編集:

```json
{
  "pokemons": [
    {
      "name": "ポケモン名",
      "item": "持ち物",
      "ability": "特性",
      "nature": "性格",
      "Ttype": "テラスタイプ",
      "effort": [H, A, B, C, D, S],
      "moves": ["技1", "技2", "技3", "技4"]
    }
  ]
}
```

#### 使用するモデルを変更する

`src/battle_api/main.py` 内の以下のパスを変更:

```python
# モデルパス
MODEL_PATH = "models/rebel_selection_vs_my_party"

# パーティパス
FIXED_PARTY_PATH = "data/my_fixed_party.json"
```

### 技術構成

- **バックエンド**: FastAPI + Python
  - `src/battle_api/main.py`: セッション管理、ReBeL AI、バトル進行
- **フロントエンド**: Next.js 15 + shadcn/ui + Tailwind CSS
  - `frontend/src/app/page.tsx`: メインページ
  - `frontend/src/components/`: UI コンポーネント

---

## ユーティリティ

```bash
# バトル履歴抽出
uv run python src/utils/extract_battle_history.py

# Word2Vecモデル学習
uv run python src/utils/train_word2vec.py
```

---

## アーキテクチャ

```
src/
├── pokemon_battle_sim/     # 対戦シミュレータ
│   ├── battle.py          # バトル状態管理
│   ├── pokemon.py         # ポケモンクラス
│   └── damage.py          # ダメージ計算
├── hypothesis/            # 仮説ベースMCTS
│   ├── hypothesis_mcts.py # 不完全情報MCTS
│   └── selfplay.py        # Self-Playデータ生成
├── policy_value_network/  # 強化学習
│   ├── network.py         # Policy-Value Network
│   ├── nn_guided_mcts.py  # NN誘導MCTS
│   ├── reinforcement_loop.py  # 強化学習ループ
│   ├── team_selection_network.py  # Team Selection Network
│   └── team_selector.py   # チーム選出ユーティリティ
├── llm/                   # LLM関連
│   ├── policy.py          # LLMベースポリシー
│   └── state_representation.py  # 状態のテキスト表現
└── damage_calculator_api/ # ダメージ計算API
```

---

## 注意事項

- `Pokemon.init()` はコード内で最初に呼ぶ必要があります
- GPU がある場合は自動的に使用されます（`--device cuda` で明示指定可能）
- 強化学習ループは時間がかかります（軽量テストオプションを使用推奨）
