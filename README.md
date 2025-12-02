# pokemon_MCTS_simulator

ポケモン対戦シミュレータ + MCTS + 強化学習

## 機能

- **ポケモン対戦シミュレーション**: Gen9対応の対戦シミュレータ
- **MCTS (Monte Carlo Tree Search)**: 仮説ベースのMCTSで不完全情報ゲームに対応
- **Policy-Value Network**: AlphaZeroスタイルの行動予測・勝率予測ネットワーク
- **Team Selection Network**: 相手チームを見て最適な3匹を選出するネットワーク
- **強化学習ループ**: Self-Play → 学習 → 評価のサイクルで継続的に強化

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

### 1. Self-Playデータ生成

```bash
# 純粋MCTSでSelf-Playデータを生成
poetry run python scripts/generate_selfplay_dataset.py \
    --trainer-json data/top_rankers/season_27.json \
    --output data/selfplay_records.jsonl \
    --num-games 100 \
    --mcts-iterations 100
```

### 2. Policy-Value Network 学習

```bash
# Self-PlayデータでPolicy-Value Networkを学習
poetry run python scripts/train_policy_value_network.py \
    --dataset data/selfplay_records.jsonl \
    --output models/policy_value \
    --hidden-dim 256 \
    --num-epochs 100
```

### 3. 強化学習ループ（Self-Play → 学習 → 評価 → 繰り返し）

```bash
# AlphaZeroスタイルの強化学習ループ
poetry run python scripts/run_reinforcement_loop.py \
    --trainer-json data/top_rankers/season_27.json \
    --output models/reinforcement \
    --num-generations 10 \
    --games-per-generation 100 \
    --evaluation-games 50

# 軽量テスト
poetry run python scripts/run_reinforcement_loop.py \
    --trainer-json data/top_rankers/season_27.json \
    --output models/reinforcement_test \
    --num-generations 3 \
    --games-per-generation 20 \
    --evaluation-games 10 \
    --training-epochs 10
```

---

## Team Selection Network

相手の6匹を見て、自分の6匹から最適な3匹を選出するネットワーク。

### 学習

```bash
# ランダムデータで初期学習
poetry run python scripts/train_team_selection.py \
    --trainer-json data/top_rankers/season_27.json \
    --output models/team_selection \
    --num-samples 10000 \
    --num-epochs 50

# Self-Playデータで学習（より高品質）
poetry run python scripts/train_team_selection.py \
    --trainer-json data/top_rankers/season_27.json \
    --selfplay-data data/selfplay_records.jsonl \
    --output models/team_selection \
    --num-epochs 100
```

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

## トーナメント

```bash
# Eloレーティングベースのトーナメント（resume対応）
DISCORD_WEBHOOK_URL={} \
POSTGRES_DB={} \
POSTGRES_PASSWORD={} \
POSTGRES_USER={} \
POSTGRES_HOST={} \
POSTGRES_PORT={} \
poetry run python scripts/matching.py --resume
```

---

## LLMデータセット生成

```bash
# 静的データセット生成（ダメージ計算ベース）
poetry run python scripts/generate_llm_static_dataset.py \
    --trainer-json data/top_rankers/season_27.json \
    --output data/llm_static_dataset.jsonl \
    --num-battles 10000

# チャット形式に変換（SFT用）
poetry run python scripts/convert_llm_static_to_chat_sft.py \
    --input data/llm_static_dataset.jsonl \
    --output data/llm_sft_chat_dataset.jsonl
```

---

## ダメージ計算API

```bash
# FastAPIサーバー起動
poetry run python src/damage_calculator_api/main.py

# または uvicorn で起動
poetry run uvicorn src.damage_calculator_api.main:app --reload --port 8000
```

API ドキュメント: http://localhost:8000/docs

---

## ユーティリティ

```bash
# バトル履歴抽出
poetry run python src/utils/extract_battle_history.py

# Word2Vecモデル学習
poetry run python src/utils/train_word2vec.py
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
- GPUがある場合は自動的に使用されます（`--device cuda` で明示指定可能）
- 強化学習ループは時間がかかります（軽量テストオプションを使用推奨）
