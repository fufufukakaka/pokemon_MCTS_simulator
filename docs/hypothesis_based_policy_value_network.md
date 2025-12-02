# 仮説ベース Policy-Value Network 設計書

## 概要

ポケモン対戦（不完全情報ゲーム）において、任意の盤面から「各行動の推奨確率（Policy）」と「現在の勝率（Value）」を予測するモデルを構築する。

## 最終ゴール

```
入力: 観測情報のみの盤面
  - 自分のパーティ (6体の詳細)
  - 相手のパーティ (見せ合いで見えた6体の種族)
  - 現在の場の状態 (誰が出ているか、HP、状態異常等)
  - これまでの観測 (相手が使った技、判明した持ち物等)

出力:
  - Policy: 各行動の推奨確率 {技1: 0.4, 技2: 0.3, 交代A: 0.2, ...}
  - Value: 現在の勝率スコア (0.0〜1.0)
```

## 設計方針

| 項目 | 決定 |
|------|------|
| 入力 | 観測情報のみ（相手の隠し情報は内部で推論） |
| アーキテクチャ | ハイブリッド（MCTS→NNの教師データ→NN推論） |
| 不完全情報の扱い | 期待値で1つの答えを出力 |
| 学習データ生成 | Self-Play |

## 不完全情報の扱い

ポケモン対戦の主要な不完全情報要素:
1. 相手の控えポケモンの詳細（技構成、持ち物、努力値）
2. 相手の技構成（最初は不明）
3. 相手の持ち物

**今回は「持ち物」に焦点を当てる**。持ち物は対戦中に判明しやすく、型を大きく規定するため、仮説空間を現実的なサイズに抑えられる。

### 持ち物が判明するタイミング

| 観測 | 判明する持ち物 |
|------|---------------|
| HP1で耐えた | きあいのタスキ |
| 技が固定された | こだわり系 |
| 2回行動した | こだわりスカーフ確定 |
| ダメージ量が異常に高い | こだわりハチマキ/メガネ |
| 状態異常無効 | ラムのみ |
| HP回復した | たべのこし、オボンのみ |
| ブーストがかかった | ブーストエナジー |

## 全体アーキテクチャ

### 学習フェーズ

```
┌─────────────────────────────────────────────────────────────────┐
│                        学習フェーズ                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. 盤面生成 (Self-Play)                                        │
│     - 仮説MCTSエージェント同士を対戦させる                        │
│     - 実戦に近い盤面が得られる                                   │
│                                                                 │
│  2. 仮説ベースMCTS                                               │
│     ┌────────────────────────────────────────┐                  │
│     │ 盤面 + 観測情報                         │                  │
│     │     ↓                                  │                  │
│     │ ItemBeliefState (持ち物の確率分布)      │                  │
│     │     ↓                                  │                  │
│     │ N個の仮説をサンプリング                 │                  │
│     │     ↓                                  │                  │
│     │ 各仮説でMCTS実行                        │                  │
│     │     ↓                                  │                  │
│     │ 結果を確率で重み付け集約                │                  │
│     │     ↓                                  │                  │
│     │ Policy (行動確率), Value (勝率)         │                  │
│     └────────────────────────────────────────┘                  │
│                                                                 │
│  3. 教師データとして保存                                         │
│     {観測盤面, Policy, Value} → dataset.jsonl                   │
│                                                                 │
│  4. Neural Network学習                                          │
│     観測盤面 → NN → Policy, Value                               │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 推論フェーズ

```
┌─────────────────────────────────────────────────────────────────┐
│                        推論フェーズ                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ユーザー入力 (観測情報)                                         │
│      ↓                                                          │
│  学習済みNN                                                      │
│      ↓                                                          │
│  Policy: {技1: 0.4, 技2: 0.3, 交代A: 0.2, ...}                  │
│  Value: 0.65                                                    │
│                                                                 │
│  (オプション: 精度が必要なら仮説MCTSも併用)                       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Self-Play 学習ループ

```
┌─────────────────────────────────────────────────────────────────┐
│                     Self-Play 学習ループ                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ Generation 0: 仮説MCTSのみで対戦                         │   │
│  │                                                         │   │
│  │   Agent A (仮説MCTS) vs Agent B (仮説MCTS)              │   │
│  │              ↓                                          │   │
│  │   各ターンの {観測盤面, Policy, Value} を記録            │   │
│  │              ↓                                          │   │
│  │   対戦終了 → 勝敗でValueを補正                          │   │
│  │              ↓                                          │   │
│  │   Dataset v0                                            │   │
│  └─────────────────────────────────────────────────────────┘   │
│                          ↓                                      │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ Neural Network v1 を学習                                 │   │
│  │                                                         │   │
│  │   Dataset v0 → 学習 → NN v1                             │   │
│  └─────────────────────────────────────────────────────────┘   │
│                          ↓                                      │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ Generation 1: NN + MCTS で対戦                          │   │
│  │                                                         │   │
│  │   Agent A (NN v1 + 仮説MCTS) vs Agent B (同)            │   │
│  │   - NNでPolicy/Valueを高速に推定                        │   │
│  │   - MCTSの探索をNNで誘導（探索効率UP）                   │   │
│  │              ↓                                          │   │
│  │   Dataset v1                                            │   │
│  └─────────────────────────────────────────────────────────┘   │
│                          ↓                                      │
│                        ...繰り返し...                            │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## コンポーネント設計

### 1. ItemPriorDatabase

ポケモンごとの持ち物事前確率分布を管理。

```python
# データ構造
{
  "カイリュー": {"いかさまダイス": 0.33, "こだわりハチマキ": 0.33, ...},
  "ハバタクカミ": {"ブーストエナジー": 0.4, "こだわりメガネ": 0.35, ...},
  ...
}
```

データソース: `data/top_rankers/season_27.json`

### 2. ItemBeliefState

対戦中の相手持ち物に対する信念（確率分布）を管理。

```python
class ItemBeliefState:
    """対戦中の相手持ち物に対する信念"""

    def __init__(self, opponent_pokemon_names, prior_db):
        # 事前分布で初期化
        self.beliefs = {name: prior_db[name] for name in opponent_pokemon_names}

    def update(self, pokemon_name, observation):
        """観測に基づいてベイズ更新"""
        # 例: "きあいのタスキ発動" → タスキ確率=1.0, 他=0.0
        pass

    def sample_hypothesis(self) -> dict[str, str]:
        """現在の信念から持ち物の組み合わせを1つサンプリング"""
        pass
```

### 3. HypothesisMCTS

仮説ベースのMCTS。複数の持ち物仮説に対してMCTSを実行し、結果を集約。

```python
class HypothesisMCTS:
    """仮説ベースのMCTS"""

    def search(self, battle, player, belief_state, n_hypotheses=50, mcts_iterations=200):
        results = []

        for _ in range(n_hypotheses):
            # 仮説をサンプリング
            item_hypothesis = belief_state.sample_hypothesis()

            # 仮説を適用したBattleを作成
            hypo_battle = self.apply_hypothesis(battle, item_hypothesis)

            # MCTS実行
            policy, value = self.run_mcts(hypo_battle, player, mcts_iterations)
            results.append((policy, value))

        # 結果を集約（平均）
        return self.aggregate(results)
```

### 4. ObservationEncoder

観測情報をNNの入力形式（テンソル）に変換。

```python
class ObservationEncoder:
    """観測情報をNNの入力形式に変換"""

    def encode(self, battle, player, belief_state) -> Tensor:
        # 自分の情報（完全）
        # 相手の情報（観測されたもののみ）
        # 持ち物の信念分布
        # 場の状態
        pass
```

### 5. PolicyValueNetwork

Policy-Value を同時に予測するニューラルネットワーク。

```python
class PolicyValueNetwork(nn.Module):
    """Policy-Value Network"""

    def forward(self, x):
        # 共通の特徴抽出
        features = self.encoder(x)

        # Policy head: 各行動の確率
        policy = self.policy_head(features)  # softmax

        # Value head: 勝率
        value = self.value_head(features)    # sigmoid → [0, 1]

        return policy, value
```

## 実装ロードマップ

### Phase 1: 基盤構築 ✅ 完了
- [x] ItemPriorDatabase（持ち物事前確率）
- [x] ItemBeliefState（信念状態管理）
- [x] 観測からの信念更新ロジック

### Phase 2: 仮説MCTS ✅ 完了
- [x] 既存MCTSを拡張して仮説サンプリング対応
- [x] 複数仮説の結果集約
- [x] HypothesisMCTSBattleクラス（MyMCTSBattleの仮説対応版）

### Phase 3: Self-Play データ生成 ✅ 完了
- [x] 仮説MCTS同士の対戦ループ（SelfPlayGenerator）
- [x] 盤面・Policy・Valueの記録フォーマット（TurnRecord, GameRecord）
- [x] データ生成スクリプト（scripts/generate_selfplay_dataset.py）

### Phase 4: Neural Network ✅ 完了
- [x] ObservationEncoder（盤面→テンソル）
- [x] PolicyValueNetwork設計・学習
- [x] Trainer（学習ループ）
- [x] NN誘導型MCTS（NNGuidedMCTS）

### Phase 5: 強化学習ループ ✅ 完了
- [x] モデル評価器（新旧モデル対戦）
- [x] 強化学習ループ（Self-Play → 学習 → 評価 → 採用判定）
- [x] 実行スクリプト

## ファイル構成

```
src/
├── hypothesis/
│   ├── __init__.py
│   ├── item_prior_database.py    # Phase 1 ✅
│   ├── item_belief_state.py      # Phase 1 ✅
│   ├── hypothesis_mcts.py        # Phase 2 ✅
│   └── selfplay.py               # Phase 3 ✅
├── policy_value_network/
│   ├── __init__.py               # Phase 4 ✅
│   ├── observation_encoder.py    # Phase 4 ✅
│   ├── network.py                # Phase 4 ✅
│   ├── dataset.py                # Phase 4 ✅
│   ├── trainer.py                # Phase 4 ✅
│   ├── nn_guided_mcts.py         # Phase 4 ✅ (NN誘導型MCTS)
│   ├── evaluator.py              # Phase 5 ✅ (モデル評価)
│   └── reinforcement_loop.py     # Phase 5 ✅ (強化学習ループ)
scripts/
├── generate_selfplay_dataset.py  # Phase 3 ✅
├── train_policy_value_network.py # Phase 4 ✅
└── run_reinforcement_loop.py     # Phase 5 ✅
```

## 使い方

### 1. 単発のSelf-Play + 学習

```bash
# Self-Playデータ生成
poetry run python scripts/generate_selfplay_dataset.py \
  --num-games 100 --output data/selfplay_dataset.jsonl

# Neural Network学習
poetry run python scripts/train_policy_value_network.py \
  --dataset data/selfplay_dataset.jsonl \
  --output models/policy_value_v1
```

### 2. 強化学習ループ（AlphaZeroスタイル）

```bash
# 完全な強化学習ループ
poetry run python scripts/run_reinforcement_loop.py \
  --trainer-json data/top_rankers/season_27.json \
  --output models/reinforcement \
  --num-generations 10 \
  --games-per-generation 100

# 軽量テスト
poetry run python scripts/run_reinforcement_loop.py \
  --num-generations 3 \
  --games-per-generation 20 \
  --evaluation-games 10 \
  --training-epochs 10
```

## 参考資料

- AlphaGo / AlphaZero: Policy-Value Networkの基本アーキテクチャ
- Pluribus (ポーカーAI): 不完全情報ゲームへのアプローチ
- Information Set MCTS: 不完全情報を扱うMCTSの拡張
