# ReBeL (Recursive Belief-based Learning) for Pokemon Battles

ポケモンバトルのような不完全情報ゲームに対する ReBeL 実装。

## 概要

ReBeL は Meta AI が開発した、不完全情報ゲームのための強化学習アルゴリズムです。
このモジュールでは、ポケモンバトルにおける以下の隠し情報を適切に扱います：

- 相手の技構成
- 相手の持ち物
- 相手のテラスタイプ
- 相手の努力値配分（EV）

## アーキテクチャ

```
┌─────────────────────────────────────────────────────────────┐
│                    ReBeL System                              │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │ Belief State │───▶│  PBS         │───▶│ CFR Solver   │  │
│  │              │    │ (Public      │    │              │  │
│  │ - 型仮説分布  │    │  Belief      │    │ - 戦略計算   │  │
│  │ - 持ち物分布  │    │  State)      │    │ - Nash均衡   │  │
│  │ - テラス分布  │    │              │    │              │  │
│  └──────────────┘    └──────────────┘    └──────────────┘  │
│         │                   │                   │           │
│         ▼                   ▼                   ▼           │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │ Usage DB     │    │ Value Network│    │ Action       │  │
│  │              │    │              │    │ Selection    │  │
│  │ - 使用率統計  │    │ - PBS評価    │    │              │  │
│  │ - 事前分布   │    │ - 勝率予測   │    │ - 探索/活用  │  │
│  └──────────────┘    └──────────────┘    └──────────────┘  │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## インストール

```bash
poetry install
```

## 基本的な使い方

### 1. ReBeL バトル AI を使用する

```python
from src.rebel import (
    ReBeLBattle,
    PokemonBeliefState,
    PublicBeliefState,
    ReBeLSolver,
    CFRConfig,
)
from src.hypothesis.pokemon_usage_database import PokemonUsageDatabase
from src.pokemon_battle_sim.battle import Battle
from src.pokemon_battle_sim.pokemon import Pokemon

# 使用率データベースを読み込み
usage_db = PokemonUsageDatabase.from_json("data/pokedb_usage/season_37_top150.json")

# バトル初期化
Pokemon.init()
battle = Battle()
battle.reset_game()

# チームを設定（省略）
# battle.selected[0].append(...)
# battle.selected[1].append(...)

# ターン0で場に出す
battle.proceed(commands=[Battle.SKIP, Battle.SKIP])

# ReBeL ソルバーを初期化
cfr_config = CFRConfig(
    num_iterations=50,      # CFR イテレーション数
    num_world_samples=20,   # 仮説サンプル数
)
solver = ReBeLSolver(
    value_network=None,     # None の場合はヒューリスティック評価
    cfr_config=cfr_config,
    use_simplified=True,    # 簡略化 CFR を使用
)

# 信念状態を初期化（相手のポケモン名のリストから）
belief = PokemonBeliefState(
    opponent_pokemon_names=[p.name for p in battle.selected[1]],
    usage_db=usage_db,
)

# PBS (Public Belief State) を構築
pbs = PublicBeliefState.from_battle(battle, perspective=0, belief=belief)

# 行動を選択
action = solver.get_action(pbs, battle, explore=False)
print(f"Selected action: {action}")
```

### 2. ReBeL vs MCTS 比較

```bash
# 基本的な比較（10試合）
PYTHONPATH=. poetry run python scripts/compare_rebel_vs_mcts.py \
  --trainer-json data/top_rankers/season_27.json \
  --usage-db data/pokedb_usage/season_37_top150.json \
  --num-matches 10

# パラメータ調整版
PYTHONPATH=. poetry run python scripts/compare_rebel_vs_mcts.py \
  --trainer-json data/top_rankers/season_27.json \
  --usage-db data/pokedb_usage/season_37_top150.json \
  --num-matches 50 \
  --rebel-cfr-iterations 50 \
  --rebel-world-samples 20 \
  --mcts-iterations 200 \
  --mcts-hypotheses 30 \
  --output results/rebel_vs_mcts.json
```

### 3. 強化学習トレーニング

```bash
# 軽量テスト（動作確認用）
PYTHONPATH=. poetry run python scripts/train_rebel.py \
  --trainer-json data/top_rankers/season_27.json \
  --usage-db data/pokedb_usage/season_37_top150.json \
  --output models/rebel_test \
  --num-iterations 5 \
  --games-per-iteration 10 \
  --cfr-iterations 20 \
  --cfr-world-samples 10 \
  --num-epochs 3

# 本格的なトレーニング
PYTHONPATH=. poetry run python scripts/train_rebel.py \
  --trainer-json data/top_rankers/season_27.json \
  --usage-db data/pokedb_usage/season_37_top150.json \
  --output models/rebel_full \
  --num-iterations 100 \
  --games-per-iteration 50 \
  --cfr-iterations 50 \
  --cfr-world-samples 20 \
  --batch-size 64 \
  --learning-rate 1e-4 \
  --num-epochs 10 \
  --hidden-dim 256 \
  --device cuda \
  --evaluate \
  --eval-games 50

# チェックポイントから再開
PYTHONPATH=. poetry run python scripts/train_rebel.py \
  --resume models/rebel_full/checkpoint_iter50 \
  --num-iterations 50 \
  --output models/rebel_full
```

## モジュール構成

### `src/rebel/`

| ファイル | 説明 |
|---------|------|
| `belief_state.py` | 信念状態の管理（`PokemonBeliefState`, `PokemonTypeHypothesis`） |
| `public_state.py` | 公開ゲーム状態（`PublicGameState`, `PublicBeliefState`） |
| `ev_template.py` | 努力値テンプレート推定 |
| `cfr_solver.py` | CFR サブゲーム解決（`CFRSubgameSolver`, `SimplifiedCFRSolver`, `ReBeLSolver`） |
| `value_network.py` | Value Network（`PBSEncoder`, `ReBeLValueNetwork`） |
| `battle_interface.py` | バトル AI インターフェース（`ReBeLBattle`） |
| `trainer.py` | 強化学習トレーナー（`ReBeLTrainer`） |

### `scripts/`

| ファイル | 説明 |
|---------|------|
| `compare_rebel_vs_mcts.py` | ReBeL と HypothesisMCTS の性能比較 |
| `train_rebel.py` | ReBeL 強化学習トレーニングスクリプト |

## 主要クラスの詳細

### PokemonTypeHypothesis

相手ポケモンの「型」仮説を表現する immutable なデータクラス。

```python
@dataclass(frozen=True)
class PokemonTypeHypothesis:
    moves: tuple[str, ...]    # 4技（ソート済み）
    item: str                 # 持ち物
    tera_type: str           # テラスタイプ
    nature: str              # 性格
    ability: str             # 特性
    ev_spread_type: EVSpreadType  # EV配分タイプ
```

### PokemonBeliefState

相手パーティ全体に対する信念（確率分布）を管理。

```python
belief = PokemonBeliefState(
    opponent_pokemon_names=["ガブリアス", "サーフゴー", "ハバタクカミ"],
    usage_db=usage_db,
    max_hypotheses=50,  # 保持する仮説の最大数
)

# 持ち物分布を取得
item_dist = belief.get_item_distribution("ガブリアス")
# {"きあいのタスキ": 0.35, "こだわりハチマキ": 0.25, ...}

# 仮説をサンプリング
worlds = belief.sample_worlds(n=10)
# [{"ガブリアス": Hypothesis(...), "サーフゴー": Hypothesis(...), ...}, ...]

# 観測で更新
from src.rebel import Observation, ObservationType
obs = Observation(
    pokemon_name="ガブリアス",
    observation_type=ObservationType.MOVE_USED,
    value="じしん",
)
belief.update(obs)
```

### EV テンプレート

性格と種族値から努力値配分を推定。

```python
from src.rebel import estimate_ev_spread_type, get_ev_spread, EVSpreadType

# 性格から EV 配分タイプを推定
ev_type = estimate_ev_spread_type(
    nature="ようき",              # 攻撃↑ 特攻↓
    base_stats=[108, 130, 95, 80, 85, 102],  # ガブリアスの種族値
)
# EVSpreadType.PHYSICAL_SPEED (AS252)

# EV 配分を取得
ev_spread = get_ev_spread("ようき", base_stats)
# EVSpread(hp=4, attack=252, defense=0, sp_attack=0, sp_defense=0, speed=252)
```

### CFR ソルバー

Counterfactual Regret Minimization による戦略計算。

```python
from src.rebel import CFRConfig, ReBeLSolver

config = CFRConfig(
    num_iterations=50,      # CFR イテレーション数
    num_world_samples=20,   # 仮説サンプル数
    exploration_rate=0.3,   # 探索率
)

solver = ReBeLSolver(
    value_network=None,     # or ReBeLValueNetwork instance
    cfr_config=config,
    use_simplified=True,    # True: 高速な簡略化版
)

# 戦略を計算
my_strategy, opp_strategy = solver.solve(pbs, battle)
# my_strategy: {action_id: probability}

# 行動を選択
action = solver.get_action(
    pbs, battle,
    explore=True,           # 探索モード
    temperature=1.0,        # 温度パラメータ
)
```

### Value Network

PBS から勝率を予測するニューラルネットワーク。

```python
from src.rebel import ReBeLValueNetwork, PBSEncoder

# ネットワーク作成
network = ReBeLValueNetwork(
    hidden_dim=256,
    num_res_blocks=4,
    dropout=0.1,
)

# 予測（推論）
my_value, opp_value = network.predict(pbs)
# my_value: 自分の勝率予測 (0.0 ~ 1.0)
# opp_value: 相手の勝率予測 (0.0 ~ 1.0)

# バッチ処理
my_values, opp_values = network.forward_batch(pbs_list)
```

### ReBeLTrainer

自己対戦による強化学習ループ。

```python
from src.rebel import ReBeLTrainer, TrainingConfig, ReBeLValueNetwork

config = TrainingConfig(
    games_per_iteration=50,
    max_turns=100,
    cfr_iterations=50,
    cfr_world_samples=20,
    batch_size=32,
    learning_rate=1e-4,
    num_epochs=10,
    device="cuda",
    save_interval=10,
)

trainer = ReBeLTrainer(
    usage_db=usage_db,
    trainer_data=trainer_data,  # トレーナー JSON データ
    config=config,
    value_network=ReBeLValueNetwork(hidden_dim=256),
)

# 学習実行
trainer.train(num_iterations=100, output_dir="models/rebel")

# 評価
results = trainer.evaluate_against_baseline(
    num_games=50,
    baseline_type="random",  # or "cfr_only"
)
```

## MCTS との比較

| 項目 | HypothesisMCTS | ReBeL |
|------|----------------|-------|
| 相手の技 | 完全に見える（チート） | 仮説からサンプリング |
| 相手の持ち物 | 仮説からサンプリング | 仮説からサンプリング |
| 相手のテラス | 完全に見える（チート） | 仮説からサンプリング |
| 相手の EV | 完全に見える（チート） | 性格+種族値から推定 |
| 計算速度 | 遅い（MCTS 探索） | 速い（CFR + NN） |
| 理論的根拠 | なし | Nash 均衡への収束 |

## トレーニングループの流れ

```
┌─────────────────────────────────────────┐
│ 1. 自己対戦データ生成                     │
│    - 2プレイヤーが CFR で戦略計算         │
│    - PBS (Public Belief State) を記録    │
│    - 勝敗結果をターゲット値として設定      │
└─────────────────┬───────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────┐
│ 2. Value Network 学習                    │
│    - PBS → Value 予測                    │
│    - MSE Loss でターゲット値と比較        │
│    - 勾配更新                            │
└─────────────────┬───────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────┐
│ 3. 繰り返し                              │
│    - 改善された VN でより正確な評価       │
│    - より良い戦略の学習                   │
└─────────────────────────────────────────┘
```

## パフォーマンスチューニング

### 速度優先

```python
config = CFRConfig(
    num_iterations=20,
    num_world_samples=5,
)
solver = ReBeLSolver(
    value_network=trained_network,  # 学習済み VN を使用
    cfr_config=config,
    use_simplified=True,
)
```

### 精度優先

```python
config = CFRConfig(
    num_iterations=100,
    num_world_samples=50,
)
solver = ReBeLSolver(
    value_network=trained_network,
    cfr_config=config,
    use_simplified=False,  # 完全 CFR を使用
)
```

## 制限事項

1. **ダブルバトル非対応**: 現在はシングルバトルのみ
2. **特性の推定**: 特性は使用率データから推定するが、精度が低い場合がある
3. **動的な信念更新**: バトル中の観測による信念更新は簡略化されている
4. **計算コスト**: CFR の完全版は計算コストが高い

## 今後の拡張予定

- [ ] ダブルバトル対応
- [ ] より詳細な観測からの信念更新
- [ ] Policy Network の追加（CFR 代替）
- [ ] 分散トレーニング対応
