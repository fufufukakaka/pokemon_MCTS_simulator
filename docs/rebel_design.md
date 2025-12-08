# ReBeL (Recursive Belief-based Learning) 実装設計書

## 概要

ポケモンバトルは不完全情報ゲームであり、以下の情報が隠されている:
- 相手の技構成 (4技中、使用されるまで不明)
- 相手の持ち物
- 相手のテラスタイプ
- 相手の努力値配分

現在の HypothesisMCTS は持ち物のみを仮説としてサンプリングしているが、
ReBeL 方式では **Public Belief State (PBS)** を用いて、より包括的に不確実性を扱う。

## ReBeL アルゴリズムの概要

```
1. 公開状態 h と信念状態 β(h) から PBS を構築
2. PBS に対して価値ネットワーク V(PBS) を適用
3. サブゲーム（現在のターン）を CFR で解く
4. 得られた戦略と価値で自己対戦
5. 対戦結果で価値ネットワークを学習
```

## コンポーネント設計

### 1. PokemonBeliefState

相手ポケモンの型全体に対する信念を管理。

```python
@dataclass
class PokemonTypeHypothesis:
    """ポケモンの型仮説"""
    moves: list[str]        # 4技
    item: str               # 持ち物
    tera_type: str          # テラスタイプ
    nature: str             # 性格
    # EVは固定パターンを仮定（テンプレ型）

@dataclass
class PokemonBeliefState:
    """相手パーティ全体の信念状態"""

    # 各ポケモンの型仮説と確率
    beliefs: dict[str, dict[PokemonTypeHypothesis, float]]

    # 観測済み情報（確定情報）
    revealed_moves: dict[str, set[str]]    # 使用された技
    revealed_items: dict[str, str | None]  # 判明した持ち物
    revealed_tera: dict[str, str | None]   # 使用されたテラス

    def update(self, observation: Observation) -> None:
        """観測に基づくベイズ更新"""

    def sample_world(self) -> dict[str, PokemonTypeHypothesis]:
        """信念から1つの「世界」をサンプリング"""
```

### 2. PublicGameState

公開情報のみで構成されるゲーム状態。

```python
@dataclass
class PublicGameState:
    """観測可能な情報のみの状態"""

    # 自分の情報（完全）
    my_pokemon: list[PokemonState]
    my_active: int
    my_tera_used: bool

    # 相手の情報（観測された部分のみ）
    opp_pokemon_names: list[str]
    opp_active: int
    opp_hp_ratios: list[float]
    opp_ailments: list[str]
    opp_ranks: list[list[int]]
    opp_tera_used: bool
    opp_revealed_moves: dict[str, set[str]]
    opp_revealed_items: dict[str, str | None]

    # 場の状態
    field: FieldCondition
    turn: int
```

### 3. PublicBeliefState (PBS)

ReBeL の中核となる状態表現。

```python
@dataclass
class PublicBeliefState:
    """公開状態 + 信念状態"""

    public_state: PublicGameState
    belief: PokemonBeliefState

    # 両プレイヤーの現在の戦略（CFR で更新）
    strategies: tuple[dict[int, float], dict[int, float]]

    # 両プレイヤーの期待値
    values: tuple[float, float]
```

### 4. ReBeL Value Network

PBS を入力として価値を予測するネットワーク。

```python
class ReBeLValueNetwork(nn.Module):
    """PBS から両プレイヤーの期待値を予測"""

    def __init__(self, ...):
        # 公開状態エンコーダー
        self.public_encoder = PublicStateEncoder(...)

        # 信念状態エンコーダー
        self.belief_encoder = BeliefEncoder(...)

        # 戦略エンコーダー
        self.strategy_encoder = StrategyEncoder(...)

        # 価値予測ヘッド
        self.value_head = nn.Sequential(...)

    def forward(self, pbs: PublicBeliefState) -> tuple[float, float]:
        """両プレイヤーの期待値を予測"""
```

### 5. CFR サブゲーム解決

現在のターンを CFR (Counterfactual Regret Minimization) で解く。

```python
class CFRSubgameSolver:
    """1ターンのサブゲームをCFRで解く"""

    def __init__(
        self,
        value_network: ReBeLValueNetwork,
        usage_db: PokemonUsageDatabase,
        num_iterations: int = 100,
    ):
        self.value_network = value_network
        self.usage_db = usage_db
        self.num_iterations = num_iterations

    def solve(self, pbs: PublicBeliefState) -> tuple[dict[int, float], dict[int, float]]:
        """
        CFR でサブゲームを解く

        Returns:
            両プレイヤーの平均戦略
        """
        # 累積リグレットと累積戦略
        regrets = [{}, {}]
        strategies = [{}, {}]

        for _ in range(self.num_iterations):
            # 信念からワールドをサンプリング
            world = pbs.belief.sample_world()

            # 具体的な Battle 状態を構築
            battle = self._instantiate_battle(pbs.public_state, world)

            # CFR の1イテレーション
            for player in [0, 1]:
                self._cfr_iteration(battle, player, regrets, strategies, pbs)

        # 平均戦略を計算
        return self._compute_average_strategy(strategies)
```

### 6. ReBeL 学習ループ

```python
class ReBeLTrainer:
    """ReBeL の学習ループ"""

    def __init__(
        self,
        value_network: ReBeLValueNetwork,
        usage_db: PokemonUsageDatabase,
        cfr_iterations: int = 100,
        learning_rate: float = 1e-4,
    ):
        self.value_network = value_network
        self.cfr_solver = CFRSubgameSolver(value_network, usage_db, cfr_iterations)
        self.optimizer = torch.optim.Adam(value_network.parameters(), lr=learning_rate)

    def generate_training_data(self, num_games: int) -> list[TrainingExample]:
        """自己対戦でデータ生成"""
        examples = []

        for _ in range(num_games):
            battle = self._create_random_battle()
            pbs = self._init_pbs(battle)

            while battle.winner() is None:
                # CFR でサブゲーム解決
                strategies = self.cfr_solver.solve(pbs)

                # 学習データとして保存
                examples.append(TrainingExample(
                    pbs=pbs.copy(),
                    target_strategies=strategies,
                    # 終局時に実際の結果で更新
                ))

                # 行動選択と状態遷移
                actions = self._sample_actions(strategies)
                pbs = self._transition(pbs, actions)

            # 終局結果で価値ターゲットを更新
            self._update_value_targets(examples, battle.winner())

        return examples

    def train_step(self, examples: list[TrainingExample]) -> float:
        """1ステップの学習"""
        # 価値ネットワークの予測と実際の結果の損失を計算
        # ...
```

## 実装優先順位

### Phase 1: 基盤 (Week 1-2)
1. `PokemonBeliefState` の実装
   - `PokemonUsageDatabase` との統合
   - 観測による更新ロジック
2. `PublicGameState` の実装
   - `Battle` からの変換
3. テストの作成

### Phase 2: CFR (Week 3-4)
1. 簡易 CFR の実装
   - 1ターンのサブゲーム解決
   - 仮説サンプリングベース
2. `ReBeLValueNetwork` の実装
   - 既存 `PolicyValueNetwork` を拡張

### Phase 3: 学習ループ (Week 5-6)
1. `ReBeLTrainer` の実装
2. 自己対戦データ生成
3. 学習パイプライン

### Phase 4: 評価・最適化 (Week 7-8)
1. HypothesisMCTS との性能比較
2. ハイパーパラメータチューニング
3. 計算効率の最適化

## 技術的考慮事項

### 計算量の削減

ReBeL は計算コストが高いため、以下の工夫が必要:

1. **仮説の枝刈り**: 事前確率が低い型は除外
2. **深さの制限**: CFR の探索深さを制限（1-2ターン先読み）
3. **サンプリング数の調整**: 仮説サンプル数を動的に調整
4. **キャッシング**: 同一 PBS の結果をキャッシュ

### 信念状態の表現

組み合わせ爆発を防ぐため:

1. **独立仮定**: 技・持ち物・テラスは独立と仮定
2. **テンプレ型への集約**: 使用率上位の型パターンに集約
3. **段階的詳細化**: ゲーム進行に応じて詳細化

### ポケモンバトル特有の考慮

1. **確定情報の増加**: ターンが進むと情報が増える
2. **テラスタルの1回制限**: 戦略的に重要な不可逆選択
3. **交代の情報価値**: 控えを出すと技が見える可能性

## ファイル構成

```
src/rebel/
├── __init__.py
├── belief_state.py       # PokemonBeliefState
├── public_state.py       # PublicGameState, PublicBeliefState
├── value_network.py      # ReBeLValueNetwork
├── cfr_solver.py         # CFRSubgameSolver
├── trainer.py            # ReBeLTrainer
└── battle_interface.py   # Battle との統合

scripts/
├── train_rebel.py        # 学習スクリプト
└── evaluate_rebel.py     # 評価スクリプト
```

## 参考文献

- Brown et al., "Combining Deep Reinforcement Learning and Search for Imperfect-Information Games" (ReBeL paper)
- Zinkevich et al., "Regret Minimization in Games with Incomplete Information" (CFR paper)
