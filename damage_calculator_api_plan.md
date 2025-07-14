# ポケモンSVダメージ計算機API開発計画

## 現在のアーキテクチャ分析

### 優位性
- **高精度計算エンジン**: `oneshot_damages()` は全補正要素を考慮した完全実装
- **包括的データセット**: SV世代対応、テラスタル機能込み
- **実戦レベル精度**: 16段階ダメージロール、全ての特性・道具・状況補正

### 構造的課題
- **密結合設計**: Battle クラス依存で単発計算が困難
- **状態管理過剰**: フル対戦状態が必要だが、ダメージ計算には不要
- **API未対応**: Web API化されていない

## 4段階開発計画

### Phase 1: Core Damage Engine Extraction (2週間)
**目標**: Battle依存を解消し、独立したダメージ計算エンジンを構築

#### 1.1 ダメージ計算コアの抽出
```python
class DamageCalculator:
    def __init__(self):
        # 静的データの初期化（Pokemon.init() 相当）
        self.load_pokemon_data()
        self.load_move_data()
        self.load_item_data()
    
    def calculate_damage(self, attacker: PokemonState, defender: PokemonState, 
                        move: MoveInput, conditions: BattleConditions) -> DamageResult
```

#### 1.2 最小限データ構造の設計
```python
@dataclass
class PokemonState:
    species: str
    level: int = 50
    stats: Dict[str, int]  # 実数値
    nature: str = "まじめ"
    ability: str
    item: Optional[str] = None
    terastal_type: Optional[str] = None
    is_terastalized: bool = False
    status_ailment: Optional[str] = None
    stat_boosts: Dict[str, int] = field(default_factory=lambda: {})

@dataclass
class MoveInput:
    name: str
    move_type: Optional[str] = None  # タイプ変更特性用
    is_critical: bool = False
    power_modifier: float = 1.0

@dataclass
class BattleConditions:
    weather: Optional[str] = None
    terrain: Optional[str] = None
    trick_room: bool = False
    gravity: bool = False
    
@dataclass
class DamageResult:
    damage_range: List[int]
    damage_percentage: List[float]
    ko_probability: float
    guaranteed_ko_hits: int
    calculation_details: dict
```

#### 1.3 依存関数の段階的移植
既存のBattle クラスメソッドを独立した関数に変換:

- `attack_type_correction()` → `calculate_stab_modifier()`
- `defence_type_correction()` → `calculate_type_effectiveness()`
- `power_correction()` → `calculate_move_power()`
- `attack_correction()` → `calculate_attack_stat()`
- `defence_correction()` → `calculate_defense_stat()`
- `damage_correction()` → `calculate_final_modifier()`
- `critical_probability()` → `calculate_crit_probability()`

### Phase 2: API Design & FastAPI Setup (1週間)
**目標**: RESTful API設計とFastAPIアプリケーション基盤構築

#### 2.1 エンドポイント設計
```python
# 基本ダメージ計算
POST /api/v1/damage/calculate      # 単発ダメージ計算
POST /api/v1/damage/range          # ダメージ範囲計算
POST /api/v1/damage/matchup        # 複数技比較
GET  /api/v1/pokemon/{name}        # ポケモン情報取得
GET  /api/v1/moves/{name}          # 技情報取得
GET  /api/v1/types/effectiveness   # タイプ相性表
```

#### 2.2 リクエスト/レスポンス形式定義
```python
class DamageRequest(BaseModel):
    attacker: PokemonState
    defender: PokemonState
    move: MoveInput
    field_conditions: Optional[FieldConditions] = None

class DamageResponse(BaseModel):
    damage_range: List[int]           # [min, max]
    damage_percentage: List[float]    # HP割合
    ko_probability: float
    guaranteed_ko_hits: int
    calculation_details: CalculationBreakdown

class CalculationBreakdown(BaseModel):
    base_damage: int
    type_effectiveness: float
    stab_modifier: float
    ability_modifier: float
    item_modifier: float
    weather_modifier: float
    critical_hit: bool
    random_factor: float
```

#### 2.3 FastAPIアプリケーション基盤
```python
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="Pokemon SV Damage Calculator API",
    description="高精度ポケモンSVダメージ計算API",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### Phase 3: Implementation & Testing (3週間)
**目標**: 完全機能するAPI実装とテスト

#### 3.1 FastAPIアプリケーション実装
```python
@app.post("/api/v1/damage/calculate")
async def calculate_damage(request: DamageRequest) -> DamageResponse:
    try:
        calculator = DamageCalculator()
        result = calculator.calculate_damage(
            request.attacker, request.defender, 
            request.move, request.field_conditions
        )
        return DamageResponse.from_calculation(result)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/api/v1/pokemon/{name}")
async def get_pokemon_info(name: str) -> PokemonInfo:
    data = PokemonDataRepository.get_pokemon_data(name)
    if not data:
        raise HTTPException(status_code=404, detail="Pokemon not found")
    return PokemonInfo.from_data(data)
```

#### 3.2 テスト戦略
**Unit Tests**: 各補正関数の単体テスト
```python
def test_type_effectiveness():
    # でんき vs みず = 2.0倍
    assert calculate_type_effectiveness("でんき", ["みず"]) == 2.0
    
def test_stab_calculation():
    # テラスタル時のSTAB計算
    assert calculate_stab_modifier("ほのお", ["ほのお"], "ほのお", True) == 2.0
```

**Integration Tests**: 既存battle.pyとの結果照合テスト
```python
def test_damage_calculation_accuracy():
    # 既存システムとの結果比較
    battle_result = existing_battle_system.calculate()
    api_result = new_calculator.calculate()
    assert battle_result == api_result
```

**Performance Tests**: 大量リクエスト処理テスト
```python
def test_concurrent_requests():
    # 100並行リクエストで<100ms平均レスポンス
    pass
```

#### 3.3 エラーハンドリング
- 不正なポケモン名/技名の検証
- 無効な能力値範囲チェック
- データ不整合の検出と修正
- 適切なHTTPステータスコード返却

### Phase 4: Advanced Features (2週間)
**目標**: 実用的ダメージ計算機としての機能拡張

#### 4.1 高度な計算機能
```python
@app.post("/api/v1/damage/range")
async def damage_range_analysis(request: RangeRequest):
    # 確定数計算、KO確率計算
    return RangeAnalysisResponse(
        min_rolls_to_ko=2,
        max_rolls_to_ko=3,
        ko_probability_by_rolls=[0.0, 0.0, 0.6875, 1.0]
    )
    
@app.post("/api/v1/damage/matchup")
async def move_comparison(request: MatchupRequest):
    # 複数技の威力比較
    return MatchupResponse(
        moves=[
            {"name": "10まんボルト", "damage_range": [150, 176]},
            {"name": "かみなり", "damage_range": [180, 211]}
        ],
        recommendation="かみなり"
    )
```

#### 4.2 パフォーマンス最適化
**データキャッシング戦略**
```python
from functools import lru_cache

class PokemonDataRepository:
    @lru_cache(maxsize=1000)
    def get_pokemon_data(self, name: str) -> PokemonData:
        return self._load_pokemon_data(name)
```

**非同期処理対応**
```python
import asyncio

async def batch_damage_calculation(requests: List[DamageRequest]):
    tasks = [calculate_damage_async(req) for req in requests]
    return await asyncio.gather(*tasks)
```

## 実装上の重要検討事項

### 技術スタック
- **Backend**: FastAPI + Python 3.12
- **Validation**: Pydantic v2
- **Testing**: pytest + httpx
- **Documentation**: FastAPI自動生成Swagger UI
- **Deployment**: Docker + uvicorn

### アーキテクチャパターン
**Repository Pattern**: データアクセス層分離
```python
class PokemonDataRepository:
    def get_pokemon_data(self, name: str) -> PokemonData
    def get_move_data(self, name: str) -> MoveData
    def get_type_effectiveness(self) -> TypeChart
```

**Factory Pattern**: ポケモンインスタンス生成
```python
class PokemonFactory:
    @staticmethod
    def create_pokemon(species: str, **kwargs) -> Pokemon
```

**Strategy Pattern**: 計算方式の切り替え
```python
class DamageCalculationStrategy:
    def calculate(self, attacker, defender, move) -> DamageResult
```

### データ管理戦略
```python
class DataManager:
    def __init__(self):
        self._pokemon_cache = {}
        self._move_cache = {}
        self._type_chart = None
        self._load_all_data()
    
    def _load_all_data(self):
        # data/ ディレクトリから全データ読み込み
        self._load_pokemon_data()
        self._load_move_data()
        self._load_type_chart()
```

### パフォーマンス目標
- **レスポンス時間**: <100ms (95percentile)
- **同時リクエスト**: 100 req/sec
- **メモリ使用量**: <512MB
- **稼働率**: 99.9% uptime

## ファイル構成提案

```
src/
├── damage_calculator_api/
│   ├── __init__.py
│   ├── main.py                    # FastAPIアプリエントリーポイント
│   ├── models/
│   │   ├── __init__.py
│   │   ├── request_models.py      # リクエストモデル
│   │   ├── response_models.py     # レスポンスモデル
│   │   └── pokemon_models.py      # ポケモンデータモデル
│   ├── calculators/
│   │   ├── __init__.py
│   │   ├── damage_calculator.py   # メインダメージ計算エンジン
│   │   ├── stat_calculator.py     # 能力値計算
│   │   └── type_calculator.py     # タイプ相性計算
│   ├── repositories/
│   │   ├── __init__.py
│   │   ├── pokemon_repository.py  # ポケモンデータアクセス
│   │   └── move_repository.py     # 技データアクセス
│   ├── routers/
│   │   ├── __init__.py
│   │   ├── damage.py             # ダメージ計算エンドポイント
│   │   └── pokemon.py            # ポケモン情報エンドポイント
│   └── utils/
│       ├── __init__.py
│       ├── data_loader.py        # データ読み込みユーティリティ
│       └── validators.py         # バリデーション関数
├── tests/
│   ├── test_calculators/
│   ├── test_repositories/
│   └── test_routers/
└── data/                         # 既存データディレクトリ活用
```

## 開発開始推奨事項

1. **Phase 1から段階的実装**: 全体を一度に変更せず、動作確認しながら進行
2. **既存テストとの照合**: 移植時は必ず既存の計算結果と比較検証
3. **文書化重視**: 複雑な補正ロジックは詳細にドキュメント化
4. **CI/CD構築**: GitHub Actions等で自動テスト・デプロイパイプライン整備

## 期待される成果物

この計画により、既存の高品質なダメージ計算ロジックを活用しつつ、以下の実用的なWeb APIが構築可能です:

- **高精度ダメージ計算**: 全補正要素を考慮した実戦レベル計算
- **柔軟な入力形式**: JSON APIによる簡単な利用
- **包括的機能**: 確定数計算、KO確率、技比較等
- **高パフォーマンス**: 100ms以下の高速レスポンス
- **拡張性**: 新機能追加や他システム連携が容易

開発完了後は、ポケモン対戦コミュニティで広く活用される実用的なツールとなることが期待されます。
