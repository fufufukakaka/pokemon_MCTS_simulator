"""
ReBeL (Recursive Belief-based Learning) for Pokemon Battles

不完全情報ゲームとしてのポケモンバトルに対するReBeL実装。

主要コンポーネント:
- PokemonBeliefState: 相手の型（技・持ち物・テラス等）に対する信念状態
- PublicGameState: 公開情報のみで構成されるゲーム状態
- PublicBeliefState: 公開状態 + 信念状態（PBS）
- CFRSubgameSolver: CFR によるサブゲーム解決
- ReBeLValueNetwork: PBS から価値を予測するネットワーク
- ReBeLBattle: ReBeL を使用したバトル AI

使用例:
    from src.rebel import ReBeLBattle, load_rebel_battle
    from src.hypothesis.pokemon_usage_database import PokemonUsageDatabase

    # 使用率データベースを読み込み
    usage_db = PokemonUsageDatabase.from_json("data/pokedb_usage/season_37_top150.json")

    # ReBeL バトルを作成
    battle = ReBeLBattle(usage_db)

    # または既存モデルをロード
    battle = load_rebel_battle(
        usage_db_path="data/pokedb_usage/season_37_top150.json",
        value_network_path="models/rebel/value_network.pt",
    )
"""

from .belief_state import (
    ObservationType,
    Observation,
    PokemonTypeHypothesis,
    PokemonBeliefState,
)
from .team_composition_belief import (
    TeamCompositionHypothesis,
    TeamCompositionBelief,
)
from .full_belief_state import (
    SampledWorld,
    FullBeliefState,
)
from .ev_template import (
    EVSpread,
    EVSpreadType,
    EV_TEMPLATES,
    estimate_ev_spread_type,
    get_ev_spread,
    get_ev_spread_from_pokemon_name,
)
from .public_state import (
    PublicPokemonState,
    PublicGameState,
    PublicBeliefState,
    instantiate_battle_from_hypothesis,
)
from .cfr_solver import (
    CFRConfig,
    CFRSubgameSolver,
    SimplifiedCFRSolver,
    LightweightCFRSolver,
    ReBeLSolver,
    default_value_estimator,
)
from .value_network import (
    PBSEncoder,
    ReBeLValueNetwork,
    ReBeLPolicyValueNetwork,
)
from .battle_interface import (
    ReBeLBattle,
    ReBeLMCTSAdapter,
    load_rebel_battle,
)
from .trainer import (
    TrainingExample,
    GameResult,
    TrainingConfig,
    ReBeLTrainer,
)

__all__ = [
    # Belief State
    "ObservationType",
    "Observation",
    "PokemonTypeHypothesis",
    "PokemonBeliefState",
    # Team Composition Belief
    "TeamCompositionHypothesis",
    "TeamCompositionBelief",
    # Full Belief State
    "SampledWorld",
    "FullBeliefState",
    # EV Template
    "EVSpread",
    "EVSpreadType",
    "EV_TEMPLATES",
    "estimate_ev_spread_type",
    "get_ev_spread",
    "get_ev_spread_from_pokemon_name",
    # Public State
    "PublicPokemonState",
    "PublicGameState",
    "PublicBeliefState",
    "instantiate_battle_from_hypothesis",
    # CFR Solver
    "CFRConfig",
    "CFRSubgameSolver",
    "SimplifiedCFRSolver",
    "LightweightCFRSolver",
    "ReBeLSolver",
    "default_value_estimator",
    # Value Network
    "PBSEncoder",
    "ReBeLValueNetwork",
    "ReBeLPolicyValueNetwork",
    # Battle Interface
    "ReBeLBattle",
    "ReBeLMCTSAdapter",
    "load_rebel_battle",
    # Trainer
    "TrainingExample",
    "GameResult",
    "TrainingConfig",
    "ReBeLTrainer",
]
