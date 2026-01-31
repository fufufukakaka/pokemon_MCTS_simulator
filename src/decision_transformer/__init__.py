"""
Pokemon Battle Decision Transformer

End-to-End Transformerモデルで選出からバトル行動まで統一的に扱う。
ReBeL + Selection BERT の代替として、より高速な推論を実現する。

Usage:
    # モデルのロードと推論
    from src.decision_transformer import (
        load_decision_transformer_ai,
        DecisionTransformerAI,
    )

    ai = load_decision_transformer_ai("models/decision_transformer/final")
    selection = ai.get_selection(my_team, opp_team)
    action = ai.get_battle_command(battle, player)
    analysis = ai.get_analysis(battle, player)

    # MCTS統合による数ターン先読み（安定択を選択）
    ai_with_mcts = load_decision_transformer_ai(
        "models/decision_transformer/final",
        use_mcts=True,
        mcts_simulations=200,
        mcts_max_depth=10,
    )
    action = ai_with_mcts.get_battle_command(battle, player)

    # 学習
    from src.decision_transformer import (
        PokemonBattleTransformerConfig,
        TrainingConfig,
        DecisionTransformerTrainer,
    )

    config = TrainingConfig(num_iterations=50, games_per_iteration=100)
    trainer = DecisionTransformerTrainer(config)
    trainer.train(trainer_data, output_dir="models/dt")
"""

from .ai_service import (
    DecisionTransformerAI,
    DecisionTransformerAIConfig,
    clear_dt_ai_cache,
    load_decision_transformer_ai,
)
from .dt_guided_mcts import (
    DTGuidedMCTS,
    DTGuidedMCTSConfig,
    DTGuidedMCTSNode,
)
from .config import (
    PokemonBattleTransformerConfig,
    TrainingConfig,
)
from .data_generator import (
    GeneratorConfig,
    TrajectoryGenerator,
    generate_random_trajectories,
)
from .dataset import (
    BattleTrajectory,
    BattleTrajectoryDataset,
    FieldState,
    PokemonState,
    TrajectoryPool,
    TurnRecord,
    TurnState,
    collate_fn,
    load_trajectories_from_jsonl,
    save_trajectories_to_jsonl,
)
from .embeddings import (
    PokemonBattleEmbeddings,
)
from .heads import (
    ActionHead,
    CombinedHead,
    SelectionHead,
    ValueHead,
)
from .model import (
    PokemonBattleTransformer,
    TransformerBlock,
    load_model,
)
from .tokenizer import (
    BattleSequenceTokenizer,
    BattleVocab,
)
from .trainer import (
    DecisionTransformerTrainer,
)

__all__ = [
    # Config
    "PokemonBattleTransformerConfig",
    "TrainingConfig",
    # Model
    "PokemonBattleTransformer",
    "TransformerBlock",
    "load_model",
    # Embeddings
    "PokemonBattleEmbeddings",
    # Heads
    "SelectionHead",
    "ActionHead",
    "ValueHead",
    "CombinedHead",
    # Tokenizer
    "BattleSequenceTokenizer",
    "BattleVocab",
    # Dataset
    "PokemonState",
    "FieldState",
    "TurnState",
    "TurnRecord",
    "BattleTrajectory",
    "TrajectoryPool",
    "BattleTrajectoryDataset",
    "collate_fn",
    "save_trajectories_to_jsonl",
    "load_trajectories_from_jsonl",
    # Data Generator
    "GeneratorConfig",
    "TrajectoryGenerator",
    "generate_random_trajectories",
    # Trainer
    "DecisionTransformerTrainer",
    # AI Service
    "DecisionTransformerAI",
    "DecisionTransformerAIConfig",
    "load_decision_transformer_ai",
    "clear_dt_ai_cache",
    # DT-Guided MCTS (数ターン先読み)
    "DTGuidedMCTS",
    "DTGuidedMCTSConfig",
    "DTGuidedMCTSNode",
]
