"""
Policy-Value Network モジュール

ポケモン対戦の盤面からPolicy（行動確率）とValue（勝率）を予測する
ニューラルネットワークを提供する。

AlphaZeroスタイルの強化学習ループもサポート。
Team Selectionネットワークで最適なチーム選出も行う。
"""

from .dataset import SelfPlayDataset
from .evaluator import EvaluationResult, ModelEvaluator, load_model_for_evaluation
from .network import PolicyValueNetwork
from .nn_guided_mcts import NNGuidedMCTS, NNGuidedMCTSConfig
from .observation_encoder import ObservationEncoder
from .reinforcement_loop import ReinforcementLoop, ReinforcementLoopConfig
from .team_selection_dataset import (
    TeamSelectionDataset,
    TeamSelectionRecord,
    generate_selection_data_from_battles,
)
from .team_selection_encoder import TeamSelectionEncoder
from .team_selection_network import TeamSelectionNetwork, TeamSelectionNetworkConfig
from .team_selection_trainer import TeamSelectionTrainer, TeamSelectionTrainingConfig
from .team_selector import (
    HybridTeamSelector,
    NNTeamSelector,
    RandomTeamSelector,
    TopNTeamSelector,
    load_team_selector,
)
from .trainer import PolicyValueTrainer, TrainingConfig

__all__ = [
    # Core
    "ObservationEncoder",
    "PolicyValueNetwork",
    "SelfPlayDataset",
    # Training
    "PolicyValueTrainer",
    "TrainingConfig",
    # NN-guided MCTS
    "NNGuidedMCTS",
    "NNGuidedMCTSConfig",
    # Evaluation
    "ModelEvaluator",
    "EvaluationResult",
    "load_model_for_evaluation",
    # Reinforcement Loop
    "ReinforcementLoop",
    "ReinforcementLoopConfig",
    # Team Selection
    "TeamSelectionEncoder",
    "TeamSelectionNetwork",
    "TeamSelectionNetworkConfig",
    "TeamSelectionDataset",
    "TeamSelectionRecord",
    "TeamSelectionTrainer",
    "TeamSelectionTrainingConfig",
    "generate_selection_data_from_battles",
    # Team Selector
    "TopNTeamSelector",
    "RandomTeamSelector",
    "NNTeamSelector",
    "HybridTeamSelector",
    "load_team_selector",
]
