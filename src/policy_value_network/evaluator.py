"""
モデル評価器

新旧モデルを対戦させて、どちらが強いかを評価する。
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from tqdm import tqdm

from src.hypothesis.item_belief_state import ItemBeliefState
from src.hypothesis.item_prior_database import ItemPriorDatabase
from src.pokemon_battle_sim.battle import Battle
from src.pokemon_battle_sim.pokemon import Pokemon

from .network import PolicyValueNetwork
from .nn_guided_mcts import NNGuidedMCTS, NNGuidedMCTSConfig
from .observation_encoder import ObservationEncoder
from .team_selector import TeamSelectorProtocol, TopNTeamSelector


@dataclass
class EvaluationResult:
    """評価結果"""

    wins: int
    losses: int
    draws: int
    total_games: int

    @property
    def win_rate(self) -> float:
        if self.total_games == 0:
            return 0.0
        return self.wins / self.total_games

    @property
    def win_rate_without_draws(self) -> float:
        decided = self.wins + self.losses
        if decided == 0:
            return 0.5
        return self.wins / decided

    def __repr__(self) -> str:
        return (
            f"EvaluationResult(wins={self.wins}, losses={self.losses}, "
            f"draws={self.draws}, win_rate={self.win_rate:.1%})"
        )


class ModelEvaluator:
    """
    モデル評価器

    2つのモデル（またはMCTS）を対戦させて強さを比較する。
    """

    def __init__(
        self,
        prior_db: ItemPriorDatabase,
        trainer_data_path: str | Path,
        mcts_config: Optional[NNGuidedMCTSConfig] = None,
        team_selector: Optional[TeamSelectorProtocol] = None,
        fixed_party: Optional[list[dict]] = None,
    ):
        self.prior_db = prior_db
        self.mcts_config = mcts_config or NNGuidedMCTSConfig()
        self.team_selector = team_selector or TopNTeamSelector()
        self.fixed_party = fixed_party

        # トレーナーデータ読み込み
        with open(trainer_data_path, "r", encoding="utf-8") as f:
            self.trainers = json.load(f)

    def evaluate_models(
        self,
        model_a: NNGuidedMCTS,
        model_b: NNGuidedMCTS,
        num_games: int = 100,
        max_turns: int = 100,
    ) -> EvaluationResult:
        """
        2つのモデルを対戦させて評価

        Args:
            model_a: 評価対象モデル（新モデル）
            model_b: ベースラインモデル（旧モデル）
            num_games: 対戦数
            max_turns: 最大ターン数

        Returns:
            model_aの視点での勝敗結果
        """
        wins = 0
        losses = 0
        draws = 0

        for game_idx in tqdm(range(num_games), desc="Evaluating"):
            # 固定パーティモードかどうかで分岐
            if self.fixed_party:
                # Player 0: 固定パーティ
                full_team0 = self.fixed_party[:6]
                # Player 1: ランダムな対戦相手
                trainer1 = random.choice(self.trainers)
                full_team1 = trainer1["pokemons"][:6]
            else:
                # 従来の動作: 両方ランダム
                trainer0, trainer1 = random.sample(self.trainers, 2)
                full_team0 = trainer0["pokemons"][:6]
                full_team1 = trainer1["pokemons"][:6]

            # Team Selectorで選出
            team0 = self.team_selector.select(full_team0, full_team1, num_select=3)
            team1 = self.team_selector.select(full_team1, full_team0, num_select=3)

            # 固定パーティモード: model_aは常にPlayer 0（固定パーティ側）
            # 従来モード: 先手後手を交互に
            if self.fixed_party:
                # 固定パーティモードでは、model_aは常にPlayer 0
                winner = self._play_game(
                    model_a, model_b, team0, team1, max_turns
                )
                if winner == 0:
                    wins += 1
                elif winner == 1:
                    losses += 1
                else:
                    draws += 1
            elif game_idx % 2 == 0:
                winner = self._play_game(
                    model_a, model_b, team0, team1, max_turns
                )
                if winner == 0:
                    wins += 1
                elif winner == 1:
                    losses += 1
                else:
                    draws += 1
            else:
                winner = self._play_game(
                    model_b, model_a, team0, team1, max_turns
                )
                if winner == 1:
                    wins += 1
                elif winner == 0:
                    losses += 1
                else:
                    draws += 1

        return EvaluationResult(
            wins=wins,
            losses=losses,
            draws=draws,
            total_games=num_games,
        )

    def evaluate_vs_mcts(
        self,
        model: NNGuidedMCTS,
        mcts_iterations: int = 100,
        num_games: int = 100,
        max_turns: int = 100,
    ) -> EvaluationResult:
        """
        モデルと純粋MCTSを対戦させて評価

        Args:
            model: 評価対象のNNモデル
            mcts_iterations: 純粋MCTSのイテレーション数
            num_games: 対戦数
            max_turns: 最大ターン数

        Returns:
            modelの視点での勝敗結果
        """
        from src.hypothesis.hypothesis_mcts import HypothesisMCTS

        pure_mcts = HypothesisMCTS(
            prior_db=self.prior_db,
            n_hypotheses=self.mcts_config.n_hypotheses,
            mcts_iterations=mcts_iterations,
        )

        wins = 0
        losses = 0
        draws = 0

        for game_idx in tqdm(range(num_games), desc="Evaluating vs MCTS"):
            trainer0, trainer1 = random.sample(self.trainers, 2)
            full_team0 = trainer0["pokemons"][:6]
            full_team1 = trainer1["pokemons"][:6]

            # Team Selectorで選出
            team0 = self.team_selector.select(full_team0, full_team1, num_select=3)
            team1 = self.team_selector.select(full_team1, full_team0, num_select=3)

            if game_idx % 2 == 0:
                winner = self._play_game_vs_mcts(
                    model, pure_mcts, team0, team1, max_turns, nn_player=0
                )
                if winner == 0:
                    wins += 1
                elif winner == 1:
                    losses += 1
                else:
                    draws += 1
            else:
                winner = self._play_game_vs_mcts(
                    model, pure_mcts, team0, team1, max_turns, nn_player=1
                )
                if winner == 1:
                    wins += 1
                elif winner == 0:
                    losses += 1
                else:
                    draws += 1

        return EvaluationResult(
            wins=wins,
            losses=losses,
            draws=draws,
            total_games=num_games,
        )

    def _play_game(
        self,
        model_0: NNGuidedMCTS,
        model_1: NNGuidedMCTS,
        team0: list[dict],
        team1: list[dict],
        max_turns: int,
    ) -> Optional[int]:
        """1試合を実行（NNモデル vs NNモデル）"""
        battle = Battle(seed=random.randint(0, 2**31))
        battle.reset_game()

        # ポケモン設定
        for i, team in enumerate([team0, team1]):
            for p_data in team:
                p = Pokemon(p_data["name"])
                p.item = p_data.get("item", "")
                p.nature = p_data.get("nature", "まじめ")
                p.ability = p_data.get("ability", "")
                p.Ttype = p_data.get("Ttype", "")
                p.moves = p_data.get("moves", [])
                p.effort = p_data.get("effort", [0] * 6)
                battle.selected[i].append(p)

        battle.pokemon = [battle.selected[0][0], battle.selected[1][0]]

        # 信念状態
        belief_states = [
            ItemBeliefState([p.name for p in battle.selected[1]], self.prior_db),
            ItemBeliefState([p.name for p in battle.selected[0]], self.prior_db),
        ]

        models = [model_0, model_1]

        turn = 0
        while battle.winner() is None and turn < max_turns:
            turn += 1

            commands = [Battle.SKIP, Battle.SKIP]
            for player in range(2):
                available = battle.available_commands(player)
                if available:
                    commands[player] = models[player].get_best_action(
                        battle, player, belief_states[player]
                    )

            battle.proceed(commands=commands)

        return battle.winner()

    def _play_game_vs_mcts(
        self,
        nn_model: NNGuidedMCTS,
        pure_mcts,  # HypothesisMCTS
        team0: list[dict],
        team1: list[dict],
        max_turns: int,
        nn_player: int,
    ) -> Optional[int]:
        """1試合を実行（NNモデル vs 純粋MCTS）"""
        battle = Battle(seed=random.randint(0, 2**31))
        battle.reset_game()

        for i, team in enumerate([team0, team1]):
            for p_data in team:
                p = Pokemon(p_data["name"])
                p.item = p_data.get("item", "")
                p.nature = p_data.get("nature", "まじめ")
                p.ability = p_data.get("ability", "")
                p.Ttype = p_data.get("Ttype", "")
                p.moves = p_data.get("moves", [])
                p.effort = p_data.get("effort", [0] * 6)
                battle.selected[i].append(p)

        battle.pokemon = [battle.selected[0][0], battle.selected[1][0]]

        belief_states = [
            ItemBeliefState([p.name for p in battle.selected[1]], self.prior_db),
            ItemBeliefState([p.name for p in battle.selected[0]], self.prior_db),
        ]

        turn = 0
        while battle.winner() is None and turn < max_turns:
            turn += 1

            commands = [Battle.SKIP, Battle.SKIP]
            for player in range(2):
                available = battle.available_commands(player)
                if not available:
                    continue

                if player == nn_player:
                    commands[player] = nn_model.get_best_action(
                        battle, player, belief_states[player]
                    )
                else:
                    pv = pure_mcts.search(
                        battle, player, belief_states[player]
                    )
                    if pv.policy:
                        commands[player] = max(pv.policy.items(), key=lambda x: x[1])[0]
                    else:
                        commands[player] = random.choice(available)

            battle.proceed(commands=commands)

        return battle.winner()


def load_model_for_evaluation(
    model_dir: str | Path,
    prior_db: ItemPriorDatabase,
    device: str = "cpu",
) -> NNGuidedMCTS:
    """評価用にモデルを読み込み"""
    model_dir = Path(model_dir)

    # エンコーダー
    encoder = ObservationEncoder.load(model_dir / "encoder.json")

    # 行動辞書
    with open(model_dir / "action_vocab.json", "r", encoding="utf-8") as f:
        action_vocab = json.load(f)

    # 設定
    with open(model_dir / "config.json", "r") as f:
        config = json.load(f)

    # モデル
    # configにnum_actionsが保存されていればそれを使う（トレーニング時のmax_actionsに対応）
    # なければ互換性のためaction_vocabから計算
    input_dim = config.get("input_dim", encoder.get_flat_dim())
    num_actions = config.get("num_actions", len(action_vocab) + 1)

    model = PolicyValueNetwork(
        input_dim=input_dim,
        hidden_dim=config["hidden_dim"],
        num_res_blocks=config.get("num_res_blocks", 4),
        num_actions=num_actions,
        dropout=0.0,  # 評価時はドロップアウトなし
    )

    checkpoint = torch.load(model_dir / "best_model.pt", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # MCTS設定
    mcts_config = NNGuidedMCTSConfig(device=device)

    return NNGuidedMCTS(
        model=model,
        encoder=encoder,
        prior_db=prior_db,
        action_vocab=action_vocab,
        config=mcts_config,
    )


import torch
