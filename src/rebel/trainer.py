"""
ReBeL Trainer

自己対戦によるデータ生成と Value Network の学習を行う。
"""

from __future__ import annotations

import json
import random
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from src.hypothesis.pokemon_usage_database import PokemonUsageDatabase
from src.pokemon_battle_sim.battle import Battle
from src.pokemon_battle_sim.pokemon import Pokemon

from .belief_state import Observation, ObservationType, PokemonBeliefState
from .cfr_solver import CFRConfig, ReBeLSolver
from .public_state import PublicBeliefState, PublicGameState
from .value_network import ReBeLValueNetwork


@dataclass
class TrainingExample:
    """学習用データの1サンプル"""

    # PBS のシリアライズ可能な表現
    public_state_dict: dict
    belief_summary: dict  # {pokemon_name: {item_dist, tera_dist}}

    # CFR で計算された戦略
    my_strategy: dict[int, float]
    opp_strategy: dict[int, float]

    # ターゲット値（終局後に設定）
    target_my_value: Optional[float] = None
    target_opp_value: Optional[float] = None

    # 実際に選択された行動
    action: Optional[int] = None


@dataclass
class GameResult:
    """1試合の結果"""

    game_id: str
    winner: Optional[int]
    total_turns: int
    examples: list[TrainingExample] = field(default_factory=list)


@dataclass
class TrainingConfig:
    """学習設定"""

    # データ生成
    games_per_iteration: int = 100
    max_turns: int = 100

    # CFR 設定
    cfr_iterations: int = 50
    cfr_world_samples: int = 20

    # 学習設定
    batch_size: int = 32
    learning_rate: float = 1e-4
    num_epochs: int = 10
    weight_decay: float = 1e-5

    # その他
    device: str = "cpu"
    save_interval: int = 10


class SelfPlayDataset(Dataset):
    """自己対戦データのデータセット"""

    def __init__(self, examples: list[TrainingExample]):
        # ターゲット値が設定されているサンプルのみ使用
        self.examples = [e for e in examples if e.target_my_value is not None]

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        ex = self.examples[idx]
        return {
            "public_state": ex.public_state_dict,
            "belief": ex.belief_summary,
            "my_strategy": ex.my_strategy,
            "target_my_value": ex.target_my_value,
            "target_opp_value": ex.target_opp_value,
        }


class ReBeLTrainer:
    """
    ReBeL 学習ループ

    1. 自己対戦でデータ生成（CFR で戦略計算）
    2. Value Network の学習
    3. 繰り返し
    """

    def __init__(
        self,
        usage_db: PokemonUsageDatabase,
        trainer_data: list[dict],
        config: Optional[TrainingConfig] = None,
        value_network: Optional[ReBeLValueNetwork] = None,
    ):
        """
        Args:
            usage_db: ポケモン使用率データベース
            trainer_data: トレーナーデータ（チーム情報）
            config: 学習設定
            value_network: 既存の Value Network（None の場合は新規作成）
        """
        self.usage_db = usage_db
        self.trainer_data = trainer_data
        self.config = config or TrainingConfig()

        # Value Network
        self.value_network = value_network or ReBeLValueNetwork()
        self.value_network.to(self.config.device)

        # Optimizer
        self.optimizer = optim.AdamW(
            self.value_network.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        # CFR ソルバー
        cfr_config = CFRConfig(
            num_iterations=self.config.cfr_iterations,
            num_world_samples=self.config.cfr_world_samples,
        )
        self.solver = ReBeLSolver(
            value_network=None,  # 最初はヒューリスティック
            cfr_config=cfr_config,
            use_simplified=True,
        )

        # 学習履歴
        self.training_history: list[dict] = []

    def generate_game(self, game_id: str) -> GameResult:
        """
        1試合を実行してデータを生成

        Args:
            game_id: ゲームID

        Returns:
            GameResult
        """
        # ランダムに2人のトレーナーを選択
        trainer0, trainer1 = random.sample(self.trainer_data, 2)

        # バトル初期化
        Pokemon.init()
        battle = Battle()
        battle.reset_game()

        # チーム設定（3体選出も含む）
        self._setup_team(battle, 0, trainer0)
        self._setup_team(battle, 1, trainer1)

        # ターン0で先頭のポケモンを場に出す
        battle.proceed(commands=[Battle.SKIP, Battle.SKIP])

        # 信念状態の初期化
        beliefs = [
            PokemonBeliefState(
                [p.name for p in battle.selected[1]],
                self.usage_db,
            ),
            PokemonBeliefState(
                [p.name for p in battle.selected[0]],
                self.usage_db,
            ),
        ]

        examples: list[TrainingExample] = []
        turn = 0

        while battle.winner() is None and turn < self.config.max_turns:
            turn += 1

            for player in [0, 1]:
                if battle.winner() is not None:
                    break

                # PBS 構築
                pbs = PublicBeliefState.from_battle(battle, player, beliefs[player])

                # CFR で戦略計算
                my_strategy, opp_strategy = self.solver.solve(pbs, battle)

                # 学習データを保存
                example = TrainingExample(
                    public_state_dict=self._serialize_public_state(pbs.public_state),
                    belief_summary=self._serialize_belief(beliefs[player]),
                    my_strategy=my_strategy,
                    opp_strategy=opp_strategy,
                )
                examples.append(example)

                # 行動選択
                action = self.solver.get_action(pbs, battle, explore=True, temperature=1.0)
                example.action = action

            # 相手の行動
            opp_pbs = PublicBeliefState.from_battle(battle, 1, beliefs[1])
            opp_action = self.solver.get_action(opp_pbs, battle, explore=True)

            # ターン実行
            commands = [
                examples[-2].action if len(examples) >= 2 else Battle.SKIP,
                opp_action,
            ]
            battle.proceed(commands=commands)

            # 観測更新（簡略化: 技使用のみ）
            self._update_beliefs_from_battle(battle, beliefs)

        # 終局結果でターゲット値を設定
        winner = battle.winner()
        for i, example in enumerate(examples):
            player = i % 2
            if winner is not None:
                example.target_my_value = 1.0 if winner == player else 0.0
                example.target_opp_value = 1.0 if winner != player else 0.0
            else:
                # 未決着（引き分け扱い）
                example.target_my_value = 0.5
                example.target_opp_value = 0.5

        return GameResult(
            game_id=game_id,
            winner=winner,
            total_turns=turn,
            examples=examples,
        )

    def _generate_game_with_pbs(
        self, game_id: str
    ) -> tuple[GameResult, list[tuple[PublicBeliefState, Optional[float], Optional[float]]]]:
        """
        1試合を実行してデータを生成（PBS オブジェクトも返す）

        Returns:
            (GameResult, list of (PBS, target_my_value, target_opp_value))
        """
        # ランダムに2人のトレーナーを選択
        trainer0, trainer1 = random.sample(self.trainer_data, 2)

        # バトル初期化
        Pokemon.init()
        battle = Battle()
        battle.reset_game()

        # チーム設定（3体選出も含む）
        self._setup_team(battle, 0, trainer0)
        self._setup_team(battle, 1, trainer1)

        # ターン0で先頭のポケモンを場に出す
        battle.proceed(commands=[Battle.SKIP, Battle.SKIP])

        # 信念状態の初期化
        beliefs = [
            PokemonBeliefState(
                [p.name for p in battle.selected[1]],
                self.usage_db,
            ),
            PokemonBeliefState(
                [p.name for p in battle.selected[0]],
                self.usage_db,
            ),
        ]

        examples: list[TrainingExample] = []
        pbs_records: list[tuple[PublicBeliefState, int]] = []  # (PBS, player)
        turn = 0
        last_actions = [Battle.SKIP, Battle.SKIP]

        while battle.winner() is None and turn < self.config.max_turns:
            turn += 1

            for player in [0, 1]:
                if battle.winner() is not None:
                    break

                # PBS 構築
                try:
                    pbs = PublicBeliefState.from_battle(battle, player, beliefs[player])
                except Exception:
                    continue

                # PBS を記録
                pbs_records.append((deepcopy(pbs), player))

                # CFR で戦略計算
                try:
                    my_strategy, opp_strategy = self.solver.solve(pbs, battle)
                except Exception:
                    my_strategy = {}
                    opp_strategy = {}

                # 学習データを保存
                example = TrainingExample(
                    public_state_dict=self._serialize_public_state(pbs.public_state),
                    belief_summary=self._serialize_belief(beliefs[player]),
                    my_strategy=my_strategy,
                    opp_strategy=opp_strategy,
                )
                examples.append(example)

                # 行動選択
                try:
                    action = self.solver.get_action(pbs, battle, explore=True, temperature=1.0)
                except Exception:
                    available = battle.available_commands(player)
                    action = random.choice(available) if available else Battle.SKIP

                example.action = action
                last_actions[player] = action

            # ターン実行
            try:
                battle.proceed(commands=last_actions)
            except Exception:
                break

            # 観測更新（簡略化）
            self._update_beliefs_from_battle(battle, beliefs)

        # 終局結果でターゲット値を設定
        winner = battle.winner()

        # PBS にターゲット値を付与
        pbs_with_targets: list[tuple[PublicBeliefState, Optional[float], Optional[float]]] = []
        for pbs, player in pbs_records:
            if winner is not None:
                target_my = 1.0 if winner == player else 0.0
                target_opp = 1.0 if winner != player else 0.0
            else:
                target_my = 0.5
                target_opp = 0.5
            pbs_with_targets.append((pbs, target_my, target_opp))

        # TrainingExample にもターゲット値を設定
        for i, example in enumerate(examples):
            player = i % 2
            if winner is not None:
                example.target_my_value = 1.0 if winner == player else 0.0
                example.target_opp_value = 1.0 if winner != player else 0.0
            else:
                example.target_my_value = 0.5
                example.target_opp_value = 0.5

        result = GameResult(
            game_id=game_id,
            winner=winner,
            total_turns=turn,
            examples=examples,
        )

        return result, pbs_with_targets

    def _setup_team(self, battle: Battle, player: int, trainer: dict) -> None:
        """トレーナーのチームを設定（3体選出）"""
        pokemons_data = trainer.get("pokemons", [])[:6]

        # ポケモンオブジェクトを作成
        team: list[Pokemon] = []
        for pokemon_data in pokemons_data:
            pokemon = Pokemon(pokemon_data.get("name", "ピカチュウ"))
            pokemon.item = pokemon_data.get("item", "")
            pokemon.ability = pokemon_data.get("ability", "")
            pokemon.moves = pokemon_data.get("moves", [])[:4]
            pokemon.Ttype = pokemon_data.get("tera_type", "ノーマル")

            # ステータス設定
            if "evs" in pokemon_data:
                pokemon.effort = pokemon_data["evs"]
            if "nature" in pokemon_data:
                pokemon.nature = pokemon_data["nature"]

            team.append(pokemon)

        # ランダムに3体選出
        import random
        num_select = min(3, len(team))
        if num_select == 0:
            # ポケモンがない場合はダミー
            dummy = Pokemon("ピカチュウ")
            battle.selected[player].append(dummy)
        else:
            selected_indices = random.sample(range(len(team)), num_select)
            for idx in selected_indices:
                battle.selected[player].append(team[idx])

    def _serialize_public_state(self, ps: PublicGameState) -> dict:
        """PublicGameState をシリアライズ"""
        return {
            "perspective": ps.perspective,
            "turn": ps.turn,
            "my_pokemon_name": ps.my_pokemon.name,
            "my_hp_ratio": ps.my_pokemon.hp_ratio,
            "opp_pokemon_name": ps.opp_pokemon.name,
            "opp_hp_ratio": ps.opp_pokemon.hp_ratio,
            "my_tera_available": ps.my_tera_available,
            "opp_tera_available": ps.opp_tera_available,
        }

    def _serialize_belief(self, belief: PokemonBeliefState) -> dict:
        """信念状態をシリアライズ"""
        result = {}
        for pokemon_name in belief.beliefs:
            result[pokemon_name] = {
                "item_dist": belief.get_item_distribution(pokemon_name),
                "tera_dist": belief.get_tera_distribution(pokemon_name),
            }
        return result

    def _update_beliefs_from_battle(
        self, battle: Battle, beliefs: list[PokemonBeliefState]
    ) -> None:
        """バトルの進行から信念を更新（簡略化版）"""
        # 実際の実装では、バトルログを解析して観測を抽出する
        # ここでは簡略化のため省略
        pass

    def train_iteration(self, iteration: int) -> dict:
        """
        1イテレーションの学習

        Args:
            iteration: イテレーション番号

        Returns:
            学習統計
        """
        # データ生成
        print(f"Iteration {iteration}: Generating games...")
        all_examples = []
        all_pbs_data: list[tuple[PublicBeliefState, float, float]] = []
        wins = {0: 0, 1: 0, None: 0}

        for i in range(self.config.games_per_iteration):
            game_id = f"iter{iteration}_game{i}"
            result, pbs_data = self._generate_game_with_pbs(game_id)
            all_examples.extend(result.examples)
            all_pbs_data.extend(pbs_data)
            wins[result.winner] = wins.get(result.winner, 0) + 1

        print(f"  Generated {len(all_examples)} examples from {self.config.games_per_iteration} games")
        print(f"  Wins: P0={wins[0]}, P1={wins[1]}, Draw={wins[None]}")

        # 有効なデータのみ抽出
        valid_data = [(pbs, my_v, opp_v) for pbs, my_v, opp_v in all_pbs_data if my_v is not None]
        if len(valid_data) == 0:
            print("  No valid examples, skipping training")
            return {"iteration": iteration, "examples": 0}

        # 学習
        print(f"  Training on {len(valid_data)} examples...")
        total_loss = 0.0
        num_batches = 0

        self.value_network.train()
        device = torch.device(self.config.device)
        self.value_network.to(device)

        for epoch in range(self.config.num_epochs):
            epoch_loss = 0.0
            random.shuffle(valid_data)

            # ミニバッチ処理
            for batch_start in range(0, len(valid_data), self.config.batch_size):
                batch = valid_data[batch_start:batch_start + self.config.batch_size]
                if len(batch) == 0:
                    continue

                pbs_batch = [item[0] for item in batch]
                target_my = torch.tensor([item[1] for item in batch], device=device, dtype=torch.float)
                target_opp = torch.tensor([item[2] for item in batch], device=device, dtype=torch.float)

                # Forward
                self.optimizer.zero_grad()
                try:
                    pred_my, pred_opp = self.value_network.forward_batch(pbs_batch)
                except Exception as e:
                    # エンコードエラーの場合はスキップ
                    print(f"    Batch encoding error: {e}")
                    continue

                # Loss (MSE)
                loss_my = F.mse_loss(pred_my, target_my)
                loss_opp = F.mse_loss(pred_opp, target_opp)
                loss = loss_my + loss_opp

                # Backward
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.value_network.parameters(), 1.0)
                self.optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

            if (epoch + 1) % 5 == 0:
                print(f"    Epoch {epoch + 1}: loss = {epoch_loss / max(1, num_batches):.4f}")

            total_loss += epoch_loss

        avg_loss = total_loss / max(num_batches, 1)
        print(f"  Average loss: {avg_loss:.4f}")

        stats = {
            "iteration": iteration,
            "examples": len(all_examples),
            "games": self.config.games_per_iteration,
            "avg_loss": avg_loss,
            "wins_p0": wins[0],
            "wins_p1": wins[1],
            "draws": wins[None],
        }
        self.training_history.append(stats)

        return stats

    def train(self, num_iterations: int, output_dir: str) -> None:
        """
        学習ループを実行

        Args:
            num_iterations: イテレーション数
            output_dir: 出力ディレクトリ
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        for iteration in range(1, num_iterations + 1):
            stats = self.train_iteration(iteration)

            # 定期保存
            if iteration % self.config.save_interval == 0:
                self.save(output_path / f"checkpoint_iter{iteration}")

        # 最終保存
        self.save(output_path / "final")

        # 学習履歴を保存
        with open(output_path / "training_history.json", "w") as f:
            json.dump(self.training_history, f, indent=2)

    def save(self, path: Path) -> None:
        """モデルを保存"""
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.value_network.state_dict(), path / "value_network.pt")
        torch.save(self.optimizer.state_dict(), path / "optimizer.pt")

        # エンコーダーの辞書を保存
        encoder_state = {
            "pokemon_to_id": self.value_network.encoder.pokemon_to_id,
            "move_to_id": self.value_network.encoder.move_to_id,
            "item_to_id": self.value_network.encoder.item_to_id,
        }
        with open(path / "encoder_vocab.json", "w", encoding="utf-8") as f:
            json.dump(encoder_state, f, ensure_ascii=False, indent=2)

    def load(self, path: Path) -> None:
        """モデルを読み込み"""
        self.value_network.load_state_dict(
            torch.load(path / "value_network.pt", map_location=self.config.device)
        )
        self.optimizer.load_state_dict(
            torch.load(path / "optimizer.pt", map_location=self.config.device)
        )

        # エンコーダーの辞書を読み込み
        with open(path / "encoder_vocab.json", "r", encoding="utf-8") as f:
            encoder_state = json.load(f)
        self.value_network.encoder.pokemon_to_id = encoder_state["pokemon_to_id"]
        self.value_network.encoder.move_to_id = encoder_state["move_to_id"]
        self.value_network.encoder.item_to_id = encoder_state["item_to_id"]

    def evaluate_against_baseline(
        self,
        num_games: int = 50,
        baseline_type: str = "random",
    ) -> dict:
        """
        ベースラインとの対戦評価

        Args:
            num_games: 評価試合数
            baseline_type: ベースラインの種類 ("random" or "cfr_only")

        Returns:
            評価結果
        """
        from src.hypothesis.hypothesis_mcts import HypothesisMCTS
        from src.hypothesis.pokemon_usage_database import ItemPriorDatabaseAdapter

        print(f"Evaluating against {baseline_type} baseline ({num_games} games)...")

        wins = {0: 0, 1: 0, None: 0}
        total_turns = 0

        for i in range(num_games):
            trainer0, trainer1 = random.sample(self.trainer_data, 2)

            # バトル初期化
            Pokemon.init()
            battle = Battle()
            battle.reset_game()

            self._setup_team(battle, 0, trainer0)
            self._setup_team(battle, 1, trainer1)
            battle.proceed(commands=[Battle.SKIP, Battle.SKIP])

            # ReBeL (Player 0) with trained network
            belief0 = PokemonBeliefState(
                [p.name for p in battle.selected[1]],
                self.usage_db,
            )

            # ベースライン (Player 1)
            if baseline_type == "random":
                # ランダム行動
                def get_baseline_action(battle, player):
                    available = battle.available_commands(player)
                    return random.choice(available) if available else Battle.SKIP
            else:
                # CFR only (no neural network)
                baseline_solver = ReBeLSolver(
                    value_network=None,
                    cfr_config=CFRConfig(num_iterations=20, num_world_samples=10),
                    use_simplified=True,
                )
                belief1 = PokemonBeliefState(
                    [p.name for p in battle.selected[0]],
                    self.usage_db,
                )

                def get_baseline_action(battle, player):
                    pbs = PublicBeliefState.from_battle(battle, player, belief1)
                    return baseline_solver.get_action(pbs, battle, explore=True)

            turn = 0
            while battle.winner() is None and turn < 100:
                turn += 1

                # ReBeL action
                try:
                    pbs = PublicBeliefState.from_battle(battle, 0, belief0)
                    action0 = self.solver.get_action(pbs, battle, explore=False)
                except Exception:
                    available = battle.available_commands(0)
                    action0 = random.choice(available) if available else Battle.SKIP

                # Baseline action
                try:
                    action1 = get_baseline_action(battle, 1)
                except Exception:
                    available = battle.available_commands(1)
                    action1 = random.choice(available) if available else Battle.SKIP

                try:
                    battle.proceed(commands=[action0, action1])
                except Exception:
                    break

            winner = battle.winner()
            wins[winner] = wins.get(winner, 0) + 1
            total_turns += turn

            if (i + 1) % 10 == 0:
                print(f"  Game {i + 1}: ReBeL={wins[0]}, Baseline={wins[1]}, Draw={wins[None]}")

        win_rate = wins[0] / num_games if num_games > 0 else 0.0
        avg_turns = total_turns / num_games if num_games > 0 else 0

        results = {
            "rebel_wins": wins[0],
            "baseline_wins": wins[1],
            "draws": wins[None],
            "win_rate": win_rate,
            "avg_turns": avg_turns,
            "baseline_type": baseline_type,
        }

        print(f"\nEvaluation Results:")
        print(f"  ReBeL wins:    {wins[0]} ({win_rate * 100:.1f}%)")
        print(f"  Baseline wins: {wins[1]} ({(1 - win_rate) * 100:.1f}%)")
        print(f"  Avg turns:     {avg_turns:.1f}")

        return results
