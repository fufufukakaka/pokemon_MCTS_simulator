"""
強化学習ループ

AlphaZeroスタイルのSelf-Play + 学習 + 評価のループを実行する。
"""

from __future__ import annotations

import json
import logging
import random
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import torch
from tqdm import tqdm

from src.hypothesis import (
    GameRecord,
    ItemPriorDatabase,
    SelfPlayGenerator,
    save_records_to_jsonl,
)
from src.hypothesis.item_belief_state import ItemBeliefState
from src.pokemon_battle_sim.battle import Battle
from src.pokemon_battle_sim.pokemon import Pokemon

from .evaluator import EvaluationResult, ModelEvaluator, load_model_for_evaluation
from .network import PolicyValueNetwork
from .nn_guided_mcts import NNGuidedMCTS, NNGuidedMCTSConfig
from .observation_encoder import ObservationEncoder
from .team_selector import (
    HybridTeamSelector,
    NNTeamSelector,
    RandomTeamSelector,
    TeamSelectorProtocol,
    TopNTeamSelector,
)
from .trainer import PolicyValueTrainer, TrainingConfig

logger = logging.getLogger(__name__)


@dataclass
class ReinforcementLoopConfig:
    """強化学習ループの設定"""

    # 全体設定
    num_generations: int = 10
    output_dir: str = "models/reinforcement"

    # Self-Play設定
    games_per_generation: int = 100
    max_turns: int = 100

    # MCTS設定
    mcts_simulations: int = 100
    n_hypotheses: int = 10

    # 学習設定
    training_epochs: int = 50
    batch_size: int = 64
    hidden_dim: int = 256
    num_res_blocks: int = 4
    learning_rate: float = 1e-3

    # 評価設定
    evaluation_games: int = 50
    win_rate_threshold: float = 0.55  # この勝率を超えたら新モデルを採用

    # チーム選出設定
    team_selection_mode: str = "random"  # "top_n", "random", "nn"
    team_selection_random_prob: float = 0.1  # NNモード時のランダム選出確率

    # デバイス
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class ReinforcementLoop:
    """
    強化学習ループ

    1. Self-Play: 現在のモデルで対戦データを生成
    2. Training: 生成したデータで新モデルを学習
    3. Evaluation: 新旧モデルを対戦させて評価
    4. 新モデルが強ければ採用、そうでなければ棄却
    5. 1に戻る
    """

    def __init__(
        self,
        prior_db: ItemPriorDatabase,
        trainer_data_path: str | Path,
        config: Optional[ReinforcementLoopConfig] = None,
    ):
        self.prior_db = prior_db
        self.trainer_data_path = Path(trainer_data_path)
        self.config = config or ReinforcementLoopConfig()

        # トレーナーデータ
        with open(trainer_data_path, "r", encoding="utf-8") as f:
            self.trainers = json.load(f)

        # 出力ディレクトリ
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 現在のベストモデル
        self.current_model: Optional[NNGuidedMCTS] = None
        self.current_encoder: Optional[ObservationEncoder] = None
        self.current_action_vocab: dict[str, int] = {}
        self.generation = 0

        # Team Selector初期化
        self.team_selector = self._init_team_selector()

        # 履歴
        self.history: list[dict] = []

    def _init_team_selector(self) -> TeamSelectorProtocol:
        """Team Selectorを初期化"""
        mode = self.config.team_selection_mode
        if mode == "top_n":
            return TopNTeamSelector()
        elif mode == "random":
            return RandomTeamSelector()
        elif mode == "nn":
            # NNモードはモデルがロードされてから設定
            return HybridTeamSelector(
                nn_selector=None,
                random_prob=self.config.team_selection_random_prob,
            )
        else:
            logger.warning(f"Unknown team_selection_mode: {mode}, using random")
            return RandomTeamSelector()

    def run(self):
        """強化学習ループを実行"""
        logger.info("=" * 60)
        logger.info("強化学習ループ開始")
        logger.info("=" * 60)
        logger.info(f"世代数: {self.config.num_generations}")
        logger.info(f"Self-Play試合数/世代: {self.config.games_per_generation}")
        logger.info(f"評価試合数: {self.config.evaluation_games}")
        logger.info(f"勝率閾値: {self.config.win_rate_threshold:.1%}")
        logger.info("=" * 60)

        for gen in range(self.config.num_generations):
            self.generation = gen
            logger.info(f"\n{'='*60}")
            logger.info(f"Generation {gen}")
            logger.info(f"{'='*60}")

            # 1. Self-Play
            dataset_path = self._run_selfplay()

            # 2. Training
            new_model_dir = self._train_model(dataset_path)

            # 3. Evaluation
            if self.current_model is not None:
                # 新旧モデルを対戦
                should_update = self._evaluate_and_decide(new_model_dir)
            else:
                # 最初の世代は無条件で採用
                should_update = True
                logger.info("最初の世代なので無条件で採用")

            # 4. モデル更新
            if should_update:
                self._update_best_model(new_model_dir)
                logger.info(f"✓ Generation {gen} のモデルを採用")
            else:
                logger.info(f"✗ Generation {gen} のモデルは棄却")

            # 履歴保存
            self._save_history()

        logger.info("\n" + "=" * 60)
        logger.info("強化学習ループ完了!")
        logger.info("=" * 60)

    def _run_selfplay(self) -> Path:
        """Self-Playでデータ生成"""
        logger.info("Self-Play開始...")

        dataset_path = self.output_dir / f"generation_{self.generation}" / "selfplay_data.jsonl"
        dataset_path.parent.mkdir(parents=True, exist_ok=True)

        if self.current_model is None:
            # 最初の世代: 純粋MCTSでSelf-Play
            logger.info("純粋MCTSでSelf-Play")
            records = self._selfplay_with_mcts()
        else:
            # 以降: NN誘導MCTSでSelf-Play
            logger.info("NN誘導MCTSでSelf-Play")
            records = self._selfplay_with_nn()

        save_records_to_jsonl(records, dataset_path)
        logger.info(f"Self-Play完了: {len(records)}試合, {sum(len(r.turns) for r in records)}ターン記録")

        return dataset_path

    def _selfplay_with_mcts(self) -> list[GameRecord]:
        """純粋MCTSでSelf-Play"""
        generator = SelfPlayGenerator(
            prior_db=self.prior_db,
            n_hypotheses=self.config.n_hypotheses,
            mcts_iterations=self.config.mcts_simulations,
        )

        records = []
        for game_idx in tqdm(range(self.config.games_per_generation), desc="Self-Play (MCTS)"):
            trainer0, trainer1 = random.sample(self.trainers, 2)
            full_team0 = trainer0["pokemons"][:6]
            full_team1 = trainer1["pokemons"][:6]

            # Team Selectorで選出
            team0 = self.team_selector.select(full_team0, full_team1, num_select=3)
            team1 = self.team_selector.select(full_team1, full_team0, num_select=3)

            record = generator.generate_game(
                trainer0_pokemons=team0,
                trainer1_pokemons=team1,
                trainer0_name=trainer0["name"],
                trainer1_name=trainer1["name"],
                game_id=f"gen{self.generation}_game{game_idx:05d}",
                max_turns=self.config.max_turns,
            )
            records.append(record)

        return records

    def _selfplay_with_nn(self) -> list[GameRecord]:
        """NN誘導MCTSでSelf-Play"""
        from src.hypothesis.selfplay import (
            FieldCondition,
            GameRecord,
            PokemonState,
            TurnRecord,
            action_id_to_str,
            policy_to_str_dict,
        )

        records = []

        for game_idx in tqdm(range(self.config.games_per_generation), desc="Self-Play (NN)"):
            trainer0, trainer1 = random.sample(self.trainers, 2)
            full_team0 = trainer0["pokemons"][:6]
            full_team1 = trainer1["pokemons"][:6]

            # Team Selectorで選出
            team0 = self.team_selector.select(full_team0, full_team1, num_select=3)
            team1 = self.team_selector.select(full_team1, full_team0, num_select=3)

            # バトル初期化
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

            # 信念状態
            belief_states = [
                ItemBeliefState([p.name for p in battle.selected[1]], self.prior_db),
                ItemBeliefState([p.name for p in battle.selected[0]], self.prior_db),
            ]

            # 記録
            game_record = GameRecord(
                game_id=f"gen{self.generation}_game{game_idx:05d}",
                player0_trainer=trainer0["name"],
                player1_trainer=trainer1["name"],
                player0_team=[p.name for p in battle.selected[0]],
                player1_team=[p.name for p in battle.selected[1]],
                winner=None,
                total_turns=0,
                turns=[],
            )

            turn = 0
            while battle.winner() is None and turn < self.config.max_turns:
                turn += 1

                commands = [Battle.SKIP, Battle.SKIP]
                policies = [{}, {}]
                values = [0.5, 0.5]

                for player in range(2):
                    available = battle.available_commands(player)
                    if not available:
                        continue

                    # NN誘導MCTSで探索
                    policy, value = self.current_model.search(
                        battle, player, belief_states[player]
                    )
                    policies[player] = policy
                    values[player] = value

                    if policy:
                        commands[player] = max(policy.items(), key=lambda x: x[1])[0]
                    else:
                        commands[player] = random.choice(available)

                # 記録
                for player in range(2):
                    if policies[player]:
                        turn_record = self._create_turn_record_for_selfplay(
                            battle, player, turn, policies[player], values[player],
                            commands[player], belief_states[player]
                        )
                        game_record.turns.append(turn_record)

                battle.proceed(commands=commands)

            game_record.winner = battle.winner()
            game_record.total_turns = turn

            # Value補正
            if game_record.winner is not None:
                alpha = 0.7
                for turn_record in game_record.turns:
                    outcome = 1.0 if turn_record.player == game_record.winner else 0.0
                    turn_record.value = alpha * turn_record.value + (1 - alpha) * outcome

            records.append(game_record)

        return records

    def _create_turn_record_for_selfplay(
        self,
        battle: Battle,
        player: int,
        turn: int,
        policy: dict[int, float],
        value: float,
        action_id: int,
        belief_state: ItemBeliefState,
    ) -> "TurnRecord":
        """Self-Play用にTurnRecordを作成"""
        from src.hypothesis.selfplay import (
            FieldCondition,
            PokemonState,
            TurnRecord,
            action_id_to_str,
            policy_to_str_dict,
        )

        def create_pokemon_state(pokemon) -> PokemonState:
            if pokemon is None:
                return PokemonState(
                    name="", hp=0, max_hp=0, hp_ratio=0.0, ailment="",
                    rank=[0] * 8, types=[], ability="", item="", moves=[],
                    terastallized=False, tera_type=""
                )
            max_hp = pokemon.status[0] if pokemon.status[0] > 0 else 1
            return PokemonState(
                name=pokemon.name,
                hp=pokemon.hp,
                max_hp=max_hp,
                hp_ratio=pokemon.hp / max_hp,
                ailment=getattr(pokemon, "ailment", ""),
                rank=list(getattr(pokemon, "rank", [0] * 8)),
                types=list(getattr(pokemon, "types", [])),
                ability=getattr(pokemon, "ability", ""),
                item=getattr(pokemon, "item", ""),
                moves=list(getattr(pokemon, "moves", [])),
                terastallized=getattr(pokemon, "terastal", False),
                tera_type=getattr(pokemon, "Ttype", ""),
            )

        opp = 1 - player
        cond = battle.condition

        field = FieldCondition(
            sunny=cond.get("sunny", 0),
            rainy=cond.get("rainy", 0),
            snow=cond.get("snow", 0),
            sandstorm=cond.get("sandstorm", 0),
            electric_field=cond.get("elecfield", 0),
            grass_field=cond.get("glassfield", 0),
            psychic_field=cond.get("psycofield", 0),
            mist_field=cond.get("mistfield", 0),
            gravity=cond.get("gravity", 0),
            trick_room=cond.get("trickroom", 0),
            reflector=list(cond.get("reflector", [0, 0])),
            light_screen=list(cond.get("lightwall", [0, 0])),
            tailwind=list(cond.get("oikaze", [0, 0])),
            safeguard=list(cond.get("safeguard", [0, 0])),
            mist=list(cond.get("whitemist", [0, 0])),
            spikes=list(cond.get("makibishi", [0, 0])),
            toxic_spikes=list(cond.get("dokubishi", [0, 0])),
            stealth_rock=list(cond.get("stealthrock", [0, 0])),
            sticky_web=list(cond.get("nebanet", [0, 0])),
        )

        opp_team = [p.name for p in battle.selected[opp]]
        item_beliefs = {name: belief_state.get_belief(name) for name in opp_team}

        my_bench = [create_pokemon_state(p) for p in battle.selected[player] if p != battle.pokemon[player]]
        opp_bench = [create_pokemon_state(p) for p in battle.selected[opp] if p != battle.pokemon[opp]]

        # Policyを文字列に変換
        policy_str = policy_to_str_dict(battle, player, policy)

        return TurnRecord(
            turn=turn,
            player=player,
            my_pokemon=create_pokemon_state(battle.pokemon[player]),
            my_bench=my_bench,
            opp_pokemon=create_pokemon_state(battle.pokemon[opp]),
            opp_bench=opp_bench,
            field=field,
            item_beliefs=item_beliefs,
            policy=policy_str,
            value=value,
            action=action_id_to_str(battle, player, action_id),
            action_id=action_id,
        )

    def _train_model(self, dataset_path: Path) -> Path:
        """新しいモデルを学習"""
        logger.info("モデル学習開始...")

        model_dir = self.output_dir / f"generation_{self.generation}" / "model"

        # 学習設定
        training_config = TrainingConfig(
            hidden_dim=self.config.hidden_dim,
            num_res_blocks=self.config.num_res_blocks,
            batch_size=self.config.batch_size,
            learning_rate=self.config.learning_rate,
            num_epochs=self.config.training_epochs,
            device=self.config.device,
        )

        trainer = PolicyValueTrainer(config=training_config)

        history = trainer.train(
            dataset_path=dataset_path,
            output_dir=model_dir,
            max_actions=100,
        )

        logger.info(f"学習完了: Final Val Loss = {history['val_loss'][-1]:.4f}")

        return model_dir

    def _evaluate_and_decide(self, new_model_dir: Path) -> bool:
        """新モデルを評価して採用するか決定"""
        logger.info("モデル評価開始...")

        # 新モデルを読み込み
        new_model = load_model_for_evaluation(
            new_model_dir, self.prior_db, self.config.device
        )

        # 評価器
        evaluator = ModelEvaluator(
            prior_db=self.prior_db,
            trainer_data_path=self.trainer_data_path,
            team_selector=self.team_selector,
        )

        # 対戦
        result = evaluator.evaluate_models(
            model_a=new_model,
            model_b=self.current_model,
            num_games=self.config.evaluation_games,
        )

        logger.info(f"評価結果: {result}")
        logger.info(f"勝率: {result.win_rate:.1%} (閾値: {self.config.win_rate_threshold:.1%})")

        # 履歴に追加
        self.history.append({
            "generation": self.generation,
            "wins": result.wins,
            "losses": result.losses,
            "draws": result.draws,
            "win_rate": result.win_rate,
            "adopted": result.win_rate >= self.config.win_rate_threshold,
        })

        return result.win_rate >= self.config.win_rate_threshold

    def _update_best_model(self, model_dir: Path):
        """ベストモデルを更新"""
        # 新モデルを読み込み
        self.current_model = load_model_for_evaluation(
            model_dir, self.prior_db, self.config.device
        )
        self.current_encoder = self.current_model.encoder
        self.current_action_vocab = self.current_model.action_vocab

        # ベストモデルとしてコピー
        best_dir = self.output_dir / "best_model"
        if best_dir.exists():
            shutil.rmtree(best_dir)
        shutil.copytree(model_dir, best_dir)

        logger.info(f"ベストモデルを更新: {best_dir}")

    def _save_history(self):
        """履歴を保存"""
        history_path = self.output_dir / "history.json"
        with open(history_path, "w") as f:
            json.dump(self.history, f, indent=2)
