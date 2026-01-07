"""
Trajectory Data Generator

自己対戦によるバトル軌跡データの生成。
AlphaZero スタイルの自己対戦ループをサポート。
"""

from __future__ import annotations

import logging
import random
import uuid
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

import torch

from src.pokemon_battle_sim.battle import Battle
from src.pokemon_battle_sim.pokemon import Pokemon

from .dataset import (
    BattleTrajectory,
    FieldState,
    ObservationTracker,
    ObservedPokemonState,
    PokemonState,
    TurnRecord,
    TurnState,
)

if TYPE_CHECKING:
    from .model import PokemonBattleTransformer
    from .tokenizer import BattleSequenceTokenizer

logger = logging.getLogger(__name__)


def _parallel_game_worker(args: tuple) -> dict | None:
    """
    並列ゲーム生成のワーカー関数（モジュールレベルで定義してpickle可能にする）

    Args:
        args: (game_id, trainer_data, config_dict) のタプル

    Returns:
        BattleTrajectory の辞書表現、または失敗時は None
    """
    game_id, trainer_data, config_dict = args
    try:
        # 統計データパスを取得
        usage_data_path = config_dict.get("usage_data_path") if config_dict else None
        Pokemon.init(usage_data_path=usage_data_path, verbose=False)

        config = GeneratorConfig(**config_dict) if config_dict else GeneratorConfig()

        # モデルとトークナイザをロード（MCTS使用時）
        model = None
        tokenizer = None
        if config.use_mcts and config.model_checkpoint_path:
            from pathlib import Path

            from .model import load_model
            from .tokenizer import BattleSequenceTokenizer

            checkpoint_path = Path(config.model_checkpoint_path)
            if checkpoint_path.exists():
                # 並列ワーカーでは GPU を使わない（メモリ共有の問題を避ける）
                worker_device = "cpu"
                model = load_model(
                    checkpoint_path=str(checkpoint_path),
                    device=worker_device,
                )

                tokenizer_path = checkpoint_path / "tokenizer"
                if tokenizer_path.exists():
                    tokenizer = BattleSequenceTokenizer.load(tokenizer_path, model.config)
                else:
                    tokenizer = BattleSequenceTokenizer(model.config)

                # ワーカー用にdeviceをCPUに設定
                config.device = worker_device

                logger.debug(f"Worker {game_id}: Loaded model from {checkpoint_path}")
            else:
                logger.warning(f"Worker {game_id}: Checkpoint not found at {checkpoint_path}, using RandomPolicy")

        gen = TrajectoryGenerator(
            trainer_data=trainer_data,
            config=config,
            model=model,
            tokenizer=tokenizer,
        )
        traj = gen.generate_trajectory(game_id=game_id)
        return traj.to_dict()
    except Exception as e:
        logger.error(f"Worker failed for {game_id}: {e}")
        import traceback
        traceback.print_exc()
        return None


@dataclass
class GeneratorConfig:
    """データ生成の設定"""

    # 探索設定
    epsilon: float = 0.1  # ε-greedy の ε
    temperature: float = 1.0  # サンプリング温度

    # ゲーム設定
    max_turns: int = 100  # 最大ターン数

    # 並列化
    num_workers: int = 1

    # 統計データ
    usage_data_path: str | None = None  # 統計データのパス（None なら Pokemon.init のデフォルト）

    # MCTS設定 (Expert Iteration)
    use_mcts: bool = False  # MCTSを使用するか
    mcts_simulations: int = 100  # MCTSシミュレーション回数
    mcts_max_depth: int = 6  # 最大探索深度
    mcts_c_puct: float = 1.5  # 探索バランスパラメータ
    device: str = "cpu"  # デバイス

    # 並列ワーカー用: モデルチェックポイントパス
    # 並列処理でMCTSを使う場合、各ワーカーがこのパスからモデルをロードする
    model_checkpoint_path: str | None = None


def _pokemon_to_state(pokemon: Pokemon | None) -> PokemonState:
    """Pokemon オブジェクトを PokemonState に変換"""
    if pokemon is None:
        return PokemonState(name="", hp_ratio=0.0)

    max_hp = pokemon.status[0] if pokemon.status else 1
    hp_ratio = pokemon.hp / max_hp if max_hp > 0 else 0.0

    # pokemon.condition から状態変化を取得
    condition = getattr(pokemon, "condition", {})

    # ステータス [H, A, B, C, D, S]
    status = list(pokemon.status) if pokemon.status else [0] * 6

    return PokemonState(
        name=pokemon.name,
        hp_ratio=hp_ratio,
        ailment=pokemon.ailment or "",
        rank=list(pokemon.rank[:8]) if pokemon.rank else [0] * 8,
        types=list(pokemon.types) if pokemon.types else [],
        terastallized=pokemon.terastal if hasattr(pokemon, "terastal") else False,
        tera_type=pokemon.Ttype if hasattr(pokemon, "Ttype") else "",
        status=status,
        item=pokemon.item or "",
        ability=pokemon.ability or "",
        moves=list(pokemon.moves) if pokemon.moves else [],
        # 状態変化
        confusion=condition.get("confusion", 0),
        critical_rank=condition.get("critical", 0),
        aquaring=bool(condition.get("aquaring", 0)),
        healblock=condition.get("healblock", 0),
        magnetrise=condition.get("magnetrise", 0),
        noroi=bool(condition.get("noroi", 0)),
        horobi=condition.get("horobi", 0),
        yadorigi=bool(condition.get("yadorigi", 0)),
        encore=condition.get("encore", 0),
        chohatsu=condition.get("chohatsu", 0),
        change_block=bool(condition.get("change_block", 0)),
        meromero=bool(condition.get("meromero", 0)),
        bind=int(condition.get("bind", 0)),  # bind は小数の場合があるので int に
        sub_hp=getattr(pokemon, "sub_hp", 0),
        fixed_move=getattr(pokemon, "fixed_move", "") or "",
        inaccessible=getattr(pokemon, "inaccessible", 0),
    )


def _pokemon_to_observed_state(
    pokemon: Pokemon | None,
    tracker: ObservationTracker,
) -> ObservedPokemonState:
    """
    Pokemon オブジェクトを ObservedPokemonState に変換

    tracker に記録された観測情報のみを使用し、未観測情報は隠す。
    """
    if pokemon is None:
        return ObservedPokemonState(name="", hp_ratio=0.0, is_revealed=False)

    name = pokemon.name
    obs = tracker.get_or_create(name)

    # HP、状態異常、ランクは常に観測可能
    max_hp = pokemon.status[0] if pokemon.status else 1
    hp_ratio = pokemon.hp / max_hp if max_hp > 0 else 0.0

    # pokemon.condition から状態変化を取得
    condition = getattr(pokemon, "condition", {})

    # 現在のタイプ（テラスタル時は変化）
    current_types = list(pokemon.types) if pokemon.types else []

    return ObservedPokemonState(
        name=name,
        hp_ratio=hp_ratio,
        ailment=pokemon.ailment or "",
        rank=list(pokemon.rank[:8]) if pokemon.rank else [0] * 8,
        types=current_types,
        terastallized=obs.terastallized,  # tracker から
        tera_type=obs.tera_type,  # tracker から
        # 観測済み情報（tracker から）
        revealed_moves=list(obs.revealed_moves),
        revealed_item=obs.revealed_item,
        revealed_ability=obs.revealed_ability,
        # 状態変化
        confusion=condition.get("confusion", 0),
        critical_rank=condition.get("critical", 0),
        aquaring=bool(condition.get("aquaring", 0)),
        healblock=condition.get("healblock", 0),
        magnetrise=condition.get("magnetrise", 0),
        noroi=bool(condition.get("noroi", 0)),
        horobi=condition.get("horobi", 0),
        yadorigi=bool(condition.get("yadorigi", 0)),
        encore=condition.get("encore", 0),
        chohatsu=condition.get("chohatsu", 0),
        change_block=bool(condition.get("change_block", 0)),
        meromero=bool(condition.get("meromero", 0)),
        bind=int(condition.get("bind", 0)),
        sub_hp=getattr(pokemon, "sub_hp", 0),
        inaccessible=getattr(pokemon, "inaccessible", 0),
        is_revealed=obs.is_revealed,
    )


def _battle_to_field_state(battle: Battle) -> FieldState:
    """Battle の condition を FieldState に変換"""
    condition = battle.condition

    # 天候
    weather = ""
    weather_turns = 0
    if condition.get("sunny", 0) > 0:
        weather = "sunny"
        weather_turns = condition["sunny"]
    elif condition.get("rainy", 0) > 0:
        weather = "rainy"
        weather_turns = condition["rainy"]
    elif condition.get("snow", 0) > 0:
        weather = "snow"
        weather_turns = condition["snow"]
    elif condition.get("sandstorm", 0) > 0:
        weather = "sandstorm"
        weather_turns = condition["sandstorm"]

    # フィールド
    terrain = ""
    terrain_turns = 0
    if condition.get("elecfield", 0) > 0:
        terrain = "electric"
        terrain_turns = condition["elecfield"]
    elif condition.get("glassfield", 0) > 0:
        terrain = "grass"
        terrain_turns = condition["glassfield"]
    elif condition.get("psycofield", 0) > 0:
        terrain = "psychic"
        terrain_turns = condition["psycofield"]
    elif condition.get("mistfield", 0) > 0:
        terrain = "mist"
        terrain_turns = condition["mistfield"]

    return FieldState(
        weather=weather,
        weather_turns=weather_turns,
        terrain=terrain,
        terrain_turns=terrain_turns,
        trick_room=condition.get("trickroom", 0),
        gravity=condition.get("gravity", 0),
        reflector=tuple(condition.get("reflector", [0, 0])),
        light_screen=tuple(condition.get("lightwall", [0, 0])),
        tailwind=tuple(condition.get("oikaze", [0, 0])),
        stealth_rock=tuple(x > 0 for x in condition.get("stealthrock", [0, 0])),
        spikes=tuple(condition.get("makibishi", [0, 0])),
        toxic_spikes=tuple(condition.get("dokubishi", [0, 0])),
        sticky_web=tuple(x > 0 for x in condition.get("nebanet", [0, 0])),
        # 追加の盤面状態
        safeguard=tuple(condition.get("safeguard", [0, 0])),
        white_mist=tuple(condition.get("whitemist", [0, 0])),
        wish=tuple(int(x) for x in condition.get("wish", [0, 0])),  # wish は小数の場合があるので int に
    )


def _get_turn_state(
    battle: Battle,
    player: int,
    opp_tracker: ObservationTracker,
) -> TurnState:
    """現在のターン状態を取得（不完全情報を考慮）

    Args:
        battle: バトル状態
        player: 視点のプレイヤー番号
        opp_tracker: 相手の観測トラッカー

    Returns:
        TurnState（相手情報は観測済み情報のみ）
    """
    opponent = 1 - player

    # 自分のポケモン（完全情報）
    my_active = _pokemon_to_state(battle.pokemon[player])
    my_bench = []
    for p in battle.selected[player]:
        if p is not None and p is not battle.pokemon[player]:
            my_bench.append(_pokemon_to_state(p))

    # 相手のポケモン（観測済み情報のみ）
    opp_active = _pokemon_to_observed_state(battle.pokemon[opponent], opp_tracker)
    opp_bench = []
    for p in battle.selected[opponent]:
        if p is not None and p is not battle.pokemon[opponent]:
            # 場に出たことがあるポケモンのみベンチに表示
            obs = opp_tracker.get_or_create(p.name)
            if obs.is_revealed:
                opp_bench.append(_pokemon_to_observed_state(p, opp_tracker))

    # 未公開の相手ポケモン数
    opp_unrevealed = 3 - len(opp_tracker.revealed_selection)
    opp_unrevealed = max(0, opp_unrevealed)

    # フィールド
    field = _battle_to_field_state(battle)

    # 利用可能な行動
    available_actions = battle.available_commands(player, phase="battle")

    return TurnState(
        turn=battle.turn,
        player=player,
        my_active=my_active,
        my_bench=my_bench,
        opp_active=opp_active,
        opp_bench=opp_bench,
        opp_unrevealed_count=opp_unrevealed,
        field_state=field,
        available_actions=available_actions,
    )


class RandomPolicy:
    """ランダムポリシー（ベースライン）"""

    def get_selection(self, team_size: int = 6) -> list[int]:
        """ランダムに3匹選出"""
        indices = list(range(team_size))
        random.shuffle(indices)
        return indices[:3]

    def get_action(self, available_actions: list[int]) -> int:
        """ランダムに行動選択"""
        if not available_actions:
            return Battle.SKIP
        return random.choice(available_actions)


class EpsilonGreedyPolicy:
    """ε-greedy ポリシー"""

    def __init__(
        self,
        model: "PokemonBattleTransformer",
        tokenizer: "BattleSequenceTokenizer",
        epsilon: float = 0.1,
        temperature: float = 1.0,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.epsilon = epsilon
        self.temperature = temperature
        self.random_policy = RandomPolicy()

    def get_selection(
        self,
        my_team: list[str],
        opp_team: list[str],
    ) -> list[int]:
        """選出を決定"""
        if random.random() < self.epsilon:
            return self.random_policy.get_selection(len(my_team))

        selected, _ = self.model.get_selection(
            my_team=my_team,
            opp_team=opp_team,
            tokenizer=self.tokenizer,
            target_return=1.0,
            deterministic=False,
            temperature=self.temperature,
        )
        return selected

    def get_action(
        self,
        context: dict[str, torch.Tensor],
        action_mask: torch.Tensor,
        available_actions: list[int],
    ) -> int:
        """行動を決定"""
        if random.random() < self.epsilon:
            return self.random_policy.get_action(available_actions)

        action_id, _ = self.model.get_action(
            context=context,
            action_mask=action_mask,
            deterministic=False,
            temperature=self.temperature,
        )
        return action_id


class MCTSPolicy:
    """MCTS を使ったポリシー（Expert Iteration 用）"""

    def __init__(
        self,
        model: "PokemonBattleTransformer",
        tokenizer: "BattleSequenceTokenizer",
        epsilon: float = 0.1,
        temperature: float = 1.0,
        mcts_simulations: int = 100,
        mcts_max_depth: int = 6,
        mcts_c_puct: float = 1.5,
        device: str = "cpu",
    ):
        from .dt_guided_mcts import DTGuidedMCTS, DTGuidedMCTSConfig

        self.model = model
        self.tokenizer = tokenizer
        self.epsilon = epsilon
        self.temperature = temperature
        self.random_policy = RandomPolicy()

        # MCTSを初期化
        mcts_config = DTGuidedMCTSConfig(
            n_simulations=mcts_simulations,
            max_depth=mcts_max_depth,
            c_puct=mcts_c_puct,
            temperature=temperature,
            device=device,
        )
        self.mcts = DTGuidedMCTS(
            model=model,
            tokenizer=tokenizer,
            config=mcts_config,
        )

    def get_selection(
        self,
        my_team: list[str],
        opp_team: list[str],
    ) -> list[int]:
        """選出を決定（MCTSは選出には使わない）"""
        if random.random() < self.epsilon:
            return self.random_policy.get_selection(len(my_team))

        selected, _ = self.model.get_selection(
            my_team=my_team,
            opp_team=opp_team,
            tokenizer=self.tokenizer,
            target_return=1.0,
            deterministic=False,
            temperature=self.temperature,
        )
        return selected

    def get_action(
        self,
        battle: "Battle",
        player: int,
        available_actions: list[int],
    ) -> int:
        """MCTSを使って行動を決定"""
        if random.random() < self.epsilon:
            return self.random_policy.get_action(available_actions)

        # MCTS探索
        policy, _, _ = self.mcts.search(
            battle=battle,
            player=player,
            target_return=1.0,
        )

        if not policy:
            return self.random_policy.get_action(available_actions)

        # 温度に基づいて行動選択
        if self.temperature == 0:
            action_id = max(policy.items(), key=lambda x: x[1])[0]
        else:
            actions = list(policy.keys())
            probs = [p ** (1 / self.temperature) for p in policy.values()]
            prob_sum = sum(probs)
            probs = [p / prob_sum for p in probs]
            action_id = random.choices(actions, weights=probs)[0]

        return action_id

    def reset(self):
        """MCTSのコンテキストをリセット"""
        self.mcts.reset()


class TrajectoryGenerator:
    """
    バトル軌跡の生成器

    自己対戦によるデータ生成をサポート。
    """

    def __init__(
        self,
        trainer_data: list[dict[str, Any]],
        config: GeneratorConfig | None = None,
        model: Optional["PokemonBattleTransformer"] = None,
        tokenizer: Optional["BattleSequenceTokenizer"] = None,
    ):
        """
        Args:
            trainer_data: トレーナーデータ（パーティ情報を含む）
            config: 生成設定
            model: 使用するモデル（None ならランダムポリシー）
            tokenizer: トークナイザ（model 使用時に必要）
        """
        self.trainer_data = trainer_data
        self.config = config or GeneratorConfig()
        self.model = model
        self.tokenizer = tokenizer
        self.use_mcts = self.config.use_mcts and model is not None

        # Pokemon データの初期化
        Pokemon.init(usage_data_path=self.config.usage_data_path)

        # ポリシーの設定
        if model is not None and tokenizer is not None:
            if self.config.use_mcts:
                # MCTSベースのポリシー（Expert Iteration）
                self.policy = MCTSPolicy(
                    model=model,
                    tokenizer=tokenizer,
                    epsilon=self.config.epsilon,
                    temperature=self.config.temperature,
                    mcts_simulations=self.config.mcts_simulations,
                    mcts_max_depth=self.config.mcts_max_depth,
                    mcts_c_puct=self.config.mcts_c_puct,
                    device=self.config.device,
                )
                logger.info(
                    f"Using MCTS policy: {self.config.mcts_simulations} sims, "
                    f"depth={self.config.mcts_max_depth}"
                )
            else:
                self.policy = EpsilonGreedyPolicy(
                    model=model,
                    tokenizer=tokenizer,
                    epsilon=self.config.epsilon,
                    temperature=self.config.temperature,
                )
        else:
            self.policy = RandomPolicy()

    def _create_pokemon(self, data: dict[str, Any]) -> Pokemon:
        """データから Pokemon を作成"""
        pokemon = Pokemon(data["name"])
        pokemon.item = data.get("item", "")
        pokemon.ability = data.get("ability", "")
        pokemon.Ttype = data.get("Ttype", data.get("tera_type", ""))
        pokemon.nature = data.get("nature", "")
        pokemon.moves = data.get("moves", [])

        effort = data.get("effort", [0, 0, 0, 0, 0, 0])
        if isinstance(effort, list) and len(effort) == 6:
            pokemon.effort = effort

        pokemon.update_status()
        return pokemon

    def generate_trajectory(self, game_id: str | None = None) -> BattleTrajectory:
        """
        1試合の軌跡を生成

        Args:
            game_id: ゲームID（None なら自動生成）

        Returns:
            BattleTrajectory
        """
        if game_id is None:
            game_id = str(uuid.uuid4())[:8]

        # ランダムに2つのトレーナーを選択
        trainer0, trainer1 = random.sample(self.trainer_data, 2)
        team0_data = trainer0["pokemons"]
        team1_data = trainer1["pokemons"]

        # チーム名リスト
        team0_names = [p["name"] for p in team0_data]
        team1_names = [p["name"] for p in team1_data]

        # 選出を決定
        if isinstance(self.policy, (EpsilonGreedyPolicy, MCTSPolicy)):
            selection0 = self.policy.get_selection(team0_names, team1_names)
            selection1 = self.policy.get_selection(team1_names, team0_names)
        else:
            selection0 = self.policy.get_selection()
            selection1 = self.policy.get_selection()

        # Battle を初期化
        battle = Battle()
        battle.reset_game()

        # ポケモンを設定
        battle.selected[0] = [
            self._create_pokemon(team0_data[i]) for i in selection0
        ]
        battle.selected[1] = [
            self._create_pokemon(team1_data[i]) for i in selection1
        ]

        # 観測トラッカーを初期化
        # player0 は player1 の情報を追跡、player1 は player0 の情報を追跡
        tracker0 = ObservationTracker()  # player0 視点（player1 を追跡）
        tracker1 = ObservationTracker()  # player1 視点（player0 を追跡）

        # バトル開始
        battle.proceed(commands=[None, None])

        # 初期の先発ポケモンを公開情報として記録
        if battle.pokemon[0]:
            tracker1.reveal_pokemon(battle.pokemon[0].name)
            # 先発ポケモンの特性発動を検出（いかく、ひでり、あめふらし等）
            self._detect_ability_reveal(battle.pokemon[0], tracker1)
        if battle.pokemon[1]:
            tracker0.reveal_pokemon(battle.pokemon[1].name)
            self._detect_ability_reveal(battle.pokemon[1], tracker0)

        # ターン記録
        player0_turns = []
        player1_turns = []

        # 行動前のポケモン状態を記録（技使用検出用）
        prev_pokemon0 = battle.pokemon[0]
        prev_pokemon1 = battle.pokemon[1]

        # バトルループ
        for turn in range(self.config.max_turns):
            if battle.winner() is not None:
                break

            # 各プレイヤーの状態を記録（観測情報のみ）
            state0 = _get_turn_state(battle, 0, tracker0)
            state1 = _get_turn_state(battle, 1, tracker1)

            # 行動を決定
            available0 = battle.available_commands(0, phase="battle")
            available1 = battle.available_commands(1, phase="battle")

            if isinstance(self.policy, MCTSPolicy):
                # MCTSベースの行動選択
                action0 = self.policy.get_action(battle, 0, available0) if available0 else Battle.SKIP
                action1 = self.policy.get_action(battle, 1, available1) if available1 else Battle.SKIP
            elif isinstance(self.policy, EpsilonGreedyPolicy):
                # モデルベースの行動選択
                # NOTE: 本来はコンテキストをエンコードすべきだが、簡略化のためランダム
                # （選出予測のみモデルを使用し、行動はランダム探索で多様性を確保）
                action0 = random.choice(available0) if available0 else Battle.SKIP
                action1 = random.choice(available1) if available1 else Battle.SKIP
            else:
                action0 = self.policy.get_action(available0)
                action1 = self.policy.get_action(available1)

            # 記録
            player0_turns.append(TurnRecord(
                state=state0,
                action=action0,
                action_name=self._action_to_name(battle, 0, action0),
                reward=0.0,
            ))
            player1_turns.append(TurnRecord(
                state=state1,
                action=action1,
                action_name=self._action_to_name(battle, 1, action1),
                reward=0.0,
            ))

            # 技使用を記録（行動前に記録）
            self._record_move_usage(action0, prev_pokemon0, tracker1)
            self._record_move_usage(action1, prev_pokemon1, tracker0)

            # テラスタルを記録
            self._record_terastallization(action0, prev_pokemon0, tracker1)
            self._record_terastallization(action1, prev_pokemon1, tracker0)

            # バトルを進行
            try:
                battle.proceed(commands=[action0, action1])
            except Exception as e:
                logger.warning(f"Battle error: {e}")
                break

            # ターン後の情報更新
            self._update_observations_after_turn(battle, tracker0, tracker1, prev_pokemon0, prev_pokemon1)

            # 次のターン用に現在のポケモンを記録
            prev_pokemon0 = battle.pokemon[0]
            prev_pokemon1 = battle.pokemon[1]

            # 交代が必要な場合
            for player in [0, 1]:
                if battle.breakpoint[player]:
                    change_available = battle.available_commands(player, phase="change")
                    if change_available:
                        change_cmd = random.choice(change_available)
                        battle.reserved_change_commands[player].append(change_cmd)
                        battle.proceed()

                        # 交代後のポケモンを公開情報として記録
                        if player == 0 and battle.pokemon[0]:
                            tracker1.reveal_pokemon(battle.pokemon[0].name)
                            self._detect_ability_reveal(battle.pokemon[0], tracker1)
                            prev_pokemon0 = battle.pokemon[0]
                        elif player == 1 and battle.pokemon[1]:
                            tracker0.reveal_pokemon(battle.pokemon[1].name)
                            self._detect_ability_reveal(battle.pokemon[1], tracker0)
                            prev_pokemon1 = battle.pokemon[1]

        # 勝者を決定
        winner = battle.winner()

        # 最終報酬を設定
        if winner == 0 and player0_turns:
            player0_turns[-1].reward = 1.0
        if winner == 1 and player1_turns:
            player1_turns[-1].reward = 1.0
        if winner == 0 and player1_turns:
            player1_turns[-1].reward = -1.0
        if winner == 1 and player0_turns:
            player0_turns[-1].reward = -1.0

        return BattleTrajectory(
            game_id=game_id,
            player0_team=team0_data,
            player1_team=team1_data,
            player0_selection=selection0,
            player1_selection=selection1,
            player0_turns=player0_turns,
            player1_turns=player1_turns,
            player0_observations=tracker0,
            player1_observations=tracker1,
            winner=winner,
            total_turns=battle.turn,
        )

    def _record_move_usage(
        self,
        action: int,
        pokemon: Pokemon | None,
        tracker: ObservationTracker,
    ) -> None:
        """技使用を観測トラッカーに記録"""
        if pokemon is None:
            return

        # 技アクション (0-3) またはテラスタル技 (10-13)
        move_idx = -1
        if 0 <= action <= 3:
            move_idx = action
        elif 10 <= action <= 13:
            move_idx = action - 10

        if move_idx >= 0 and move_idx < len(pokemon.moves):
            move_name = pokemon.moves[move_idx]
            tracker.reveal_move(pokemon.name, move_name)

    def _record_terastallization(
        self,
        action: int,
        pokemon: Pokemon | None,
        tracker: ObservationTracker,
    ) -> None:
        """テラスタルを観測トラッカーに記録"""
        if pokemon is None:
            return

        # テラスタル技 (10-13)
        if 10 <= action <= 13:
            tera_type = getattr(pokemon, "Ttype", "")
            if tera_type:
                tracker.reveal_tera(pokemon.name, tera_type)

    def _detect_ability_reveal(
        self,
        pokemon: Pokemon | None,
        tracker: ObservationTracker,
    ) -> None:
        """場に出た時に発動する特性を検出"""
        if pokemon is None:
            return

        # 場に出た時に即座に発動する特性
        instant_abilities = {
            "いかく", "ひでり", "あめふらし", "すなおこし", "ゆきふらし",
            "エレキメイカー", "グラスメイカー", "ミストメイカー", "サイコメイカー",
            "おみとおし", "かたやぶり", "ダウンロード", "トレース", "よちむ",
            "こだいかっせい", "クォークチャージ", "ひひいろのこどう",
            "わざわいのうつわ", "わざわいのつるぎ", "わざわいのおふだ", "わざわいのたま",
        }

        ability = pokemon.ability or ""
        if ability in instant_abilities:
            tracker.reveal_ability(pokemon.name, ability)

    def _update_observations_after_turn(
        self,
        battle: Battle,
        tracker0: ObservationTracker,
        tracker1: ObservationTracker,
        prev_pokemon0: Pokemon | None,
        prev_pokemon1: Pokemon | None,
    ) -> None:
        """ターン終了後の観測更新（持ち物・特性の発動検出）"""
        # 持ち物の発動を検出
        self._detect_item_reveal(battle.pokemon[0], prev_pokemon0, tracker1)
        self._detect_item_reveal(battle.pokemon[1], prev_pokemon1, tracker0)

        # ターン終了時に発動する特性を検出
        self._detect_end_turn_ability(battle.pokemon[0], tracker1)
        self._detect_end_turn_ability(battle.pokemon[1], tracker0)

        # 新しいポケモンが場に出ていれば記録
        if battle.pokemon[0] and prev_pokemon0 and battle.pokemon[0].name != prev_pokemon0.name:
            tracker1.reveal_pokemon(battle.pokemon[0].name)
            self._detect_ability_reveal(battle.pokemon[0], tracker1)
        if battle.pokemon[1] and prev_pokemon1 and battle.pokemon[1].name != prev_pokemon1.name:
            tracker0.reveal_pokemon(battle.pokemon[1].name)
            self._detect_ability_reveal(battle.pokemon[1], tracker0)

    def _detect_item_reveal(
        self,
        pokemon: Pokemon | None,
        prev_pokemon: Pokemon | None,
        tracker: ObservationTracker,
    ) -> None:
        """持ち物の発動を検出"""
        if pokemon is None:
            return

        # 持ち物が消費された場合
        prev_item = getattr(prev_pokemon, "item", "") if prev_pokemon and prev_pokemon.name == pokemon.name else ""
        current_item = pokemon.item or ""

        # 消費されて判明
        if prev_item and not current_item:
            tracker.reveal_item(pokemon.name, prev_item)

        # 特定の持ち物は効果発動時に判明
        detectable_items = {
            "きあいのタスキ", "たべのこし", "くろいヘドロ", "いのちのたま",
            "ゴツゴツメット", "とつげきチョッキ", "ブーストエナジー", "ふうせん",
            "オボンのみ", "ラムのみ", "カゴのみ", "ヤチェのみ", "シュカのみ",
            "ハバンのみ", "ホズのみ", "リンドのみ", "ソクノのみ", "ヨプのみ",
            "こだわりハチマキ", "こだわりメガネ", "こだわりスカーフ",
        }
        if current_item in detectable_items:
            # 持ち物が変化したか、HP変動があった場合に発動とみなす
            # （簡略化：持ち物があれば記録）
            pass  # 実際の発動検出は複雑なのでスキップ

    def _detect_end_turn_ability(
        self,
        pokemon: Pokemon | None,
        tracker: ObservationTracker,
    ) -> None:
        """ターン終了時に発動する特性を検出"""
        if pokemon is None:
            return

        end_turn_abilities = {
            "かそく", "ポイズンヒール", "ムラっけ", "サンパワー",
            "あめうけざら", "かんそうはだ", "スロースタート",
        }

        ability = pokemon.ability or ""
        if ability in end_turn_abilities:
            # 実際の発動確認は複雑なのでスキップ（モデルが学習で吸収）
            pass

    def _action_to_name(self, battle: Battle, player: int, action: int) -> str:
        """行動IDを名前に変換"""
        pokemon = battle.pokemon[player]
        if pokemon is None:
            return f"action_{action}"

        if 0 <= action <= 3:
            if action < len(pokemon.moves):
                return pokemon.moves[action]
            return f"move_{action}"
        elif 10 <= action <= 13:
            move_idx = action - 10
            if move_idx < len(pokemon.moves):
                return f"tera+{pokemon.moves[move_idx]}"
            return f"tera_move_{move_idx}"
        elif 20 <= action <= 25:
            bench_idx = action - 20
            if bench_idx < len(battle.selected[player]):
                bench_pokemon = battle.selected[player][bench_idx]
                if bench_pokemon:
                    return f"switch_{bench_pokemon.name}"
            return f"switch_{bench_idx}"
        elif action == 30:
            return "struggle"
        elif action == -1:
            return "skip"
        else:
            return f"action_{action}"

    def generate_batch(
        self,
        num_games: int,
        num_workers: int | None = None,
    ) -> list[BattleTrajectory]:
        """
        複数のゲームを生成

        Args:
            num_games: 生成するゲーム数
            num_workers: 並列ワーカー数（None なら設定値を使用）

        Returns:
            軌跡のリスト
        """
        num_workers = num_workers or self.config.num_workers

        if num_workers <= 1:
            # シングルスレッド
            trajectories = []
            for i in range(num_games):
                traj = self.generate_trajectory(game_id=f"game_{i}")
                trajectories.append(traj)
                if (i + 1) % 10 == 0:
                    logger.info(f"Generated {i + 1}/{num_games} games")
            return trajectories
        else:
            # マルチプロセス（注意: モデルの共有は複雑なのでランダムポリシーのみ）
            return self._generate_batch_parallel(num_games, num_workers)

    def _generate_batch_parallel(
        self,
        num_games: int,
        num_workers: int,
    ) -> list[BattleTrajectory]:
        """並列でゲームを生成（MCTSも対応）"""
        trajectories = []

        # 設定を辞書化（pickle可能にするため）
        config_dict = {
            "epsilon": self.config.epsilon,
            "temperature": self.config.temperature,
            "max_turns": self.config.max_turns,
            "num_workers": 1,  # ワーカー内では並列化しない
            "usage_data_path": self.config.usage_data_path,
            # MCTS設定
            "use_mcts": self.config.use_mcts,
            "mcts_simulations": self.config.mcts_simulations,
            "mcts_max_depth": self.config.mcts_max_depth,
            "mcts_c_puct": self.config.mcts_c_puct,
            "device": self.config.device,
            "model_checkpoint_path": self.config.model_checkpoint_path,
        }

        if self.config.use_mcts and self.config.model_checkpoint_path:
            logger.info(
                f"Parallel MCTS enabled: {num_workers} workers, "
                f"{self.config.mcts_simulations} sims, "
                f"checkpoint={self.config.model_checkpoint_path}"
            )

        # 引数リストを作成
        args_list = [
            (f"game_{i}", self.trainer_data, config_dict)
            for i in range(num_games)
        ]

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {
                executor.submit(_parallel_game_worker, args): i
                for i, args in enumerate(args_list)
            }

            for future in as_completed(futures):
                try:
                    traj_dict = future.result()
                    if traj_dict is not None:
                        traj = BattleTrajectory.from_dict(traj_dict)
                        trajectories.append(traj)

                        if len(trajectories) % 10 == 0:
                            logger.info(f"Generated {len(trajectories)}/{num_games} games")
                except Exception as e:
                    logger.error(f"Game generation failed: {e}")

        return trajectories


def generate_random_trajectories(
    trainer_json_path: str,
    num_games: int,
    output_path: str | None = None,
) -> list[BattleTrajectory]:
    """
    ランダムポリシーで軌跡を生成

    Args:
        trainer_json_path: トレーナーJSONのパス
        num_games: 生成するゲーム数
        output_path: 出力先（None なら保存しない）

    Returns:
        軌跡のリスト
    """
    import json
    from pathlib import Path

    from .dataset import save_trajectories_to_jsonl

    # トレーナーデータをロード
    with open(trainer_json_path, encoding="utf-8") as f:
        trainer_data = json.load(f)

    # 生成
    generator = TrajectoryGenerator(trainer_data=trainer_data)
    trajectories = generator.generate_batch(num_games)

    # 保存
    if output_path:
        save_trajectories_to_jsonl(trajectories, Path(output_path))
        logger.info(f"Saved {len(trajectories)} trajectories to {output_path}")

    return trajectories
