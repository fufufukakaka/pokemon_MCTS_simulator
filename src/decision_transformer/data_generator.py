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
    PokemonState,
    TurnRecord,
    TurnState,
)

if TYPE_CHECKING:
    from .model import PokemonBattleTransformer
    from .tokenizer import BattleSequenceTokenizer

logger = logging.getLogger(__name__)


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


def _pokemon_to_state(pokemon: Pokemon | None) -> PokemonState:
    """Pokemon オブジェクトを PokemonState に変換"""
    if pokemon is None:
        return PokemonState(name="", hp_ratio=0.0)

    max_hp = pokemon.status[0] if pokemon.status else 1
    hp_ratio = pokemon.hp / max_hp if max_hp > 0 else 0.0

    # pokemon.condition から状態変化を取得
    condition = getattr(pokemon, "condition", {})

    return PokemonState(
        name=pokemon.name,
        hp_ratio=hp_ratio,
        ailment=pokemon.ailment or "",
        rank=list(pokemon.rank[:8]) if pokemon.rank else [0] * 8,
        types=list(pokemon.types) if pokemon.types else [],
        terastallized=pokemon.terastal if hasattr(pokemon, "terastal") else False,
        tera_type=pokemon.Ttype if hasattr(pokemon, "Ttype") else "",
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


def _get_turn_state(battle: Battle, player: int) -> TurnState:
    """現在のターン状態を取得"""
    opponent = 1 - player

    # 自分のポケモン
    my_active = _pokemon_to_state(battle.pokemon[player])
    my_bench = []
    for p in battle.selected[player]:
        if p is not None and p is not battle.pokemon[player]:
            my_bench.append(_pokemon_to_state(p))

    # 相手のポケモン
    opp_active = _pokemon_to_state(battle.pokemon[opponent])
    opp_bench = []
    for p in battle.selected[opponent]:
        if p is not None and p is not battle.pokemon[opponent]:
            opp_bench.append(_pokemon_to_state(p))

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
        field=field,
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

        # Pokemon データの初期化
        Pokemon.init()

        # ポリシーの設定
        if model is not None and tokenizer is not None:
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
        if isinstance(self.policy, EpsilonGreedyPolicy):
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

        # バトル開始
        battle.proceed(commands=[None, None])

        # ターン記録
        player0_turns = []
        player1_turns = []

        # バトルループ
        for turn in range(self.config.max_turns):
            if battle.winner() is not None:
                break

            # 各プレイヤーの状態を記録
            state0 = _get_turn_state(battle, 0)
            state1 = _get_turn_state(battle, 1)

            # 行動を決定
            available0 = battle.available_commands(0, phase="battle")
            available1 = battle.available_commands(1, phase="battle")

            if isinstance(self.policy, EpsilonGreedyPolicy):
                # モデルベースの行動選択（簡略化：ランダムで代用）
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

            # バトルを進行
            try:
                battle.proceed(commands=[action0, action1])
            except Exception as e:
                logger.warning(f"Battle error: {e}")
                break

            # 交代が必要な場合
            for player in [0, 1]:
                if battle.breakpoint[player]:
                    change_available = battle.available_commands(player, phase="change")
                    if change_available:
                        change_cmd = random.choice(change_available)
                        battle.reserved_change_commands[player].append(change_cmd)
                        battle.proceed()

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
            winner=winner,
            total_turns=battle.turn,
        )

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
        """並列でゲームを生成（ランダムポリシーのみ）"""
        trajectories = []

        # ワーカー関数
        def worker(game_id: str, trainer_data: list) -> dict:
            Pokemon.init()
            gen = TrajectoryGenerator(
                trainer_data=trainer_data,
                config=self.config,
            )
            traj = gen.generate_trajectory(game_id=game_id)
            return traj.to_dict()

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {
                executor.submit(worker, f"game_{i}", self.trainer_data): i
                for i in range(num_games)
            }

            for future in as_completed(futures):
                try:
                    traj_dict = future.result()
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
