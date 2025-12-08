"""
ReBeL Trainer

自己対戦によるデータ生成と Value Network の学習を行う。
選出ネットワークとの統合学習もサポート。

Performance Optimizations:
- Parallel game generation with multiprocessing
- Lightweight CFR solver option
- Configurable CFR complexity
"""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import random
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
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
from src.policy_value_network.team_selection_encoder import TeamSelectionEncoder
from src.policy_value_network.team_selection_network import (
    TeamSelectionNetwork,
    TeamSelectionNetworkConfig,
)

from .belief_state import Observation, ObservationType, PokemonBeliefState
from .cfr_solver import CFRConfig, ReBeLSolver
from .full_belief_state import FullBeliefState
from .public_state import PublicBeliefState, PublicGameState
from .value_network import ReBeLValueNetwork


# ============================================================
# バトルログから観測を抽出するための定数とユーティリティ
# ============================================================

# 持ち物→観測タイプのマッピング
ITEM_OBSERVATION_MAP = {
    "きあいのタスキ": ObservationType.FOCUS_SASH_ACTIVATED,
    "たべのこし": ObservationType.LEFTOVERS_HEAL,
    "くろいヘドロ": ObservationType.BLACK_SLUDGE_HEAL,
    "いのちのたま": ObservationType.LIFE_ORB_RECOIL,
    "ゴツゴツメット": ObservationType.ROCKY_HELMET_DAMAGE,
    "とつげきチョッキ": ObservationType.ASSAULT_VEST_BLOCK,
    "ブーストエナジー": ObservationType.BOOST_ENERGY_ACTIVATED,
    "ふうせん": ObservationType.AIR_BALLOON_CONSUMED,
}

# きのみのリスト
BERRIES = [
    "オボンのみ", "ラムのみ", "カゴのみ", "クラボのみ", "モモンのみ",
    "チーゴのみ", "ナナシのみ", "ヒメリのみ", "オレンのみ", "キーのみ",
    "ウイのみ", "バンジのみ", "イアのみ", "フィラのみ", "マゴのみ",
    "イバンのみ", "ヤタピのみ", "カムラのみ", "サンのみ", "チイラのみ",
    "リュガのみ", "ズアのみ", "アッキのみ", "タラプのみ",
    # タイプ半減きのみ
    "ソクノのみ", "タンガのみ", "ヨプのみ", "シュカのみ", "バコウのみ",
    "ウタンのみ", "オッカのみ", "イトケのみ", "リンドのみ", "ヤチェのみ",
    "ビアーのみ", "ナモのみ", "リリバのみ", "ホズのみ", "ハバンのみ",
    "カシブのみ", "レンブのみ", "ロゼルのみ",
]


def extract_item_observations_from_log(
    battle_log: list[str], pokemon_name: str
) -> list[Observation]:
    """
    バトルログから持ち物発動の観測イベントを抽出

    Args:
        battle_log: バトルログ（battle.log[player]）
        pokemon_name: 対象のポケモン名

    Returns:
        観測イベントのリスト
    """
    observations = []

    for entry in battle_log:
        if not isinstance(entry, str):
            continue

        # ポケモン名がログエントリに含まれているかチェック
        if pokemon_name not in entry:
            continue

        # 持ち物発動の検出
        for item, obs_type in ITEM_OBSERVATION_MAP.items():
            if item in entry:
                observations.append(
                    Observation(
                        type=obs_type,
                        pokemon_name=pokemon_name,
                        details={"item": item, "log_entry": entry},
                    )
                )
                break

        # きのみ消費の検出
        for berry in BERRIES:
            if berry in entry and ("発動" in entry or "回復" in entry or "上がった" in entry):
                observations.append(
                    Observation(
                        type=ObservationType.BERRY_CONSUMED,
                        pokemon_name=pokemon_name,
                        details={"item": berry, "log_entry": entry},
                    )
                )
                break

        # こだわり系の検出（技固定の表示）
        if "こだわり" in entry and ("固定" in entry or "変化技" in entry):
            observations.append(
                Observation(
                    type=ObservationType.CHOICE_LOCKED,
                    pokemon_name=pokemon_name,
                    details={"log_entry": entry},
                )
            )

    return observations


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
class SelectionExample:
    """選出学習用データの1サンプル"""

    my_team_data: list[dict]  # 6匹の元データ
    opp_team_data: list[dict]  # 相手の6匹
    selected_indices: list[int]  # 選出した3匹のインデックス
    lead_index: int = 0  # 先発のインデックス（6匹中での位置）
    winner: Optional[int] = None  # 勝者（0 or 1）
    perspective: int = 0  # どちらの視点か


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

    # 固定対戦相手の設定
    # 設定すると Player 1 は常にこのパーティを使用
    fixed_opponent: Optional[dict] = None  # トレーナーデータ形式
    fixed_opponent_select_all: bool = False  # True の場合、ランダム選出ではなく先頭3体を使用

    # 選出学習の設定
    train_selection: bool = False  # 選出ネットワークも学習するか
    selection_learning_rate: float = 1e-4
    selection_explore_prob: float = 0.3  # 探索時にランダム選出する確率

    # ポケモン統計データのパス
    usage_data_path: Optional[str] = None  # None の場合はデフォルト(season22.json)を使用

    # 完全信念状態を使用するか（選出・先発の不確実性を含む）
    use_full_belief: bool = False

    # 並列化設定
    num_workers: int = 1  # 並列ゲーム生成のワーカー数（1=逐次実行）
    use_lightweight_cfr: bool = True  # 軽量CFRモード（高速だが精度低下）
    skip_cfr_for_obvious: bool = True  # 行動が1つしかない場合はCFRをスキップ


def _generate_game_worker(
    args: tuple[str, list[dict], dict, str, str, bool, bool, Optional[dict], bool, bool],
) -> tuple[
    dict,  # GameResult as dict (serializable)
    list[dict],  # PBS data as dicts
    list[dict],  # Selection data as dicts
]:
    """
    並列実行用のゲーム生成ワーカー関数

    Note: multiprocessing で使用するため、引数はシリアライズ可能である必要がある
    """
    (
        game_id,
        trainer_data,
        config_dict,
        usage_db_path,
        usage_data_path,
        train_selection,
        use_lightweight_cfr,
        fixed_opponent,
        fixed_opponent_select_all,
        use_full_belief,
    ) = args

    # 各ワーカーで必要なオブジェクトを再構築
    from src.hypothesis.pokemon_usage_database import PokemonUsageDatabase
    from src.pokemon_battle_sim.battle import Battle
    from src.pokemon_battle_sim.pokemon import Pokemon

    from .belief_state import PokemonBeliefState
    from .cfr_solver import CFRConfig, ReBeLSolver
    from .full_belief_state import FullBeliefState
    from .public_state import PublicBeliefState

    # 初期化
    Pokemon.init(usage_data_path=usage_data_path)
    usage_db = PokemonUsageDatabase.from_json(usage_db_path)

    # CFRソルバー
    cfr_config = CFRConfig(
        num_iterations=config_dict.get("cfr_iterations", 30),
        num_world_samples=config_dict.get("cfr_world_samples", 10),
    )
    solver = ReBeLSolver(
        value_network=None,
        cfr_config=cfr_config,
        use_simplified=True,
        use_lightweight=use_lightweight_cfr,
    )

    # トレーナー選択
    trainer0 = random.choice(trainer_data)
    if fixed_opponent is not None:
        trainer1 = fixed_opponent
        trainer1_fixed_selection = fixed_opponent_select_all
    else:
        trainer1 = random.choice(trainer_data)
        while trainer1 is trainer0 and len(trainer_data) > 1:
            trainer1 = random.choice(trainer_data)
        trainer1_fixed_selection = False

    trainer0_team = trainer0.get("pokemons", [])[:6]
    trainer1_team = trainer1.get("pokemons", [])[:6]

    # バトル初期化
    battle = Battle()
    battle.reset_game()

    # チーム設定
    selected_indices_0: list[int] = []
    selected_indices_1: list[int] = []

    # ランダム選出
    available_0 = list(range(len(trainer0_team)))
    selected_indices_0 = random.sample(available_0, min(3, len(available_0)))
    for idx in selected_indices_0:
        pokemon_data = trainer0_team[idx]
        pokemon = Pokemon(pokemon_data.get("name", "ピカチュウ"))
        pokemon.item = pokemon_data.get("item", "")
        pokemon.ability = pokemon_data.get("ability", "")
        pokemon.moves = pokemon_data.get("moves", [])[:4]
        pokemon.Ttype = pokemon_data.get("Ttype") or pokemon_data.get("tera_type", "ノーマル")
        if "effort" in pokemon_data:
            pokemon.effort = pokemon_data["effort"]
        elif "evs" in pokemon_data:
            pokemon.effort = pokemon_data["evs"]
        if "nature" in pokemon_data:
            pokemon.nature = pokemon_data["nature"]
        battle.selected[0].append(pokemon)

    if trainer1_fixed_selection:
        selected_indices_1 = list(range(min(3, len(trainer1_team))))
    else:
        available_1 = list(range(len(trainer1_team)))
        selected_indices_1 = random.sample(available_1, min(3, len(available_1)))

    for idx in selected_indices_1:
        pokemon_data = trainer1_team[idx]
        pokemon = Pokemon(pokemon_data.get("name", "ピカチュウ"))
        pokemon.item = pokemon_data.get("item", "")
        pokemon.ability = pokemon_data.get("ability", "")
        pokemon.moves = pokemon_data.get("moves", [])[:4]
        pokemon.Ttype = pokemon_data.get("Ttype") or pokemon_data.get("tera_type", "ノーマル")
        if "effort" in pokemon_data:
            pokemon.effort = pokemon_data["effort"]
        elif "evs" in pokemon_data:
            pokemon.effort = pokemon_data["evs"]
        if "nature" in pokemon_data:
            pokemon.nature = pokemon_data["nature"]
        battle.selected[1].append(pokemon)

    # ターン0
    battle.proceed(commands=[Battle.SKIP, Battle.SKIP])

    # 信念状態の初期化
    full_beliefs: list[Optional[FullBeliefState]] = [None, None]

    if use_full_belief:
        # 完全信念状態を使用
        full_belief_0 = FullBeliefState(
            team_preview_names=[p.get("name", "") for p in trainer1_team],
            team_preview_data=trainer1_team,
            usage_db=usage_db,
            selector=None,
            my_team_data=trainer0_team,
        )
        full_belief_1 = FullBeliefState(
            team_preview_names=[p.get("name", "") for p in trainer0_team],
            team_preview_data=trainer0_team,
            usage_db=usage_db,
            selector=None,
            my_team_data=trainer1_team,
        )
        full_beliefs = [full_belief_0, full_belief_1]

        # 先発が判明した状態で更新
        lead_pokemon_0 = battle.pokemon[1]
        lead_pokemon_1 = battle.pokemon[0]
        if lead_pokemon_0:
            full_belief_0.update_lead_revealed(lead_pokemon_0.name)
        if lead_pokemon_1:
            full_belief_1.update_lead_revealed(lead_pokemon_1.name)

        # PokemonBeliefStateに変換
        beliefs = [
            full_belief_0.to_pokemon_belief_state() or PokemonBeliefState(
                [p.name for p in battle.selected[1]], usage_db
            ),
            full_belief_1.to_pokemon_belief_state() or PokemonBeliefState(
                [p.name for p in battle.selected[0]], usage_db
            ),
        ]
    else:
        beliefs = [
            PokemonBeliefState([p.name for p in battle.selected[1]], usage_db),
            PokemonBeliefState([p.name for p in battle.selected[0]], usage_db),
        ]

    examples: list[dict] = []
    pbs_records: list[tuple[dict, int]] = []
    turn = 0
    last_actions = [Battle.SKIP, Battle.SKIP]
    max_turns = config_dict.get("max_turns", 100)
    log_lengths: dict[int, int] = {0: 0, 1: 0}

    while battle.winner() is None and turn < max_turns:
        turn += 1

        for player in [0, 1]:
            if battle.winner() is not None:
                break

            try:
                pbs = PublicBeliefState.from_battle(battle, player, beliefs[player])
            except Exception:
                continue

            # PBS を完全にシリアライズして記録（from_dictで復元可能）
            pbs_dict = pbs.to_dict()
            pbs_records.append((pbs_dict, player))

            # CFRで戦略計算
            try:
                my_strategy, opp_strategy = solver.solve(pbs, battle)
            except Exception:
                my_strategy = {}
                opp_strategy = {}

            example = {
                "public_state_dict": pbs_dict,
                "belief_summary": {},
                "my_strategy": my_strategy,
                "opp_strategy": opp_strategy,
                "action": None,
                "target_my_value": None,
                "target_opp_value": None,
            }
            examples.append(example)

            # 行動選択
            try:
                action = solver.get_action(pbs, battle, explore=True, temperature=1.0)
            except Exception:
                available = battle.available_commands(player)
                action = random.choice(available) if available else Battle.SKIP

            example["action"] = action
            last_actions[player] = action

        # ターン実行
        try:
            battle.proceed(commands=last_actions)
        except Exception:
            break

        # FullBeliefState の更新（交代で新しいポケモンが出た場合）
        if use_full_belief:
            for player in [0, 1]:
                fb = full_beliefs[player]
                if fb is None:
                    continue
                opponent = 1 - player
                current_pokemon = battle.pokemon[opponent]
                if current_pokemon:
                    pokemon_name = current_pokemon.name
                    confirmed = fb.get_confirmed_selected_names()
                    if pokemon_name not in confirmed:
                        fb.update_pokemon_revealed(pokemon_name)
                        new_belief = fb.to_pokemon_belief_state()
                        if new_belief:
                            beliefs[player] = new_belief

        # バトルログから観測を抽出して信念を更新
        if hasattr(battle, 'log'):
            for player in [0, 1]:
                opponent = 1 - player
                belief = beliefs[player]

                # 相手のポケモン情報
                opp_pokemon = battle.pokemon[opponent]
                if opp_pokemon is None:
                    continue

                pokemon_name = opp_pokemon.name

                # ログを取得
                current_log = battle.log[player] if isinstance(battle.log, list) and len(battle.log) > player else []
                if not isinstance(current_log, list):
                    current_log = []

                prev_length = log_lengths.get(player, 0)
                log_lengths[player] = len(current_log)

                if len(current_log) <= prev_length:
                    continue

                # 新しいログエントリを取得
                new_entries = current_log[prev_length:]

                # 持ち物観測を抽出
                observations = extract_item_observations_from_log(new_entries, pokemon_name)
                for obs in observations:
                    belief.update(obs)
                    # FullBeliefState にも反映
                    if use_full_belief:
                        fb = full_beliefs[player]
                        if fb is not None:
                            fb.update(obs)

    # 終局処理
    winner = battle.winner()
    pbs_with_targets: list[dict] = []
    for pbs_dict, player in pbs_records:
        if winner is not None:
            target_my = 1.0 if winner == player else 0.0
            target_opp = 1.0 if winner != player else 0.0
        else:
            target_my = 0.5
            target_opp = 0.5
        pbs_with_targets.append({
            "pbs_dict": pbs_dict,
            "player": player,
            "target_my": target_my,
            "target_opp": target_opp,
        })

    for i, example in enumerate(examples):
        player = i % 2
        if winner is not None:
            example["target_my_value"] = 1.0 if winner == player else 0.0
            example["target_opp_value"] = 1.0 if winner != player else 0.0
        else:
            example["target_my_value"] = 0.5
            example["target_opp_value"] = 0.5

    result_dict = {
        "game_id": game_id,
        "winner": winner,
        "total_turns": turn,
        "examples": examples,
    }

    # 選出データ
    selection_data: list[dict] = []
    if train_selection:
        selection_data.append({
            "my_team_data": trainer0_team,
            "opp_team_data": trainer1_team,
            "selected_indices": selected_indices_0,
            "lead_index": selected_indices_0[0] if selected_indices_0 else 0,
            "winner": winner,
            "perspective": 0,
        })
        if not trainer1_fixed_selection:
            selection_data.append({
                "my_team_data": trainer1_team,
                "opp_team_data": trainer0_team,
                "selected_indices": selected_indices_1,
                "lead_index": selected_indices_1[0] if selected_indices_1 else 0,
                "winner": winner,
                "perspective": 1,
            })

    return result_dict, pbs_with_targets, selection_data


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
    3. （オプション）選出ネットワークの学習
    4. 繰り返し
    """

    def __init__(
        self,
        usage_db: PokemonUsageDatabase,
        trainer_data: list[dict],
        config: Optional[TrainingConfig] = None,
        value_network: Optional[ReBeLValueNetwork] = None,
        selection_network: Optional[TeamSelectionNetwork] = None,
    ):
        """
        Args:
            usage_db: ポケモン使用率データベース
            trainer_data: トレーナーデータ（チーム情報）
            config: 学習設定
            value_network: 既存の Value Network（None の場合は新規作成）
            selection_network: 選出ネットワーク（None の場合は新規作成、train_selection=True時のみ使用）
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
            use_lightweight=self.config.use_lightweight_cfr,
        )

        # 選出ネットワーク（オプション）
        self.selection_network: Optional[TeamSelectionNetwork] = None
        self.selection_optimizer: Optional[optim.Optimizer] = None
        self.selection_encoder: Optional[TeamSelectionEncoder] = None

        if self.config.train_selection:
            self.selection_encoder = TeamSelectionEncoder()
            self.selection_network = selection_network or TeamSelectionNetwork(
                TeamSelectionNetworkConfig(pokemon_feature_dim=15)
            )
            self.selection_network.to(self.config.device)
            self.selection_optimizer = optim.AdamW(
                self.selection_network.parameters(),
                lr=self.config.selection_learning_rate,
                weight_decay=self.config.weight_decay,
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
        Pokemon.init(usage_data_path=self.config.usage_data_path)
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
        log_lengths: dict[int, int] = {0: 0, 1: 0}

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

            # 観測更新（バトルログから持ち物発動等を抽出）
            log_lengths = self._update_beliefs_from_battle(battle, beliefs, None, log_lengths)

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
    ) -> tuple[
        GameResult,
        list[tuple[PublicBeliefState, Optional[float], Optional[float]]],
        list[SelectionExample],
    ]:
        """
        1試合を実行してデータを生成（PBS オブジェクトも返す）

        Returns:
            (GameResult, list of (PBS, target_my_value, target_opp_value), list of SelectionExample)
        """
        # Player 0 はランダムに選択
        trainer0 = random.choice(self.trainer_data)

        # Player 1 は固定対戦相手が設定されていればそれを使用
        if self.config.fixed_opponent is not None:
            trainer1 = self.config.fixed_opponent
            trainer1_fixed_selection = self.config.fixed_opponent_select_all
        else:
            trainer1 = random.choice(self.trainer_data)
            # 同じトレーナーが選ばれた場合は別のを選ぶ
            while trainer1 is trainer0 and len(self.trainer_data) > 1:
                trainer1 = random.choice(self.trainer_data)
            trainer1_fixed_selection = False

        # 6匹のデータを保存
        trainer0_team = trainer0.get("pokemons", [])[:6]
        trainer1_team = trainer1.get("pokemons", [])[:6]

        # バトル初期化
        Pokemon.init(usage_data_path=self.config.usage_data_path)
        battle = Battle()
        battle.reset_game()

        # 選出ネットワークを使った選出 or ランダム選出
        selected_indices_0: list[int] = []
        selected_indices_1: list[int] = []

        if self.config.train_selection and self.selection_network is not None:
            # Player 0: 選出ネットワーク or 探索
            selected_indices_0 = self._select_team_with_network(
                trainer0_team, trainer1_team, explore=True
            )
            self._setup_team_with_indices(battle, 0, trainer0_team, selected_indices_0)
        else:
            selected_indices_0 = self._setup_team(battle, 0, trainer0, fixed_selection=False)

        if trainer1_fixed_selection:
            selected_indices_1 = self._setup_team(battle, 1, trainer1, fixed_selection=True)
        elif self.config.train_selection and self.selection_network is not None:
            # 固定でない場合は選出ネットワーク
            selected_indices_1 = self._select_team_with_network(
                trainer1_team, trainer0_team, explore=True
            )
            self._setup_team_with_indices(battle, 1, trainer1_team, selected_indices_1)
        else:
            selected_indices_1 = self._setup_team(battle, 1, trainer1, fixed_selection=False)

        # ターン0で先頭のポケモンを場に出す
        battle.proceed(commands=[Battle.SKIP, Battle.SKIP])

        # 信念状態の初期化
        full_beliefs: list[Optional[FullBeliefState]] = [None, None]

        if self.config.use_full_belief:
            # 完全信念状態を使用（選出・先発の不確実性を含む）
            full_belief_0 = FullBeliefState(
                team_preview_names=[p.get("name", "") for p in trainer1_team],
                team_preview_data=trainer1_team,
                usage_db=self.usage_db,
                selector=None,  # TODO: TeamSelectorを渡す
                my_team_data=trainer0_team,
            )
            full_belief_1 = FullBeliefState(
                team_preview_names=[p.get("name", "") for p in trainer0_team],
                team_preview_data=trainer0_team,
                usage_db=self.usage_db,
                selector=None,
                my_team_data=trainer1_team,
            )
            full_beliefs = [full_belief_0, full_belief_1]

            # 先発が判明した状態で更新
            lead_pokemon_0 = battle.pokemon[1]
            lead_pokemon_1 = battle.pokemon[0]
            if lead_pokemon_0:
                full_belief_0.update_lead_revealed(lead_pokemon_0.name)
            if lead_pokemon_1:
                full_belief_1.update_lead_revealed(lead_pokemon_1.name)

            # PokemonBeliefStateに変換（互換性のため）
            beliefs = [
                full_belief_0.to_pokemon_belief_state() or PokemonBeliefState(
                    [p.name for p in battle.selected[1]],
                    self.usage_db,
                ),
                full_belief_1.to_pokemon_belief_state() or PokemonBeliefState(
                    [p.name for p in battle.selected[0]],
                    self.usage_db,
                ),
            ]
        else:
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
        log_lengths: dict[int, int] = {0: 0, 1: 0}

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

            # 観測更新（バトルログから持ち物発動等を抽出）
            log_lengths = self._update_beliefs_from_battle(battle, beliefs, full_beliefs, log_lengths)

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

        # 選出データ
        selection_examples: list[SelectionExample] = []
        if self.config.train_selection:
            # 先発を記録（selected_indicesの最初が先発）
            lead_0 = selected_indices_0[0] if selected_indices_0 else 0
            lead_1 = selected_indices_1[0] if selected_indices_1 else 0

            # Player 0 の選出データ
            selection_examples.append(
                SelectionExample(
                    my_team_data=trainer0_team,
                    opp_team_data=trainer1_team,
                    selected_indices=selected_indices_0,
                    lead_index=lead_0,
                    winner=winner,
                    perspective=0,
                )
            )
            # Player 1 の選出データ（固定選出でない場合）
            if not trainer1_fixed_selection:
                selection_examples.append(
                    SelectionExample(
                        my_team_data=trainer1_team,
                        opp_team_data=trainer0_team,
                        selected_indices=selected_indices_1,
                        lead_index=lead_1,
                        winner=winner,
                        perspective=1,
                    )
                )

        return result, pbs_with_targets, selection_examples

    def _setup_team(
        self, battle: Battle, player: int, trainer: dict, fixed_selection: bool = False
    ) -> list[int]:
        """
        トレーナーのチームを設定（3体選出）

        Args:
            battle: Battle オブジェクト
            player: プレイヤー番号 (0 or 1)
            trainer: トレーナーデータ
            fixed_selection: True の場合、ランダム選出ではなく先頭3体を使用

        Returns:
            選出されたインデックスのリスト
        """
        pokemons_data = trainer.get("pokemons", [])[:6]

        # ポケモンオブジェクトを作成
        team: list[Pokemon] = []
        for pokemon_data in pokemons_data:
            pokemon = Pokemon(pokemon_data.get("name", "ピカチュウ"))
            pokemon.item = pokemon_data.get("item", "")
            pokemon.ability = pokemon_data.get("ability", "")
            pokemon.moves = pokemon_data.get("moves", [])[:4]
            # Ttype または tera_type の両方に対応
            pokemon.Ttype = pokemon_data.get("Ttype") or pokemon_data.get("tera_type", "ノーマル")

            # ステータス設定 (evs または effort の両方に対応)
            if "effort" in pokemon_data:
                pokemon.effort = pokemon_data["effort"]
            elif "evs" in pokemon_data:
                pokemon.effort = pokemon_data["evs"]
            if "nature" in pokemon_data:
                pokemon.nature = pokemon_data["nature"]

            team.append(pokemon)

        num_select = min(3, len(team))
        selected_indices: list[int] = []

        if num_select == 0:
            # ポケモンがない場合はダミー
            dummy = Pokemon("ピカチュウ")
            battle.selected[player].append(dummy)
            selected_indices = []
        elif fixed_selection:
            # 先頭3体を固定選出
            for i in range(num_select):
                battle.selected[player].append(team[i])
            selected_indices = list(range(num_select))
        else:
            # ランダムに3体選出
            selected_indices = random.sample(range(len(team)), num_select)
            for idx in selected_indices:
                battle.selected[player].append(team[idx])

        return selected_indices

    def _select_team_with_network(
        self, my_team_data: list[dict], opp_team_data: list[dict], explore: bool = True
    ) -> list[int]:
        """
        選出ネットワークを使ってチームを選出

        Args:
            my_team_data: 自分の6匹のデータ
            opp_team_data: 相手の6匹のデータ
            explore: 探索モード（確率的に選出）

        Returns:
            選出するインデックスのリスト
        """
        if self.selection_network is None or self.selection_encoder is None:
            # ネットワークがない場合はランダム選出
            num_select = min(3, len(my_team_data))
            return random.sample(range(len(my_team_data)), num_select)

        # 探索確率でランダム選出
        if explore and random.random() < self.config.selection_explore_prob:
            num_select = min(3, len(my_team_data))
            return random.sample(range(len(my_team_data)), num_select)

        # ネットワークで選出
        self.selection_network.eval()
        with torch.no_grad():
            my_tensor = self.selection_encoder.encode_team(my_team_data)
            opp_tensor = self.selection_encoder.encode_team(opp_team_data)

            my_tensor = my_tensor.unsqueeze(0).to(self.config.device)
            opp_tensor = opp_tensor.unsqueeze(0).to(self.config.device)

            indices, _ = self.selection_network.select_team(
                my_tensor, opp_tensor, num_select=3, deterministic=not explore
            )

        return indices[0].tolist()

    def _setup_team_with_indices(
        self, battle: Battle, player: int, team_data: list[dict], indices: list[int]
    ) -> None:
        """
        指定されたインデックスでチームを設定

        Args:
            battle: Battle オブジェクト
            player: プレイヤー番号
            team_data: 6匹のポケモンデータ
            indices: 選出するインデックス
        """
        for idx in indices:
            if idx >= len(team_data):
                continue
            pokemon_data = team_data[idx]
            pokemon = Pokemon(pokemon_data.get("name", "ピカチュウ"))
            pokemon.item = pokemon_data.get("item", "")
            pokemon.ability = pokemon_data.get("ability", "")
            pokemon.moves = pokemon_data.get("moves", [])[:4]
            pokemon.Ttype = pokemon_data.get("Ttype") or pokemon_data.get("tera_type", "ノーマル")

            if "effort" in pokemon_data:
                pokemon.effort = pokemon_data["effort"]
            elif "evs" in pokemon_data:
                pokemon.effort = pokemon_data["evs"]
            if "nature" in pokemon_data:
                pokemon.nature = pokemon_data["nature"]

            battle.selected[player].append(pokemon)

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
        self,
        battle: Battle,
        beliefs: list[PokemonBeliefState],
        full_beliefs: Optional[list[Optional[FullBeliefState]]] = None,
        prev_log_lengths: Optional[dict[int, int]] = None,
    ) -> dict[int, int]:
        """バトルの進行から信念を更新

        Args:
            battle: Battle オブジェクト
            beliefs: PokemonBeliefState のリスト [player0視点, player1視点]
            full_beliefs: FullBeliefState のリスト（use_full_belief時に使用）
            prev_log_lengths: 前回のログ長さ {player: length}（初回はNone）

        Returns:
            更新後のログ長さ {player: length}
        """
        if prev_log_lengths is None:
            prev_log_lengths = {0: 0, 1: 0}

        new_log_lengths = {0: 0, 1: 0}

        # FullBeliefState が有効な場合は、新しいポケモンの登場を追跡
        if full_beliefs is not None:
            for player in [0, 1]:
                fb = full_beliefs[player]
                if fb is None:
                    continue

                opponent = 1 - player
                # 場にいるポケモンが判明しているかチェック
                current_pokemon = battle.pokemon[opponent]
                if current_pokemon:
                    pokemon_name = current_pokemon.name
                    # まだ確認されていないポケモンなら更新
                    confirmed = fb.get_confirmed_selected_names()
                    if pokemon_name not in confirmed:
                        fb.update_pokemon_revealed(pokemon_name)

                        # PokemonBeliefState も更新
                        new_belief = fb.to_pokemon_belief_state()
                        if new_belief:
                            beliefs[player] = new_belief

        # バトルログから観測を抽出して信念を更新
        if hasattr(battle, 'log'):
            for player in [0, 1]:
                opponent = 1 - player
                belief = beliefs[player]

                # 相手のポケモン情報
                opp_pokemon = battle.pokemon[opponent]
                if opp_pokemon is None:
                    continue

                pokemon_name = opp_pokemon.name

                # ログを取得
                current_log = battle.log[player] if isinstance(battle.log, list) and len(battle.log) > player else []
                if not isinstance(current_log, list):
                    current_log = []

                prev_length = prev_log_lengths.get(player, 0)
                new_log_lengths[player] = len(current_log)

                if len(current_log) <= prev_length:
                    continue

                # 新しいログエントリを取得
                new_entries = current_log[prev_length:]

                # 持ち物観測を抽出
                observations = extract_item_observations_from_log(new_entries, pokemon_name)
                for obs in observations:
                    belief.update(obs)
                    # FullBeliefState にも反映
                    if full_beliefs is not None:
                        fb = full_beliefs[player]
                        if fb is not None:
                            fb.update(obs)

        return new_log_lengths

    def _generate_games_parallel(
        self,
        iteration: int,
        num_games: int,
        num_workers: int,
    ) -> tuple[
        list[TrainingExample],
        list[tuple[PublicBeliefState, float, float]],
        list[SelectionExample],
        dict[Optional[int], int],
    ]:
        """
        並列でゲームを生成

        Args:
            iteration: イテレーション番号
            num_games: 生成するゲーム数
            num_workers: ワーカー数

        Returns:
            (examples, pbs_data, selection_data, wins)
        """
        # usage_db のパスを取得（ワーカーで再ロードするため）
        # Note: PokemonUsageDatabaseにはパスを保持する機能がないため、
        # configから取得するか、デフォルトを使用
        usage_db_path = getattr(self.usage_db, '_source_path', None)
        if usage_db_path is None:
            # デフォルトのパスを使用
            usage_db_path = "data/pokedb_usage/season_37_top150.json"

        # 設定を辞書化
        config_dict = {
            "cfr_iterations": self.config.cfr_iterations,
            "cfr_world_samples": self.config.cfr_world_samples,
            "max_turns": self.config.max_turns,
        }

        # ワーカー引数を準備
        worker_args = []
        for i in range(num_games):
            game_id = f"iter{iteration}_game{i}"
            args = (
                game_id,
                self.trainer_data,
                config_dict,
                usage_db_path,
                self.config.usage_data_path,
                self.config.train_selection,
                self.config.use_lightweight_cfr,
                self.config.fixed_opponent,
                self.config.fixed_opponent_select_all,
                self.config.use_full_belief,
            )
            worker_args.append(args)

        # 並列実行
        all_examples: list[TrainingExample] = []
        all_pbs_data: list[tuple[PublicBeliefState, float, float]] = []
        all_selection_data: list[SelectionExample] = []
        wins: dict[Optional[int], int] = {0: 0, 1: 0, None: 0}

        # ProcessPoolExecutorでゲームを並列生成
        # Note: spawn方式を使用してWindowsとの互換性を確保
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=num_workers, mp_context=ctx) as executor:
            futures = [executor.submit(_generate_game_worker, args) for args in worker_args]

            for future in as_completed(futures):
                try:
                    result_dict, pbs_data_list, selection_data_list = future.result()

                    # GameResult を復元
                    winner = result_dict["winner"]
                    wins[winner] = wins.get(winner, 0) + 1

                    # TrainingExample を復元
                    for ex_dict in result_dict["examples"]:
                        example = TrainingExample(
                            public_state_dict=ex_dict["public_state_dict"],
                            belief_summary=ex_dict["belief_summary"],
                            my_strategy=ex_dict["my_strategy"],
                            opp_strategy=ex_dict["opp_strategy"],
                            action=ex_dict["action"],
                            target_my_value=ex_dict["target_my_value"],
                            target_opp_value=ex_dict["target_opp_value"],
                        )
                        all_examples.append(example)

                    # PBS データ（簡略版）- 実際のPBSオブジェクトではなくdictを使用
                    # Note: 並列実行では完全なPBSオブジェクトを渡すのが難しいため、
                    # train_iteration側で対応が必要
                    for pbs_dict in pbs_data_list:
                        # ここではdictのまま保持（学習時に別途処理）
                        all_pbs_data.append((
                            pbs_dict,  # type: ignore
                            pbs_dict["target_my"],
                            pbs_dict["target_opp"],
                        ))

                    # SelectionExample を復元
                    for sel_dict in selection_data_list:
                        sel_ex = SelectionExample(
                            my_team_data=sel_dict["my_team_data"],
                            opp_team_data=sel_dict["opp_team_data"],
                            selected_indices=sel_dict["selected_indices"],
                            lead_index=sel_dict["lead_index"],
                            winner=sel_dict["winner"],
                            perspective=sel_dict["perspective"],
                        )
                        all_selection_data.append(sel_ex)

                except Exception as e:
                    print(f"  Worker error: {e}")
                    continue

        return all_examples, all_pbs_data, all_selection_data, wins

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
        start_time = time.time()

        all_examples = []
        all_pbs_data: list[tuple[PublicBeliefState, float, float]] = []
        all_selection_data: list[SelectionExample] = []
        wins = {0: 0, 1: 0, None: 0}

        num_workers = self.config.num_workers
        num_games = self.config.games_per_iteration

        if num_workers > 1:
            # 並列実行
            all_examples, all_pbs_data, all_selection_data, wins = self._generate_games_parallel(
                iteration, num_games, num_workers
            )
        else:
            # 逐次実行（従来通り）
            for i in range(num_games):
                game_id = f"iter{iteration}_game{i}"
                result, pbs_data, selection_data = self._generate_game_with_pbs(game_id)
                all_examples.extend(result.examples)
                all_pbs_data.extend(pbs_data)
                all_selection_data.extend(selection_data)
                wins[result.winner] = wins.get(result.winner, 0) + 1

        elapsed = time.time() - start_time
        games_per_sec = num_games / elapsed if elapsed > 0 else 0
        print(f"  Generated {len(all_examples)} examples from {num_games} games in {elapsed:.1f}s ({games_per_sec:.2f} games/s)")
        print(f"  Wins: P0={wins[0]}, P1={wins[1]}, Draw={wins[None]}")
        if self.config.train_selection:
            print(f"  Selection examples: {len(all_selection_data)}")

        # 有効なデータのみ抽出
        valid_data = [(pbs, my_v, opp_v) for pbs, my_v, opp_v in all_pbs_data if my_v is not None]
        if len(valid_data) == 0:
            print("  No valid examples, skipping training")
            return {"iteration": iteration, "examples": 0}

        # 並列実行時はPBSがdictになっているため、PublicBeliefStateに変換
        if num_workers > 1:
            print(f"  Converting {len(valid_data)} PBS dicts to objects...")
            converted_data: list[tuple[PublicBeliefState, float, float]] = []
            conversion_errors = 0
            for pbs_or_dict, my_v, opp_v in valid_data:
                if isinstance(pbs_or_dict, dict):
                    try:
                        pbs = PublicBeliefState.from_dict(pbs_or_dict, self.usage_db)
                        converted_data.append((pbs, my_v, opp_v))
                    except Exception as e:
                        conversion_errors += 1
                        if conversion_errors <= 3:
                            print(f"    Conversion error: {e}")
                else:
                    converted_data.append((pbs_or_dict, my_v, opp_v))
            if conversion_errors > 0:
                print(f"  Warning: {conversion_errors} PBS conversions failed")
            valid_data = converted_data
            print(f"  Converted {len(valid_data)} PBS objects successfully")

        # 学習
        print(f"  Training on {len(valid_data)} examples...")
        total_loss = 0.0
        num_batches = 0
        batch_errors = 0

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
                    batch_errors += 1
                    if batch_errors <= 3:  # 最初の3回だけ表示
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
                avg_epoch_loss = epoch_loss / max(1, num_batches // self.config.num_epochs)
                print(f"    Epoch {epoch + 1}: loss = {avg_epoch_loss:.4f}")

            total_loss += epoch_loss

        if batch_errors > 0:
            print(f"  Warning: {batch_errors} batches skipped due to encoding errors")

        avg_loss = total_loss / max(num_batches, 1)
        print(f"  Value Network Average loss: {avg_loss:.4f} ({num_batches} batches)")

        # 選出ネットワークの学習
        selection_loss = 0.0
        if self.config.train_selection and len(all_selection_data) > 0:
            selection_loss = self._train_selection_network(all_selection_data)
            print(f"  Selection Network Average loss: {selection_loss:.4f}")

        stats = {
            "iteration": iteration,
            "examples": len(all_examples),
            "games": self.config.games_per_iteration,
            "avg_loss": avg_loss,
            "selection_loss": selection_loss,
            "wins_p0": wins[0],
            "wins_p1": wins[1],
            "draws": wins[None],
        }
        self.training_history.append(stats)

        return stats

    def _train_selection_network(self, selection_data: list[SelectionExample]) -> float:
        """
        選出ネットワークを学習（先発予測も含む）

        Args:
            selection_data: 選出データのリスト

        Returns:
            平均損失
        """
        if self.selection_network is None or self.selection_optimizer is None:
            return 0.0

        self.selection_network.train()
        device = torch.device(self.config.device)

        total_loss = 0.0
        num_batches = 0

        random.shuffle(selection_data)

        for epoch in range(self.config.num_epochs):
            for batch_start in range(0, len(selection_data), self.config.batch_size):
                batch = selection_data[batch_start:batch_start + self.config.batch_size]
                if len(batch) == 0:
                    continue

                # バッチデータをテンソルに変換
                my_teams = []
                opp_teams = []
                selection_targets = []
                lead_targets = []
                selection_masks = []
                values = []

                for ex in batch:
                    my_tensor = self.selection_encoder.encode_team(ex.my_team_data)
                    opp_tensor = self.selection_encoder.encode_team(ex.opp_team_data)
                    my_teams.append(my_tensor)
                    opp_teams.append(opp_tensor)

                    # 選出ラベル（選ばれた3匹は1、それ以外は0）
                    sel_target = torch.zeros(6)
                    sel_mask = torch.zeros(6, dtype=torch.bool)
                    for idx in ex.selected_indices:
                        if idx < 6:
                            sel_target[idx] = 1.0
                            sel_mask[idx] = True
                    selection_targets.append(sel_target)
                    selection_masks.append(sel_mask)

                    # 先発ラベル（6匹中のインデックス）
                    lead_targets.append(ex.lead_index)

                    # 勝敗に基づく価値（自分視点で勝ち=1、負け=0）
                    if ex.winner is not None:
                        value = 1.0 if ex.winner == ex.perspective else 0.0
                    else:
                        value = 0.5
                    values.append(value)

                my_batch = torch.stack(my_teams).to(device)
                opp_batch = torch.stack(opp_teams).to(device)
                selection_target_batch = torch.stack(selection_targets).to(device)
                selection_mask_batch = torch.stack(selection_masks).to(device)
                lead_target_batch = torch.tensor(lead_targets, device=device, dtype=torch.long)
                value_batch = torch.tensor(values, device=device, dtype=torch.float).unsqueeze(1)

                # Forward（選出マスクを渡して先発確率を正しく計算）
                self.selection_optimizer.zero_grad()
                output = self.selection_network(
                    my_batch, opp_batch, selection_mask=selection_mask_batch
                )

                # 選出ロス（Binary Cross Entropy）
                selection_probs = torch.sigmoid(output["selection_logits"])
                selection_loss = F.binary_cross_entropy(selection_probs, selection_target_batch)

                # 先発ロス（Cross Entropy、選出されたポケモン内での分類）
                lead_logits = output["lead_logits"]
                lead_loss = F.cross_entropy(lead_logits, lead_target_batch)

                # 価値ロス（MSE）
                value_loss = F.mse_loss(output["value"], value_batch)

                # 合計ロス
                loss = selection_loss + 0.5 * lead_loss + 0.5 * value_loss

                # Backward
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.selection_network.parameters(), 1.0)
                self.selection_optimizer.step()

                total_loss += loss.item()
                num_batches += 1

        return total_loss / max(num_batches, 1)

    def train(self, num_iterations: int, output_dir: str) -> None:
        """
        学習ループを実行

        Args:
            num_iterations: イテレーション数
            output_dir: 出力ディレクトリ
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # ログファイルのパス
        log_file_path = output_path / "training_log.jsonl"

        for iteration in range(1, num_iterations + 1):
            stats = self.train_iteration(iteration)

            # 各イテレーションの統計をログファイルに追記
            stats_with_timestamp = {
                **stats,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            with open(log_file_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(stats_with_timestamp, ensure_ascii=False) + "\n")

            # 定期保存
            if iteration % self.config.save_interval == 0:
                self.save(output_path / f"checkpoint_iter{iteration}")

        # 最終保存
        self.save(output_path / "final")

        # 学習履歴を保存
        with open(output_path / "training_history.json", "w") as f:
            json.dump(self.training_history, f, indent=2)

        print(f"\nTraining log saved to: {log_file_path}")

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

        # 選出ネットワークの保存
        if self.selection_network is not None:
            torch.save(
                self.selection_network.state_dict(), path / "selection_network.pt"
            )
            if self.selection_optimizer is not None:
                torch.save(
                    self.selection_optimizer.state_dict(), path / "selection_optimizer.pt"
                )
            if self.selection_encoder is not None:
                self.selection_encoder.save(path / "selection_encoder.json")

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

        # 選出ネットワークの読み込み
        selection_path = path / "selection_network.pt"
        if selection_path.exists() and self.selection_network is not None:
            self.selection_network.load_state_dict(
                torch.load(selection_path, map_location=self.config.device)
            )
        selection_opt_path = path / "selection_optimizer.pt"
        if selection_opt_path.exists() and self.selection_optimizer is not None:
            self.selection_optimizer.load_state_dict(
                torch.load(selection_opt_path, map_location=self.config.device)
            )
        selection_enc_path = path / "selection_encoder.json"
        if selection_enc_path.exists():
            self.selection_encoder = TeamSelectionEncoder.load(selection_enc_path)

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
            Pokemon.init(usage_data_path=self.config.usage_data_path)
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
