"""
ReBeL Battle Interface

ReBeL を使用したバトル AI のインターフェース。
既存の Battle クラスと統合して使用する。
"""

from __future__ import annotations

from copy import deepcopy
from typing import Optional

from src.hypothesis.pokemon_usage_database import PokemonUsageDatabase
from src.pokemon_battle_sim.battle import Battle

from .belief_state import Observation, ObservationType, PokemonBeliefState
from .cfr_solver import CFRConfig, ReBeLSolver
from .public_state import PublicBeliefState
from .value_network import ReBeLValueNetwork


class ReBeLBattle(Battle):
    """
    ReBeL ベースの AI を使用するバトルクラス

    HypothesisMCTSBattle の代替として使用可能。
    """

    def __init__(
        self,
        usage_db: PokemonUsageDatabase,
        value_network: Optional[ReBeLValueNetwork] = None,
        cfr_iterations: int = 50,
        cfr_world_samples: int = 20,
        use_simplified_cfr: bool = True,
        seed: Optional[int] = None,
    ):
        """
        Args:
            usage_db: ポケモン使用率データベース
            value_network: Value Network（None の場合はヒューリスティック使用）
            cfr_iterations: CFR イテレーション数
            cfr_world_samples: ワールドサンプル数
            use_simplified_cfr: 簡略化 CFR を使用
            seed: 乱数シード
        """
        super().__init__(seed=seed)

        self.usage_db = usage_db
        self.value_network = value_network

        # CFR ソルバー
        cfr_config = CFRConfig(
            num_iterations=cfr_iterations,
            num_world_samples=cfr_world_samples,
        )
        self.solver = ReBeLSolver(
            value_network=value_network,
            cfr_config=cfr_config,
            use_simplified=use_simplified_cfr,
        )

        # 各プレイヤーの信念状態
        self.beliefs: dict[int, PokemonBeliefState] = {}

        # 観測履歴
        self.observation_log: list[Observation] = []

    def init_belief_state(self, player: int) -> PokemonBeliefState:
        """
        プレイヤーの信念状態を初期化

        Args:
            player: プレイヤー番号

        Returns:
            初期化された信念状態
        """
        opponent = 1 - player
        opponent_names = [p.name for p in self.selected[opponent]]

        belief = PokemonBeliefState(
            opponent_pokemon_names=opponent_names,
            usage_db=self.usage_db,
        )
        self.beliefs[player] = belief
        return belief

    def get_belief_state(self, player: int) -> Optional[PokemonBeliefState]:
        """信念状態を取得"""
        return self.beliefs.get(player)

    def update_belief(self, player: int, observation: Observation) -> None:
        """
        観測に基づいて信念を更新

        Args:
            player: 更新するプレイヤー
            observation: 観測イベント
        """
        if player in self.beliefs:
            self.beliefs[player].update(observation)
        self.observation_log.append(observation)

    def battle_command(self, player: int) -> int:
        """
        ReBeL で行動を選択

        Args:
            player: プレイヤー番号

        Returns:
            選択したコマンド
        """
        # 信念状態の初期化（必要なら）
        if player not in self.beliefs:
            self.init_belief_state(player)

        # PBS 構築
        pbs = PublicBeliefState.from_battle(self, player, self.beliefs[player])

        # CFR で行動選択
        return self.solver.get_action(pbs, self, explore=False)

    def change_command(self, player: int) -> int:
        """
        交代先を選択

        Args:
            player: プレイヤー番号

        Returns:
            選択したコマンド
        """
        # 交代可能なポケモンを取得
        available = self.available_commands(player, phase="change")

        if not available:
            return Battle.SKIP

        if len(available) == 1:
            return available[0]

        # 信念状態の初期化（必要なら）
        if player not in self.beliefs:
            self.init_belief_state(player)

        # PBS 構築
        pbs = PublicBeliefState.from_battle(self, player, self.beliefs[player])

        # 簡略化: 交代先は HP 比率が高いポケモンを優先
        best_switch = available[0]
        best_hp = 0.0

        for cmd in available:
            if cmd >= 20:
                idx = cmd - 20
                if idx < len(self.selected[player]):
                    pokemon = self.selected[player][idx]
                    hp_ratio = pokemon.hp / pokemon.status[0] if pokemon.status[0] > 0 else 0
                    if hp_ratio > best_hp:
                        best_hp = hp_ratio
                        best_switch = cmd

        return best_switch

    def get_policy_value(
        self, player: int, phase: str = "battle"
    ) -> tuple[dict[int, float], float]:
        """
        現在の盤面での Policy と Value を取得

        Args:
            player: プレイヤー番号
            phase: "battle" または "change"

        Returns:
            (policy, value): 戦略と期待値
        """
        if player not in self.beliefs:
            self.init_belief_state(player)

        pbs = PublicBeliefState.from_battle(self, player, self.beliefs[player])

        # CFR で戦略計算
        my_strategy, _ = self.solver.solve(pbs, self)

        # 価値は CFR で計算された期待値
        # 簡略化: ヒューリスティック値を使用
        from .cfr_solver import default_value_estimator
        value = default_value_estimator(self, player)

        return my_strategy, value

    def auto_observe_move(self, player: int, move_name: str, pokemon_name: str) -> None:
        """
        技使用の観測を自動登録

        Args:
            player: 観測するプレイヤー（この player から見た相手の技）
            move_name: 使用された技名
            pokemon_name: 技を使用したポケモン名
        """
        obs = Observation(
            type=ObservationType.MOVE_USED,
            pokemon_name=pokemon_name,
            details={"move": move_name},
        )
        self.update_belief(player, obs)

    def auto_observe_item(
        self, player: int, item_name: str, pokemon_name: str, revealed_type: ObservationType
    ) -> None:
        """
        持ち物関連の観測を自動登録

        Args:
            player: 観測するプレイヤー
            item_name: 判明した持ち物名
            pokemon_name: 対象ポケモン名
            revealed_type: 観測タイプ
        """
        obs = Observation(
            type=revealed_type,
            pokemon_name=pokemon_name,
            details={"item": item_name},
        )
        self.update_belief(player, obs)

    def auto_observe_tera(
        self, player: int, tera_type: str, pokemon_name: str
    ) -> None:
        """
        テラスタル観測を自動登録

        Args:
            player: 観測するプレイヤー
            tera_type: テラスタイプ
            pokemon_name: 対象ポケモン名
        """
        obs = Observation(
            type=ObservationType.TERASTALLIZED,
            pokemon_name=pokemon_name,
            details={"tera_type": tera_type},
        )
        self.update_belief(player, obs)

    def auto_observe_ability(
        self, player: int, ability: str, pokemon_name: str
    ) -> None:
        """
        特性発動観測を自動登録

        Args:
            player: 観測するプレイヤー
            ability: 発動した特性名
            pokemon_name: 対象ポケモン名
        """
        obs = Observation(
            type=ObservationType.ABILITY_REVEALED,
            pokemon_name=pokemon_name,
            details={"ability": ability},
        )
        self.update_belief(player, obs)


def load_rebel_battle(
    usage_db_path: str,
    value_network_path: Optional[str] = None,
    **kwargs,
) -> ReBeLBattle:
    """
    ReBeLBattle をロード

    Args:
        usage_db_path: 使用率データベースのパス
        value_network_path: Value Network のパス（オプション）
        **kwargs: ReBeLBattle の追加引数

    Returns:
        ReBeLBattle インスタンス
    """
    import torch

    usage_db = PokemonUsageDatabase.from_json(usage_db_path)

    value_network = None
    if value_network_path:
        value_network = ReBeLValueNetwork()
        value_network.load_state_dict(
            torch.load(value_network_path, map_location="cpu")
        )
        value_network.eval()

    return ReBeLBattle(
        usage_db=usage_db,
        value_network=value_network,
        **kwargs,
    )


# ============================================================
# HypothesisMCTS との互換レイヤー
# ============================================================


class ReBeLMCTSAdapter:
    """
    ReBeL を HypothesisMCTS のインターフェースで使用するアダプター

    既存のコードとの互換性を保つ。
    """

    def __init__(
        self,
        usage_db: PokemonUsageDatabase,
        cfr_iterations: int = 50,
        cfr_world_samples: int = 20,
    ):
        self.usage_db = usage_db
        cfr_config = CFRConfig(
            num_iterations=cfr_iterations,
            num_world_samples=cfr_world_samples,
        )
        self.solver = ReBeLSolver(
            value_network=None,
            cfr_config=cfr_config,
            use_simplified=True,
        )

    def search(
        self,
        battle: Battle,
        player: int,
        belief_state: PokemonBeliefState,
        phase: str = "battle",
    ) -> "PolicyValue":
        """
        HypothesisMCTS.search() と同じインターフェース
        """
        from src.hypothesis.hypothesis_mcts import PolicyValue

        pbs = PublicBeliefState(
            public_state=None,  # 簡略化
            belief=belief_state,
        )
        # PBS を完全に構築
        pbs = PublicBeliefState.from_battle(battle, player, belief_state)

        my_strategy, _ = self.solver.solve(pbs, battle)

        # 価値推定
        from .cfr_solver import default_value_estimator
        value = default_value_estimator(battle, player)

        return PolicyValue(policy=my_strategy, value=value)

    def get_best_action(
        self,
        battle: Battle,
        player: int,
        belief_state: PokemonBeliefState,
        phase: str = "battle",
    ) -> int:
        """
        HypothesisMCTS.get_best_action() と同じインターフェース
        """
        pv = self.search(battle, player, belief_state, phase)
        if not pv.policy:
            return Battle.SKIP
        return max(pv.policy.items(), key=lambda x: x[1])[0]
