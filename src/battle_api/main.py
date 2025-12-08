"""
FastAPI server for human vs ReBeL AI battles.
"""

from __future__ import annotations

import json
import random
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.hypothesis.pokemon_usage_database import PokemonUsageDatabase
from src.pokemon_battle_sim.battle import Battle
from src.pokemon_battle_sim.pokemon import Pokemon
from src.policy_value_network.team_selection_encoder import TeamSelectionEncoder
from src.policy_value_network.team_selection_network import (
    TeamSelectionNetwork,
    TeamSelectionNetworkConfig,
)
from src.rebel import (
    CFRConfig,
    FullBeliefState,
    PokemonBeliefState,
    PublicBeliefState,
    ReBeLSolver,
    ReBeLValueNetwork,
    TeamCompositionBelief,
)
from src.rebel.belief_state import Observation, ObservationType
from src.rebel.cfr_solver import default_value_estimator


# ============================================================
# Custom Battle Class for Interactive Play
# ============================================================


class InteractiveBattle(Battle):
    """Battle subclass that tracks when player needs to choose a switch pokemon."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.player_needs_switch = False
        self._prev_player_pokemon = None

    def change_command(self, player: int) -> int:
        """Override to track when player 0 needs to switch after fainting."""
        if player == 0:
            # Mark that player needs to make a switch choice
            self.player_needs_switch = True
        # Use default behavior (random selection) but track it
        return super().change_command(player)


# ============================================================
# Item Observation Detection from Battle Log
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


def update_belief_from_battle_log(
    session: "BattleSession",
    prev_log_lengths: dict[int, int],
) -> dict[int, int]:
    """
    バトルログの変化を検出して信念を更新

    Args:
        session: バトルセッション
        prev_log_lengths: 前回のログ長さ {player: length}

    Returns:
        更新後のログ長さ
    """
    if not session.rebel_belief:
        return prev_log_lengths

    battle = session.battle
    new_log_lengths = {}

    for player in [0, 1]:
        current_log = battle.log[player] if hasattr(battle, 'log') else []
        prev_length = prev_log_lengths.get(player, 0)
        new_log_lengths[player] = len(current_log)

        if len(current_log) <= prev_length:
            continue

        # 新しいログエントリを取得
        new_entries = current_log[prev_length:]
        pokemon = battle.pokemon[player]

        if pokemon:
            # 相手（player=1）のログから持ち物観測を抽出
            if player == 1:
                observations = extract_item_observations_from_log(
                    new_entries, pokemon.name
                )
                for obs in observations:
                    session.rebel_belief.update(obs)

    return new_log_lengths


# ============================================================
# Pydantic Models
# ============================================================


class PokemonDataModel(BaseModel):
    name: str
    item: str = ""
    ability: str = ""
    moves: list[str] = []
    tera_type: str = "ノーマル"
    nature: str = "まじめ"
    evs: Optional[dict[str, int]] = None


class CreateBattleRequest(BaseModel):
    player_team: list[PokemonDataModel]
    ai_trainer_index: Optional[int] = None


class SelectionRequest(BaseModel):
    session_id: str
    selected_indices: list[int]


class ActionModel(BaseModel):
    type: str  # "move", "switch", "terastallize"
    index: int
    name: str


class ActionRequest(BaseModel):
    session_id: str
    action: ActionModel


# ============================================================
# Battle Session Manager
# ============================================================


# TOD (Time Over Death) settings
TOD_TIME_LIMIT_SECONDS = 10 * 60  # 10 minutes
AI_SURRENDER_THRESHOLD = 0.05  # AI surrenders if win probability < 5%


@dataclass
class BattleSession:
    """Manages a single battle session."""

    session_id: str
    battle: InteractiveBattle
    player_team_data: list[dict]  # Original team data
    ai_team_data: list[dict]
    player_pokemon: list[Pokemon]  # Created Pokemon objects for player
    ai_pokemon: list[Pokemon]  # Created Pokemon objects for AI
    rebel_solver: ReBeLSolver
    rebel_belief: Optional[PokemonBeliefState] = None
    full_belief: Optional[FullBeliefState] = None  # Full belief with selection/lead uncertainty
    log: list[dict] = field(default_factory=list)
    turn: int = 0
    phase: str = "selection"  # "selection", "battle", "change", "finished"
    pending_switch_pokemon: Optional[str] = None  # Name of pokemon that auto-switched
    fainted_pokemon_name: Optional[str] = None  # Name of pokemon that fainted
    created_at: datetime = field(default_factory=datetime.now)  # For TOD tracking
    # バトルログの長さ追跡（持ち物観測検出用）
    battle_log_lengths: dict = field(default_factory=lambda: {0: 0, 1: 0})
    last_ai_value: Optional[float] = None  # AI's estimated win probability
    ai_selection_probs: Optional[dict] = None  # AI's selection/lead probabilities for display


class SessionManager:
    """Manages all battle sessions."""

    def __init__(self):
        self.sessions: dict[str, BattleSession] = {}
        self.usage_db: Optional[PokemonUsageDatabase] = None
        self.value_network: Optional[ReBeLValueNetwork] = None
        self.selection_network: Optional[TeamSelectionNetwork] = None
        self.selection_encoder: Optional[TeamSelectionEncoder] = None
        self.trainer_data: list[dict] = []
        self.player_party: list[dict] = []  # Fixed player party

    def load_resources(
        self,
        usage_db_path: str,
        trainer_json_path: str,
        checkpoint_path: Optional[str] = None,
        player_party_path: Optional[str] = None,
    ):
        """Load required resources."""
        Pokemon.init()

        self.usage_db = PokemonUsageDatabase.from_json(usage_db_path)

        with open(trainer_json_path, "r", encoding="utf-8") as f:
            self.trainer_data = json.load(f)

        # Load player's fixed party
        if player_party_path and Path(player_party_path).exists():
            with open(player_party_path, "r", encoding="utf-8") as f:
                party_data = json.load(f)
            # Convert format
            self.player_party = self._convert_party_format(party_data.get("pokemons", []))
            print(f"Loaded player party: {[p['name'] for p in self.player_party]}")

        if checkpoint_path:
            checkpoint_dir = Path(checkpoint_path)

            # Load value network
            value_network_path = checkpoint_dir / "value_network.pt"
            if value_network_path.exists():
                self.value_network = ReBeLValueNetwork(
                    hidden_dim=256,
                    num_res_blocks=4,
                )
                self.value_network.load_state_dict(
                    torch.load(
                        value_network_path, map_location="cpu", weights_only=True
                    )
                )
                self.value_network.eval()
                print(f"Loaded value network from {value_network_path}")

            # Load selection network
            selection_network_path = checkpoint_dir / "selection_network.pt"
            selection_encoder_path = checkpoint_dir / "selection_encoder.json"
            if selection_network_path.exists() and selection_encoder_path.exists():
                self.selection_encoder = TeamSelectionEncoder.load(
                    selection_encoder_path
                )
                self.selection_network = TeamSelectionNetwork(
                    TeamSelectionNetworkConfig(pokemon_feature_dim=15)
                )
                self.selection_network.load_state_dict(
                    torch.load(
                        selection_network_path, map_location="cpu", weights_only=True
                    )
                )
                self.selection_network.eval()
                print(f"Loaded selection network from {selection_network_path}")

    def _convert_party_format(self, pokemons: list[dict]) -> list[dict]:
        """Convert party format from my_fixed_party.json to API format."""
        converted = []
        ev_keys = ["hp", "attack", "defense", "sp_attack", "sp_defense", "speed"]
        for poke in pokemons:
            evs = {}
            if "effort" in poke:
                for i, val in enumerate(poke["effort"]):
                    if i < len(ev_keys):
                        evs[ev_keys[i]] = val
            converted.append({
                "name": poke.get("name", ""),
                "item": poke.get("item", ""),
                "ability": poke.get("ability", ""),
                "moves": poke.get("moves", []),
                "tera_type": poke.get("Ttype", poke.get("tera_type", "ノーマル")),
                "nature": poke.get("nature", "まじめ"),
                "evs": evs if evs else None,
            })
        return converted

    def select_team_with_network(
        self,
        my_team_data: list[dict],
        opp_team_data: list[dict],
        num_select: int = 3,
    ) -> tuple[list[int], Optional[dict]]:
        """
        選出ネットワークを使用してチームを選出（先発も決定）

        Returns:
            (選出するインデックスのリスト（先発が最初）, 確率情報)
        """
        if self.selection_network is None or self.selection_encoder is None:
            # ネットワークがない場合はランダム選出
            available = list(range(len(my_team_data)))
            selected = random.sample(available, min(num_select, len(available)))
            return selected, None

        self.selection_network.eval()
        with torch.no_grad():
            my_tensor = self.selection_encoder.encode_team(my_team_data)
            opp_tensor = self.selection_encoder.encode_team(opp_team_data)

            my_tensor = my_tensor.unsqueeze(0)
            opp_tensor = opp_tensor.unsqueeze(0)

            indices, selection_probs, lead_index, lead_probs = self.selection_network.select_team(
                my_tensor, opp_tensor, num_select=num_select, deterministic=True
            )

        # 確率情報を辞書で返す
        probs_info = {
            "selection_probs": selection_probs[0].tolist()[:len(my_team_data)],
            "lead_probs": lead_probs[0].tolist()[:len(my_team_data)],
            "lead_index": lead_index[0].item(),
        }

        return indices[0].tolist(), probs_info

    def create_session(
        self,
        player_team: list[dict],
        ai_trainer_index: Optional[int] = None,
    ) -> BattleSession:
        """Create a new battle session."""
        session_id = str(uuid.uuid4())

        # Select AI trainer
        if ai_trainer_index is not None and 0 <= ai_trainer_index < len(
            self.trainer_data
        ):
            ai_trainer = self.trainer_data[ai_trainer_index]
        else:
            ai_trainer = random.choice(self.trainer_data)

        ai_team_data = ai_trainer.get("pokemons", [])[:6]

        # Create battle
        battle = InteractiveBattle()
        battle.reset_game()

        # Create player's Pokemon objects
        player_pokemon_list: list[Pokemon] = []
        for poke_data in player_team[:6]:
            pokemon = Pokemon(poke_data.get("name", "ピカチュウ"))
            pokemon.item = poke_data.get("item", "")
            pokemon.ability = poke_data.get("ability", "")
            pokemon.moves = poke_data.get("moves", [])[:4]
            # Support both "Ttype" (trainer data) and "tera_type" (API request)
            pokemon.Ttype = poke_data.get("Ttype", poke_data.get("tera_type", "ノーマル"))
            if poke_data.get("nature"):
                pokemon.nature = poke_data["nature"]
            if poke_data.get("evs"):
                pokemon.effort_value = poke_data["evs"]
            player_pokemon_list.append(pokemon)

        # Create AI's Pokemon objects
        ai_pokemon_list: list[Pokemon] = []
        for poke_data in ai_team_data[:6]:
            pokemon = Pokemon(poke_data.get("name", "ピカチュウ"))
            pokemon.item = poke_data.get("item", "")
            pokemon.ability = poke_data.get("ability", "")
            pokemon.moves = poke_data.get("moves", [])[:4]
            # Support both "Ttype" (trainer data) and "tera_type" (API request)
            pokemon.Ttype = poke_data.get("Ttype", poke_data.get("tera_type", "ノーマル"))
            if poke_data.get("nature"):
                pokemon.nature = poke_data["nature"]
            if poke_data.get("evs"):
                pokemon.effort_value = poke_data["evs"]
            ai_pokemon_list.append(pokemon)

        # Create ReBeL solver
        cfr_config = CFRConfig(
            num_iterations=30,
            num_world_samples=10,
        )
        rebel_solver = ReBeLSolver(
            value_network=self.value_network,
            cfr_config=cfr_config,
            use_simplified=True,
        )

        session = BattleSession(
            session_id=session_id,
            battle=battle,
            player_team_data=player_team,
            ai_team_data=ai_team_data,
            player_pokemon=player_pokemon_list,
            ai_pokemon=ai_pokemon_list,
            rebel_solver=rebel_solver,
            phase="selection",
        )

        self.sessions[session_id] = session
        return session

    def get_session(self, session_id: str) -> BattleSession:
        """Get a session by ID."""
        if session_id not in self.sessions:
            raise HTTPException(status_code=404, detail="Session not found")
        return self.sessions[session_id]


# Global session manager
session_manager = SessionManager()


# ============================================================
# Lifespan
# ============================================================


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load resources on startup."""
    base_path = Path(__file__).parent.parent.parent

    usage_db_path = base_path / "data" / "pokedb_usage" / "season_37_top150.json"
    trainer_json_path = base_path / "data" / "top_rankers" / "season_36.json"
    checkpoint_path = (
        base_path / "models" / "rebel_selection_vs_my_party" / "checkpoint_iter25"
    )
    player_party_path = base_path / "data" / "my_fixed_party.json"

    session_manager.load_resources(
        str(usage_db_path),
        str(trainer_json_path),
        str(checkpoint_path) if checkpoint_path.exists() else None,
        str(player_party_path) if player_party_path.exists() else None,
    )

    print("Battle API ready!")
    yield


# ============================================================
# FastAPI App
# ============================================================

app = FastAPI(
    title="Pokemon Battle API",
    description="API for human vs ReBeL AI battles",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# Helper Functions
# ============================================================


def pokemon_to_state(
    pokemon: Optional[Pokemon], reveal_all: bool = False
) -> Optional[dict]:
    """Convert Pokemon to state dict."""
    if pokemon is None:
        return None

    max_hp = pokemon.status[0]  # status is [H,A,B,C,D,S]
    state = {
        "name": pokemon.name,
        "hp_ratio": (
            pokemon.hp / max_hp
            if max_hp > 0
            else 0
        ),
        "max_hp": max_hp,
        "current_hp": pokemon.hp,
        "status": pokemon.ailment if pokemon.ailment else None,  # ailment is status condition
        "item": pokemon.item if reveal_all else None,
        "is_terastallized": pokemon.terastal,
        "tera_type": pokemon.Ttype if reveal_all or pokemon.terastal else None,
        "types": pokemon.types,
        "ability": pokemon.ability if reveal_all else None,
        "stat_changes": {
            # rank is [H,A,B,C,D,S,命中,回避], index 1-5 are combat stats
            "attack": pokemon.rank[1],
            "defense": pokemon.rank[2],
            "sp_attack": pokemon.rank[3],
            "sp_defense": pokemon.rank[4],
            "speed": pokemon.rank[5],
        },
    }

    if reveal_all:
        state["moves"] = pokemon.moves
    else:
        # Only show revealed moves
        state["moves"] = getattr(pokemon, "revealed_moves", [])

    return state


def get_field_state(battle: Battle) -> dict:
    """Get field state."""
    # Get weather from battle.weather() method or condition dict
    weather = None
    # Only call weather() if pokemon are set (battle has started)
    if (hasattr(battle, "weather") and callable(battle.weather) and
        hasattr(battle, "pokemon") and battle.pokemon[0] is not None and battle.pokemon[1] is not None):
        try:
            w = battle.weather()
            if w:
                weather = w
        except (AttributeError, TypeError):
            pass

    # Get terrain from condition dict (fields)
    terrain = None
    if hasattr(battle, "condition"):
        cond = battle.condition
        # Check fields
        if cond.get("elecfield", 0) > 0:
            terrain = "エレキフィールド"
        elif cond.get("glassfield", 0) > 0:
            terrain = "グラスフィールド"
        elif cond.get("psycofield", 0) > 0:
            terrain = "サイコフィールド"
        elif cond.get("mistfield", 0) > 0:
            terrain = "ミストフィールド"

    # Get side conditions from battle.condition
    cond = getattr(battle, "condition", {})

    return {
        "weather": weather,
        "terrain": terrain,
        "player_side": {
            "stealth_rock": bool(cond.get("stealthrock", [0, 0])[0]),
            "spikes": cond.get("makibishi", [0, 0])[0],
            "toxic_spikes": cond.get("dokubishi", [0, 0])[0],
            "sticky_web": bool(cond.get("nebanet", [0, 0])[0]),
            "reflect": cond.get("reflector", [0, 0])[0],
            "light_screen": cond.get("lightwall", [0, 0])[0],
            "tailwind": cond.get("oikaze", [0, 0])[0],
        },
        "opponent_side": {
            "stealth_rock": bool(cond.get("stealthrock", [0, 0])[1]),
            "spikes": cond.get("makibishi", [0, 0])[1],
            "toxic_spikes": cond.get("dokubishi", [0, 0])[1],
            "sticky_web": bool(cond.get("nebanet", [0, 0])[1]),
            "reflect": cond.get("reflector", [0, 0])[1],
            "light_screen": cond.get("lightwall", [0, 0])[1],
            "tailwind": cond.get("oikaze", [0, 0])[1],
        },
    }


def get_available_actions(
    battle: Battle, player: int, phase: str, fainted_pokemon_name: Optional[str] = None
) -> list[dict]:
    """Get available actions for a player."""
    actions = []

    if phase == "finished":
        return actions

    # For change phase after fainting, show all alive pokemon except the fainted one
    # Include the auto-switched pokemon (marked as current) so user can confirm it
    if phase == "change" and fainted_pokemon_name:
        current_pokemon = battle.pokemon[player]
        current_pokemon_name = current_pokemon.name if current_pokemon else None
        for i, pokemon in enumerate(battle.selected[player]):
            if pokemon and pokemon.hp > 0 and pokemon.name != fainted_pokemon_name:
                is_current = pokemon.name == current_pokemon_name
                actions.append(
                    {
                        "type": "switch",
                        "index": i,
                        "name": pokemon.name + ("（現在）" if is_current else ""),
                        "is_current": is_current,
                    }
                )
        return actions

    available = battle.available_commands(player, phase)

    active_pokemon = battle.pokemon[player]

    # Check if terastallization is available
    can_tera = False
    if phase == "battle" and hasattr(battle, "can_terastal"):
        try:
            can_tera = battle.can_terastal(player)
        except Exception:
            can_tera = False

    for cmd in available:
        if 0 <= cmd <= 3:
            # Move
            if active_pokemon and cmd < len(active_pokemon.moves):
                move_name = active_pokemon.moves[cmd]
                # Get PP info
                current_pp = active_pokemon.pp[cmd] if cmd < len(active_pokemon.pp) else 0
                max_pp = Pokemon.all_moves.get(move_name, {}).get("pp", 0)
                actions.append(
                    {
                        "type": "move",
                        "index": cmd,
                        "name": move_name,
                        "pp": current_pp,
                        "max_pp": max_pp,
                    }
                )
                # Add terastallize + move option if available
                if can_tera:
                    actions.append(
                        {
                            "type": "terastallize",
                            "index": cmd,
                            "name": f"{move_name}（テラスタル）",
                            "tera_type": active_pokemon.Ttype if active_pokemon else "???",
                            "pp": current_pp,
                            "max_pp": max_pp,
                        }
                    )
        elif 20 <= cmd <= 25:
            # Switch
            switch_idx = cmd - 20
            if switch_idx < len(battle.selected[player]):
                target_pokemon = battle.selected[player][switch_idx]
                if target_pokemon and target_pokemon.hp > 0:
                    actions.append(
                        {
                            "type": "switch",
                            "index": switch_idx,
                            "name": target_pokemon.name,
                        }
                    )
        elif cmd == 30:
            # Struggle
            actions.append(
                {
                    "type": "move",
                    "index": 30,
                    "name": "わるあがき",
                }
            )

    return actions


def build_battle_state(session: BattleSession, include_log: bool = True) -> dict:
    """Build the full battle state for API response."""
    battle = session.battle

    # Determine phase
    phase = session.phase
    winner = None

    # Only check winner if battle has started (selected pokemon exist)
    if phase not in ["selection"] and len(battle.selected[0]) > 0 and len(battle.selected[1]) > 0:
        try:
            winner = battle.winner()
        except ZeroDivisionError:
            winner = None

    if winner is not None:
        phase = "finished"

    # Get active and bench pokemon
    player_active = None
    player_bench = []
    opponent_active = None
    opponent_bench = []

    if phase not in ["selection"]:
        player_active = pokemon_to_state(battle.pokemon[0], reveal_all=True)
        opponent_active = pokemon_to_state(battle.pokemon[1], reveal_all=False)

        for i, pokemon in enumerate(battle.selected[0]):
            if pokemon and pokemon != battle.pokemon[0]:
                player_bench.append(pokemon_to_state(pokemon, reveal_all=True))

        for i, pokemon in enumerate(battle.selected[1]):
            if pokemon and pokemon != battle.pokemon[1]:
                opponent_bench.append(pokemon_to_state(pokemon, reveal_all=False))

    # Get available actions
    if phase == "selection":
        available_actions = []
    elif phase == "change":
        # Pass fainted pokemon name to show all alive pokemon except the fainted one
        available_actions = get_available_actions(
            battle, 0, phase, fainted_pokemon_name=session.fainted_pokemon_name
        )
    elif phase == "battle":
        available_actions = get_available_actions(battle, 0, phase)
    else:
        available_actions = []

    # Calculate remaining time for TOD
    elapsed_seconds = (datetime.now() - session.created_at).total_seconds()
    remaining_seconds = max(0, TOD_TIME_LIMIT_SECONDS - elapsed_seconds)

    state = {
        "session_id": session.session_id,
        "turn": session.turn,
        "phase": phase,
        "winner": winner,
        "player_active": player_active,
        "player_bench": player_bench,
        "player_team": session.player_team_data,
        "opponent_active": opponent_active,
        "opponent_bench": opponent_bench,
        "opponent_team_preview": [
            {
                "name": p.get("name", "?"),
                "index": i,
                "types": session.ai_pokemon[i].types if i < len(session.ai_pokemon) else [],
            }
            for i, p in enumerate(session.ai_team_data)
        ],
        "field": get_field_state(battle),
        "available_actions": available_actions,
        "remaining_seconds": remaining_seconds,
        "time_limit_seconds": TOD_TIME_LIMIT_SECONDS,
    }

    if include_log:
        state["log"] = session.log

    return state


# ============================================================
# API Endpoints
# ============================================================


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok"}


@app.get("/api/trainers")
async def list_trainers():
    """List available AI trainers."""
    trainers = []
    for i, trainer in enumerate(session_manager.trainer_data[:50]):  # Limit to 50
        pokemons = trainer.get("pokemons", [])
        trainers.append(
            {
                "index": i,
                "name": trainer.get("name", f"Trainer {i}"),
                "pokemon_names": [p.get("name", "?") for p in pokemons[:6]],
            }
        )
    return {"trainers": trainers}


@app.get("/api/player-party")
async def get_player_party():
    """Get the player's fixed party."""
    return {"party": session_manager.player_party}


@app.post("/api/battle/create")
async def create_battle(request: CreateBattleRequest):
    """Create a new battle session."""
    if len(request.player_team) < 1:
        raise HTTPException(status_code=400, detail="Team must have at least 1 pokemon")

    player_team = [p.model_dump() for p in request.player_team]

    session = session_manager.create_session(
        player_team=player_team,
        ai_trainer_index=request.ai_trainer_index,
    )

    return {
        "session_id": session.session_id,
        "state": build_battle_state(session),
    }


@app.post("/api/battle/select")
async def select_pokemon(request: SelectionRequest):
    """Select pokemon for battle (3 from 6)."""
    session = session_manager.get_session(request.session_id)

    if session.phase != "selection":
        raise HTTPException(status_code=400, detail="Not in selection phase")

    if len(request.selected_indices) != 3:
        raise HTTPException(status_code=400, detail="Must select exactly 3 pokemon")

    battle = session.battle

    # Player selection (user's choice)
    for idx in request.selected_indices:
        if idx < 0 or idx >= len(session.player_pokemon):
            raise HTTPException(status_code=400, detail=f"Invalid pokemon index: {idx}")
        battle.selected[0].append(session.player_pokemon[idx])

    # AI selection using NN (with lead prediction)
    ai_selection, ai_probs = session_manager.select_team_with_network(
        session.ai_team_data, session.player_team_data
    )
    session.ai_selection_probs = ai_probs  # Store for display/debugging
    for idx in ai_selection:
        if idx < len(session.ai_pokemon):
            battle.selected[1].append(session.ai_pokemon[idx])

    # Initialize belief state for opponent (player's pokemon from AI's perspective)
    if session_manager.usage_db is not None:
        session.rebel_belief = PokemonBeliefState(
            opponent_pokemon_names=[p.name for p in battle.selected[0]],
            usage_db=session_manager.usage_db,
        )

    # Start battle - send out first pokemon
    battle.proceed(commands=[Battle.SKIP, Battle.SKIP])

    session.phase = "battle"
    session.turn = 1

    player_pokemon_name = battle.pokemon[0].name if battle.pokemon[0] else "???"
    ai_pokemon_name = battle.pokemon[1].name if battle.pokemon[1] else "???"
    session.log.append(
        {
            "turn": 0,
            "messages": [
                "バトル開始！",
                f"あなたは {player_pokemon_name} を繰り出した！",
                f"相手は {ai_pokemon_name} を繰り出した！",
            ],
        }
    )

    return {"state": build_battle_state(session)}


@app.post("/api/battle/action")
async def perform_action(request: ActionRequest):
    """Perform a battle action."""
    session = session_manager.get_session(request.session_id)

    if session.phase not in ["battle", "change"]:
        raise HTTPException(
            status_code=400, detail=f"Cannot perform action in phase: {session.phase}"
        )

    battle = session.battle
    action = request.action

    # Convert action to command
    player_terastal = False
    if action.type == "move":
        player_cmd = action.index
    elif action.type == "switch":
        player_cmd = 20 + action.index
    elif action.type == "terastallize":
        player_cmd = action.index  # Move index
        player_terastal = True
    else:
        raise HTTPException(
            status_code=400, detail=f"Unknown action type: {action.type}"
        )

    # Validate action
    # For change phase, we allow selecting the auto-switched pokemon (to confirm it)
    # so we skip the standard validation and do custom validation
    if session.phase == "change":
        # Validate that the selected pokemon exists and is alive (excluding fainted one)
        switch_idx = action.index
        if switch_idx < 0 or switch_idx >= len(battle.selected[0]):
            raise HTTPException(
                status_code=400, detail=f"Invalid switch index: {switch_idx}"
            )
        target = battle.selected[0][switch_idx]
        if not target or target.hp <= 0:
            raise HTTPException(
                status_code=400, detail="Cannot switch to fainted pokemon"
            )
        if session.fainted_pokemon_name and target.name == session.fainted_pokemon_name:
            raise HTTPException(
                status_code=400, detail="Cannot switch to the pokemon that just fainted"
            )
    else:
        available = battle.available_commands(0, session.phase)
        if player_cmd not in available:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid action. Available: {available}, Got: {player_cmd}",
            )

    # Check TOD (Time Over Death) - 10 minute time limit
    elapsed_seconds = (datetime.now() - session.created_at).total_seconds()
    remaining_seconds = max(0, TOD_TIME_LIMIT_SECONDS - elapsed_seconds)

    if remaining_seconds <= 0 and session.phase != "finished":
        # Time's up - determine winner by HP
        winner = battle.winner(is_timeup=True)
        session.phase = "finished"
        player_score = battle.TOD_score(0)
        ai_score = battle.TOD_score(1)
        session.log.append(
            {
                "turn": session.turn,
                "messages": [
                    "時間切れ！TOD判定に入ります。",
                    f"あなたのTODスコア: {player_score:.2f}",
                    f"AIのTODスコア: {ai_score:.2f}",
                    "勝者: " + ("あなた" if winner == 0 else "AI"),
                ],
            }
        )
        return {
            "state": build_battle_state(session),
            "ai_action": None,
            "ai_thinking_time": 0,
            "tod_triggered": True,
        }

    # AI action
    ai_cmd = Battle.SKIP
    ai_action_info = None
    ai_thinking_time = 0

    if session.phase == "battle":
        start_time = time.time()

        # Calculate AI's estimated win probability
        ai_value = default_value_estimator(battle, 1)  # AI is player 1
        session.last_ai_value = ai_value

        # Check if AI should surrender
        if ai_value < AI_SURRENDER_THRESHOLD:
            session.phase = "finished"
            session.log.append(
                {
                    "turn": session.turn,
                    "messages": [
                        f"AIの推定勝率: {ai_value * 100:.1f}%",
                        "AIは降参した！",
                        "勝者: あなた",
                    ],
                }
            )
            return {
                "state": build_battle_state(session),
                "ai_action": {"type": "surrender"},
                "ai_thinking_time": time.time() - start_time,
                "ai_surrendered": True,
            }

        # Use ReBeL solver
        pbs = PublicBeliefState.from_battle(battle, 1, session.rebel_belief)
        ai_cmd = session.rebel_solver.get_action(pbs, battle, explore=False)

        ai_thinking_time = time.time() - start_time

        # Record AI action for log
        if 0 <= ai_cmd <= 3:
            ai_pokemon = battle.pokemon[1]
            if ai_pokemon and ai_cmd < len(ai_pokemon.moves):
                ai_action_info = {"type": "move", "name": ai_pokemon.moves[ai_cmd]}
        elif 20 <= ai_cmd <= 25:
            switch_idx = ai_cmd - 20
            if switch_idx < len(battle.selected[1]):
                ai_action_info = {
                    "type": "switch",
                    "name": battle.selected[1][switch_idx].name,
                }
        elif ai_cmd == 30:
            ai_action_info = {"type": "move", "name": "わるあがき"}

    elif session.phase == "change":
        # In change phase, player chooses who to send out after fainting
        # The battle engine already auto-switched a pokemon after fainting.
        # If user chooses a different pokemon, we need to swap it.
        switch_idx = action.index
        target_pokemon = battle.selected[0][switch_idx]

        if target_pokemon and target_pokemon.hp > 0:
            current_pokemon = battle.pokemon[0]

            if current_pokemon is None or current_pokemon.name != target_pokemon.name:
                # User chose a different pokemon than the auto-switched one
                # Put the current one back and send out the chosen one
                if current_pokemon is not None:
                    current_pokemon.come_back()

                # Set the new pokemon on the field
                battle.pokemon[0] = target_pokemon
                battle.has_changed[0] = True

                # Register in observed list if not already there
                if Pokemon.find(battle.observed[0], name=target_pokemon.name) is None:
                    obs_pokemon = Pokemon(target_pokemon.name, use_template=False)
                    obs_pokemon.speed_range = [0, 999]
                    battle.observed[0].append(obs_pokemon)

                # Clear any death breakpoint
                if battle.breakpoint[0] == "death":
                    battle.breakpoint[0] = ""

                # Update speed order
                battle.update_speed_order()

                # Process landing effects
                battle.land(0)

            # Add log message
            session.log.append(
                {
                    "turn": session.turn,
                    "messages": [f"{target_pokemon.name} を繰り出した！"],
                }
            )

            # Go back to battle phase
            session.phase = "battle"
            session.turn += 1
            session.pending_switch_pokemon = None
            session.fainted_pokemon_name = None

            return {
                "state": build_battle_state(session),
                "ai_action": None,
                "ai_thinking_time": 0,
            }

    # Execute terastallization before turn if requested
    if player_terastal and battle.pokemon[0]:
        battle.pokemon[0].use_terastal()

    # Record player's pokemon before turn (to detect if it faints)
    prev_player_pokemon = battle.pokemon[0]

    # Reset the switch flag before proceed
    battle.player_needs_switch = False

    # Execute turn
    prev_hp = [
        battle.pokemon[0].hp if battle.pokemon[0] else 0,
        battle.pokemon[1].hp if battle.pokemon[1] else 0,
    ]

    battle.proceed(commands=[player_cmd, ai_cmd])

    # Build log messages
    messages = []
    player_pokemon = battle.pokemon[0]
    ai_pokemon = battle.pokemon[1]

    if action.type == "terastallize":
        tera_type = getattr(player_pokemon, "Ttype", "???") if player_pokemon else "???"
        # Get original move name (remove テラスタル suffix)
        move_name = action.name.replace("（テラスタル）", "")
        messages.append(f"{player_pokemon.name if player_pokemon else '???'} は {tera_type}テラスタルした！")
        messages.append(f"{player_pokemon.name if player_pokemon else '???'} の {move_name}！")
    elif action.type == "move":
        messages.append(
            f"{player_pokemon.name if player_pokemon else '???'} の {action.name}！"
        )
    elif action.type == "switch":
        if session.phase == "change":
            messages.append(f"{action.name} を繰り出した！")
        else:
            messages.append(f"{action.name} に交代！")

    if ai_action_info:
        if ai_action_info["type"] == "move":
            messages.append(
                f"相手の {ai_pokemon.name if ai_pokemon else '???'} の {ai_action_info['name']}！"
            )
        elif ai_action_info["type"] == "switch":
            messages.append(f"相手は {ai_action_info['name']} に交代！")

    session.log.append(
        {
            "turn": session.turn,
            "messages": messages,
        }
    )

    # Update belief based on revealed info
    if ai_action_info and ai_action_info["type"] == "move" and session.rebel_belief:
        # Update belief that AI used this move
        obs = Observation(
            type=ObservationType.MOVE_USED,
            pokemon_name=ai_pokemon.name if ai_pokemon else "",
            details={"move": ai_action_info["name"]},
        )
        session.rebel_belief.update(obs)

    # Update belief from battle log (item activations, berry consumption, etc.)
    session.battle_log_lengths = update_belief_from_battle_log(
        session, session.battle_log_lengths
    )

    # Check for phase changes
    winner = battle.winner()
    if winner is not None:
        session.phase = "finished"
        session.log.append(
            {
                "turn": session.turn,
                "messages": ["勝者: " + ("あなた" if winner == 0 else "AI")],
            }
        )
    elif battle.player_needs_switch:
        # Player's pokemon fainted and was auto-switched
        # Record the auto-switched pokemon so we can let player re-choose
        session.pending_switch_pokemon = battle.pokemon[0].name if battle.pokemon[0] else None
        # Record fainted pokemon name for filtering available switches
        session.fainted_pokemon_name = prev_player_pokemon.name if prev_player_pokemon else None
        session.phase = "change"
        # Add message about fainting
        if prev_player_pokemon and prev_player_pokemon.hp <= 0:
            session.log.append(
                {
                    "turn": session.turn,
                    "messages": [f"{prev_player_pokemon.name} は倒れた！"],
                }
            )
    elif session.phase == "change":
        # Player just switched in change phase - go back to battle
        session.phase = "battle"
        session.turn += 1
        session.pending_switch_pokemon = None
        session.fainted_pokemon_name = None
    else:
        session.phase = "battle"
        session.turn += 1

    return {
        "state": build_battle_state(session),
        "ai_action": ai_action_info,
        "ai_thinking_time": ai_thinking_time,
    }


@app.get("/api/battle/{session_id}")
async def get_battle_state(session_id: str):
    """Get current battle state."""
    session = session_manager.get_session(session_id)
    return {"state": build_battle_state(session)}


class SurrenderRequest(BaseModel):
    session_id: str


@app.post("/api/battle/surrender")
async def surrender_battle(request: SurrenderRequest):
    """Surrender the battle (player loses)."""
    session = session_manager.get_session(request.session_id)

    if session.phase == "finished":
        raise HTTPException(status_code=400, detail="Battle already finished")

    # Set battle as finished with AI as winner
    session.phase = "finished"
    session.log.append(
        {
            "turn": session.turn,
            "messages": ["あなたは降参した！", "勝者: AI"],
        }
    )

    return {
        "state": build_battle_state(session),
        "message": "You surrendered. AI wins.",
    }


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)
