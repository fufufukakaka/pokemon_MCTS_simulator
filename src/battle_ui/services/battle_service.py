"""
バトルサービス - Battleクラスのラッパー
"""

import json
import logging
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.pokemon_battle_sim.battle import Battle
from src.pokemon_battle_sim.pokemon import Pokemon
from src.battle_ui.services.session_manager import BattleSession

logger = logging.getLogger(__name__)


# データファイルパス
DATA_DIR = Path(__file__).resolve().parent.parent.parent.parent / "data"
TRAINER_FILE = DATA_DIR / "top_rankers" / "season_36.json"
MODELS_DIR = Path(__file__).resolve().parent.parent.parent.parent / "models"

# プレイヤーパーティのパス（環境変数から取得）
PLAYER_PARTY_PATH = os.environ.get("PLAYER_PARTY_PATH", None)

# チェックポイントディレクトリ（環境変数から取得）
CHECKPOINT_DIR = os.environ.get(
    "REBEL_CHECKPOINT_DIR",
    str(MODELS_DIR / "revel_full_state_selection_BERT")
)


class BattleService:
    """バトルロジックを提供するサービスクラス"""

    @staticmethod
    def get_checkpoints() -> List[Dict[str, Any]]:
        """利用可能なチェックポイント一覧を取得"""
        checkpoint_dir = Path(CHECKPOINT_DIR)
        if not checkpoint_dir.exists():
            return []

        checkpoints = []
        for item in sorted(checkpoint_dir.iterdir()):
            if item.is_dir() and item.name.startswith("checkpoint_iter"):
                # イテレーション番号を抽出
                try:
                    iter_num = int(item.name.replace("checkpoint_iter", ""))
                except ValueError:
                    continue

                # メタデータを読み込み
                meta_file = item / "checkpoint_meta.json"
                meta = {}
                if meta_file.exists():
                    with open(meta_file, "r", encoding="utf-8") as f:
                        meta = json.load(f)

                checkpoints.append({
                    "name": item.name,
                    "path": str(item),
                    "iteration": iter_num,
                    "win_rate": meta.get("win_rate"),
                    "games_played": meta.get("games_played"),
                })

        # イテレーション番号でソート
        checkpoints.sort(key=lambda x: x["iteration"])
        return checkpoints

    @staticmethod
    def get_trainers() -> List[Dict[str, Any]]:
        """トレーナー一覧を取得"""
        with open(TRAINER_FILE, "r", encoding="utf-8") as f:
            trainers = json.load(f)

        return [
            {
                "index": i,
                "name": t["name"],
                "rank": t["rank"],
                "rating": t["rating"],
                "pokemon_names": [p["name"] for p in t["pokemons"]],
            }
            for i, t in enumerate(trainers[:20])  # 上位20人のみ
        ]

    @staticmethod
    def get_trainer_party(trainer_index: int) -> List[Dict[str, Any]]:
        """指定トレーナーのパーティを取得"""
        with open(TRAINER_FILE, "r", encoding="utf-8") as f:
            trainers = json.load(f)

        if trainer_index < 0 or trainer_index >= len(trainers):
            raise ValueError(f"Invalid trainer index: {trainer_index}")

        return trainers[trainer_index]["pokemons"]

    @staticmethod
    def get_player_party() -> List[Dict[str, Any]]:
        """プレイヤーのパーティを取得"""
        # 環境変数で指定されたパーティファイルがあればそれを使用
        if PLAYER_PARTY_PATH:
            party_path = Path(PLAYER_PARTY_PATH)
            if party_path.exists():
                with open(party_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                # {"pokemons": [...]} または [...] 形式に対応
                if isinstance(data, list):
                    return data
                elif "pokemons" in data:
                    return data["pokemons"]
                else:
                    raise ValueError(f"Invalid party file format: {party_path}")

        # デフォルト：2番目のトレーナーのパーティをプレイヤー用に
        with open(TRAINER_FILE, "r", encoding="utf-8") as f:
            trainers = json.load(f)
        return trainers[1]["pokemons"]

    @staticmethod
    def create_pokemon_from_data(data: Dict[str, Any]) -> Pokemon:
        """データからPokemonインスタンスを作成"""
        pokemon = Pokemon(data["name"])
        pokemon.item = data.get("item", "")
        pokemon.ability = data.get("ability", "")
        pokemon.Ttype = data.get("Ttype", data.get("tera_type", ""))
        pokemon.nature = data.get("nature", "")
        pokemon.moves = data.get("moves", [])

        # 努力値
        effort = data.get("effort", [0, 0, 0, 0, 0, 0])
        if isinstance(effort, list) and len(effort) == 6:
            pokemon.effort = effort

        # ステータス再計算
        pokemon.update_status()

        return pokemon

    @staticmethod
    def _init_rebel_ai(session: BattleSession) -> None:
        """ReBeL AIを初期化"""
        if session.ai_mode != "rebel" or not session.checkpoint_path:
            return

        try:
            from src.battle_ui.services.rebel_ai_service import load_rebel_ai

            session.rebel_ai = load_rebel_ai(session.checkpoint_path)
            logger.info(f"ReBeL AI initialized from {session.checkpoint_path}")
        except Exception as e:
            logger.error(f"Failed to initialize ReBeL AI: {e}")
            session.rebel_ai = None

    @staticmethod
    def select_pokemon(session: BattleSession, indices: List[int]) -> None:
        """選出を確定してバトルを開始"""
        if len(indices) != 3:
            raise ValueError("Must select exactly 3 pokemon")

        if any(i < 0 or i >= 6 for i in indices):
            raise ValueError("Invalid pokemon index")

        session.selected_indices = indices

        # ReBeL AIの初期化
        BattleService._init_rebel_ai(session)

        # AIの選出
        if session.rebel_ai is not None:
            # Selection BERTを使用
            my_team_names = [p["name"] for p in session.player_team_data]
            opp_team_names = [p["name"] for p in session.opponent_team_data]
            ai_indices = session.rebel_ai.get_selection(
                my_team_names=opp_team_names,  # AIから見た自分のチーム
                opponent_team_names=my_team_names,  # AIから見た相手（プレイヤー）
                deterministic=True,
            )
            logger.info(f"ReBeL AI selected: {ai_indices}")
        else:
            # ランダム選出
            ai_indices = random.sample(range(6), 3)
        session.ai_selected_indices = ai_indices

        # Battleオブジェクトを初期化
        battle = session.battle
        battle.reset_game()

        # UI待機モードを有効化（交代コマンドがなければ中断）
        battle.interactive_mode = True

        # プレイヤーのポケモンを設定
        player_pokemon = []
        for idx in indices:
            pokemon_data = session.player_team_data[idx]
            pokemon = BattleService.create_pokemon_from_data(pokemon_data)
            player_pokemon.append(pokemon)
        battle.selected[0] = player_pokemon

        # AIのポケモンを設定
        ai_pokemon = []
        for idx in ai_indices:
            pokemon_data = session.opponent_team_data[idx]
            pokemon = BattleService.create_pokemon_from_data(pokemon_data)
            ai_pokemon.append(pokemon)
        battle.selected[1] = ai_pokemon

        # ReBeL AIの信念状態を初期化
        if session.rebel_ai is not None:
            session.rebel_ai.reset()
            player_names = [p.name for p in player_pokemon]
            session.rebel_ai.init_belief_state(1, player_names)

        # バトル開始（0ターン目：先頭のポケモンを場に出す）
        battle.proceed(commands=[None, None])

        # AIに場に出たポケモンの情報を伝える（ふうせん等）
        if session.rebel_ai is not None:
            session.rebel_ai.observe_battle_state(battle, ai_player=1)

        session.phase = "battle"

    @staticmethod
    def _get_ai_command(session: BattleSession, phase: str) -> Optional[int]:
        """AIのコマンドを取得"""
        battle = session.battle

        # ReBeL AIが利用可能な場合
        if session.rebel_ai is not None:
            try:
                if phase == "change":
                    return session.rebel_ai.get_change_command(battle, player=1)
                else:
                    return session.rebel_ai.get_battle_command(battle, player=1)
            except Exception as e:
                logger.error(f"ReBeL AI command failed: {e}")
                # フォールバック: ランダム

        # ランダム選択
        ai_commands = battle.available_commands(1, phase=phase)
        if ai_commands and ai_commands != [Battle.NO_COMMAND]:
            return random.choice(ai_commands)
        return Battle.SKIP if phase == "battle" else None

    @staticmethod
    def perform_action(
        session: BattleSession,
        action_type: str,
        action_index: int,
        with_tera: bool = False,
    ) -> None:
        """行動を実行"""
        battle = session.battle

        # プレイヤーのコマンドを決定
        if action_type == "move":
            player_command = action_index
            if with_tera:
                player_command += 10  # テラスタル込み
        elif action_type == "switch":
            player_command = 20 + action_index
        else:
            raise ValueError(f"Unknown action type: {action_type}")

        # changeフェーズの場合（プレイヤーが交代先を選んだ）
        if session.phase == "change":
            # プレイヤーの交代コマンドを予約
            battle.reserved_change_commands[0].append(player_command)

            # AIの交代コマンドも予約（AIも交代が必要な場合）
            if battle.breakpoint[1]:
                ai_command = BattleService._get_ai_command(session, phase="change")
                if ai_command is not None:
                    battle.reserved_change_commands[1].append(ai_command)

            # proceedを再開（reserved_change_commandsから取り出される）
            battle.proceed()
        else:
            # 通常のバトルフェーズ
            ai_command = BattleService._get_ai_command(session, phase="battle")
            if ai_command is None:
                ai_command = Battle.SKIP

            cmds = [player_command, ai_command]
            battle.proceed(commands=cmds)  # type: ignore

            # ターン処理後にAI側の交代が必要になった場合は予約しておく
            # (interactive_modeなのでプレイヤーの交代は中断される)
            if battle.breakpoint[1] and not battle.breakpoint[0]:
                ai_change_cmd = BattleService._get_ai_command(session, phase="change")
                if ai_change_cmd is not None:
                    battle.reserved_change_commands[1].append(ai_change_cmd)
                    battle.proceed()  # AI交代を実行

        # AIにターン後の状態を観測させる（持ち物・特性・テラスタル等）
        if session.rebel_ai is not None:
            session.rebel_ai.observe_battle_state(battle, ai_player=1)

        # 勝敗チェック
        winner = battle.winner()
        if winner is not None:
            session.phase = "finished"
            session.winner = winner
        # AIが必敗状態なら降参
        elif session.rebel_ai is not None and session.rebel_ai.should_surrender(battle, player=1):
            logger.info("ReBeL AI detected hopeless situation, surrendering")
            session.phase = "finished"
            session.winner = 0  # プレイヤー勝利
            session.ai_surrendered = True
        # プレイヤーの交代が必要な場合（breakpoint[0]が空でない文字列）
        elif battle.breakpoint[0]:
            session.phase = "change"
        else:
            session.phase = "battle"

    @staticmethod
    def surrender(session: BattleSession) -> None:
        """降参"""
        session.phase = "finished"
        session.winner = 1  # AI勝利

    @staticmethod
    def get_battle_state(
        session: BattleSession, include_ai_analysis: bool = False
    ) -> Dict[str, Any]:
        """バトル状態をテンプレート用のdictに変換"""
        battle = session.battle

        state = {
            "session_id": session.id,
            "phase": session.phase,
            "turn": battle.turn,
            "winner": session.winner,
            "player_team_data": session.player_team_data,
            "opponent_team_data": session.opponent_team_data,
            "selected_indices": session.selected_indices,
            "ai_mode": session.ai_mode,
            "ai_analysis_always_on": session.ai_analysis_always_on,
            "ai_surrendered": session.ai_surrendered,
        }

        # 選出フェーズ
        if session.phase == "selection":
            state["player_team"] = [
                BattleService._pokemon_data_to_view(p, i)
                for i, p in enumerate(session.player_team_data)
            ]
            state["opponent_preview"] = [
                {"name": p["name"], "index": i}
                for i, p in enumerate(session.opponent_team_data)
            ]
            return state

        # バトルフェーズ以降
        if battle.pokemon[0] is not None:
            state["player_active"] = BattleService._pokemon_to_view(
                battle.pokemon[0], is_player=True
            )

        if battle.pokemon[1] is not None:
            state["opponent_active"] = BattleService._pokemon_to_view(
                battle.pokemon[1], is_player=False
            )

        # ベンチポケモン（アクティブなポケモンは除外）
        state["player_bench"] = []
        active_player = battle.pokemon[0]
        for i, p in enumerate(battle.selected[0]):
            # アクティブポケモンと同一でない、かつHPが残っているポケモンのみ
            if p is not None and p is not active_player and p.hp > 0:
                state["player_bench"].append(
                    BattleService._pokemon_to_view(p, is_player=True, bench_index=i)
                )

        state["opponent_bench"] = []
        active_opponent = battle.pokemon[1]
        for i, p in enumerate(battle.selected[1]):
            # アクティブポケモンと同一でない、かつHPが残っているポケモンのみ
            if p is not None and p is not active_opponent and p.hp > 0:
                state["opponent_bench"].append(
                    BattleService._pokemon_to_view(p, is_player=False, bench_index=i)
                )

        # 利用可能なアクション
        if session.phase == "battle":
            state["available_actions"] = BattleService._get_available_actions(
                battle, player=0, phase="battle"
            )
        elif session.phase == "change":
            state["available_actions"] = BattleService._get_available_actions(
                battle, player=0, phase="change"
            )
        else:
            state["available_actions"] = []

        # フィールド状態
        state["field"] = BattleService._get_field_state(battle)

        # ログ
        state["log"] = BattleService._get_battle_log(battle)

        # AI分析データ（ReBeL AIの場合のみ）
        if include_ai_analysis and session.rebel_ai is not None:
            state["ai_analysis"] = session.rebel_ai.get_analysis(battle, player=1)
        else:
            state["ai_analysis"] = None

        return state

    @staticmethod
    def _pokemon_data_to_view(data: Dict[str, Any], index: int) -> Dict[str, Any]:
        """パーティデータをビュー用に変換"""
        return {
            "index": index,
            "name": data["name"],
            "item": data.get("item", ""),
            "ability": data.get("ability", ""),
            "tera_type": data.get("Ttype", data.get("tera_type", "")),
            "moves": data.get("moves", []),
            "nature": data.get("nature", ""),
        }

    @staticmethod
    def _pokemon_to_view(
        pokemon: Pokemon,
        is_player: bool,
        bench_index: Optional[int] = None,
    ) -> Dict[str, Any]:
        """PokemonインスタンスをビューMに変換"""
        max_hp = pokemon.status[0]  # status[0] = HP
        current_hp = pokemon.hp
        hp_ratio = current_hp / max_hp if max_hp > 0 else 0

        view = {
            "name": pokemon.name,
            "hp_ratio": hp_ratio,
            "hp_percent": int(hp_ratio * 100),
            "current_hp": current_hp,
            "max_hp": max_hp,
            "types": pokemon.types,
            "status": pokemon.ailment if pokemon.ailment else None,
            "is_terastallized": pokemon.terastal,
            "tera_type": pokemon.Ttype,
            "bench_index": bench_index,
        }

        if is_player:
            view["item"] = pokemon.item
            view["ability"] = pokemon.ability
            view["moves"] = pokemon.moves
            # rank は [H, A, B, C, D, S, 命中, 回避] のリスト
            view["stat_changes"] = {
                "attack": pokemon.rank[1],
                "defense": pokemon.rank[2],
                "sp_attack": pokemon.rank[3],
                "sp_defense": pokemon.rank[4],
                "speed": pokemon.rank[5],
            }
        else:
            # 相手は観測済み情報のみ（簡略化：常に表示）
            view["item"] = pokemon.item
            view["ability"] = pokemon.ability

        return view

    @staticmethod
    def _get_available_actions(
        battle: Battle, player: int, phase: str
    ) -> List[Dict[str, Any]]:
        """利用可能なアクションを取得"""
        commands = battle.available_commands(player, phase=phase)
        actions = []

        pokemon = battle.pokemon[player]
        if pokemon is None:
            return actions

        for cmd in commands:
            if cmd == Battle.NO_COMMAND:
                continue
            elif cmd == Battle.STRUGGLE:
                actions.append({
                    "type": "move",
                    "index": 30,
                    "name": "わるあがき",
                    "disabled": False,
                })
            elif 0 <= cmd <= 3:
                # 通常技
                move_name = pokemon.moves[cmd] if cmd < len(pokemon.moves) else "???"
                # テラスタル可能かどうか: チーム全体でまだテラスタルしていない
                can_tera = battle.can_terastal(player)
                actions.append({
                    "type": "move",
                    "index": cmd,
                    "name": move_name,
                    "disabled": False,
                    "can_tera": can_tera,
                })
            elif 10 <= cmd <= 13:
                # テラスタル技（既に通常技として追加済みなのでスキップ）
                pass
            elif 20 <= cmd <= 25:
                # 交代
                bench_idx = cmd - 20
                if bench_idx < len(battle.selected[player]):
                    bench_pokemon = battle.selected[player][bench_idx]
                    if bench_pokemon and bench_pokemon.hp > 0:
                        max_hp = bench_pokemon.status[0]
                        actions.append({
                            "type": "switch",
                            "index": bench_idx,
                            "name": bench_pokemon.name,
                            "hp_ratio": bench_pokemon.hp / max_hp if max_hp > 0 else 0,
                            "disabled": False,
                        })

        return actions

    @staticmethod
    def _get_field_state(battle: Battle) -> Dict[str, Any]:
        """フィールド状態を取得"""
        condition = battle.condition

        # 天気
        weather = None
        if condition["sunny"]:
            weather = "はれ"
        elif condition["rainy"]:
            weather = "あめ"
        elif condition["snow"]:
            weather = "ゆき"
        elif condition["sandstorm"]:
            weather = "すなあらし"

        # フィールド
        terrain = None
        if condition["elecfield"]:
            terrain = "エレキフィールド"
        elif condition["glassfield"]:
            terrain = "グラスフィールド"
        elif condition["psycofield"]:
            terrain = "サイコフィールド"
        elif condition["mistfield"]:
            terrain = "ミストフィールド"

        return {
            "weather": weather,
            "terrain": terrain,
            "trick_room": condition["trickroom"] > 0,
            "gravity": condition["gravity"] > 0,
            "player_side": {
                "stealth_rock": condition["stealthrock"][0] > 0,
                "spikes": condition["makibishi"][0],
                "toxic_spikes": condition["dokubishi"][0],
                "sticky_web": condition["nebanet"][0] > 0,
                "reflect": condition["reflector"][0],
                "light_screen": condition["lightwall"][0],
                "tailwind": condition["oikaze"][0],
            },
            "opponent_side": {
                "stealth_rock": condition["stealthrock"][1] > 0,
                "spikes": condition["makibishi"][1],
                "toxic_spikes": condition["dokubishi"][1],
                "sticky_web": condition["nebanet"][1] > 0,
                "reflect": condition["reflector"][1],
                "light_screen": condition["lightwall"][1],
                "tailwind": condition["oikaze"][1],
            },
        }

    @staticmethod
    def _get_battle_log(battle: Battle) -> List[Dict[str, Any]]:
        """バトルログを取得"""
        logs = []

        # battle.log は [player0のログ, player1のログ] の形式
        # ターンごとにまとめる
        if hasattr(battle, "log") and battle.log:
            for player in range(2):
                if battle.log[player]:
                    for msg in battle.log[player]:
                        logs.append({
                            "player": player,
                            "message": msg,
                        })

        return logs
