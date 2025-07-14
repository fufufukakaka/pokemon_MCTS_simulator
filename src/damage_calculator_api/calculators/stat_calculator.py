"""
能力値・補正計算専用モジュール

既存のbattle.py の attack_correction(), defence_correction(), power_correction() 等を独立化
"""

import logging
from typing import Dict, Optional

from src.damage_calculator_api.models.pokemon_models import (
    BattleConditions,
    MoveInput,
    PokemonState,
    StatusAilment,
    TerrainCondition,
    WeatherCondition,
)
from src.damage_calculator_api.utils.data_loader import get_data_loader

logger = logging.getLogger(__name__)


class StatCalculator:
    """
    能力値と各種補正を計算するクラス

    既存のBattle.attack_correction(), Battle.defence_correction(),
    Battle.power_correction(), Battle.damage_correction() を参考に実装
    """

    def __init__(self):
        self.data_loader = get_data_loader()

    def calculate_attack_stat(
        self,
        attacker: PokemonState,
        move_data,
        conditions: BattleConditions,
        opponent: PokemonState = None,
    ) -> int:
        """
        攻撃実数値を計算（特性・道具・状況補正込み）

        既存のBattle.attack_correction() を参考に実装
        """
        # 基本攻撃実数値を取得
        if move_data.is_physical:
            base_stat = attacker.stats.get("attack", 105)
            rank_boost = attacker.stat_boosts.get("attack", 0)
        else:
            base_stat = attacker.stats.get("sp_attack", 105)
            rank_boost = attacker.stat_boosts.get("sp_attack", 0)

        # ランク補正を適用
        rank_multiplier = self._get_rank_multiplier(rank_boost)
        stat_with_rank = int(base_stat * rank_multiplier)

        # 特性による補正
        ability_multiplier = self._get_attack_ability_multiplier(
            attacker, move_data, conditions
        )

        # 道具による補正
        item_multiplier = self._get_attack_item_multiplier(
            attacker, move_data, conditions
        )

        # 状態による補正
        status_multiplier = self._get_attack_status_multiplier(attacker, move_data)

        # その他の補正
        other_multiplier = self._get_attack_other_multiplier(
            attacker, move_data, conditions
        )

        # 相手の災いの特性による補正
        disaster_multiplier = self._get_attack_disaster_multiplier(
            attacker, move_data, opponent
        )

        # 最終攻撃実数値
        final_stat = int(
            stat_with_rank
            * ability_multiplier
            * item_multiplier
            * status_multiplier
            * other_multiplier
            * disaster_multiplier
        )

        return max(1, final_stat)  # 最低1

    def calculate_defense_stat(
        self,
        defender: PokemonState,
        move_data,
        conditions: BattleConditions,
        opponent: PokemonState = None,
    ) -> int:
        """
        防御実数値を計算（特性・道具・状況補正込み）

        既存のBattle.defence_correction() を参考に実装
        """
        # 基本防御実数値を取得
        if move_data.is_physical:
            base_stat = defender.stats.get("defense", 105)
            rank_boost = defender.stat_boosts.get("defense", 0)
        else:
            base_stat = defender.stats.get("sp_defense", 105)
            rank_boost = defender.stat_boosts.get("sp_defense", 0)

        # ランク補正を適用
        rank_multiplier = self._get_rank_multiplier(rank_boost)
        stat_with_rank = int(base_stat * rank_multiplier)

        # 特性による補正
        ability_multiplier = self._get_defense_ability_multiplier(
            defender, move_data, conditions
        )

        # 道具による補正
        item_multiplier = self._get_defense_item_multiplier(
            defender, move_data, conditions
        )

        # 壁による補正
        wall_multiplier = self._get_wall_multiplier(move_data, conditions)

        # その他の補正
        other_multiplier = self._get_defense_other_multiplier(
            defender, move_data, conditions
        )

        # 相手の災いの特性による補正
        disaster_multiplier = self._get_defense_disaster_multiplier(
            defender, move_data, opponent
        )

        # 最終防御実数値
        final_stat = int(
            stat_with_rank
            * ability_multiplier
            * item_multiplier
            * wall_multiplier
            * other_multiplier
            * disaster_multiplier
        )

        return max(1, final_stat)  # 最低1

    def calculate_move_power(
        self,
        attacker: PokemonState,
        defender: PokemonState,
        move: MoveInput,
        move_data,
        conditions: BattleConditions,
    ) -> int:
        """
        技威力を計算（特性・道具・状況補正込み）

        既存のBattle.power_correction() を参考に実装
        """
        base_power = move_data.power
        if base_power <= 0:
            return 0

        # 基本威力修正
        power_modifier = move.power_modifier

        # 天気による補正
        weather_modifier = self._get_weather_power_modifier(move_data, conditions)

        # テラインによる補正
        terrain_modifier = self._get_terrain_power_modifier(move_data, conditions)

        # 特性による補正
        ability_modifier = self._get_power_ability_modifier(
            attacker, defender, move, move_data, conditions
        )

        # 道具による補正
        item_modifier = self._get_power_item_modifier(attacker, move_data)

        # 状況による補正
        situation_modifier = self._get_power_situation_modifier(
            attacker, defender, move, move_data, conditions
        )

        # 最終威力
        final_power = int(
            base_power
            * power_modifier
            * weather_modifier
            * terrain_modifier
            * ability_modifier
            * item_modifier
            * situation_modifier
        )

        return max(1, final_power)  # 最低1

    def calculate_final_damage_modifier(
        self,
        attacker: PokemonState,
        defender: PokemonState,
        move: MoveInput,
        move_data,
        conditions: BattleConditions,
    ) -> float:
        """
        最終ダメージ補正を計算

        既存のBattle.damage_correction() を参考に実装
        """
        modifier = 1.0

        # 特性による最終ダメージ補正
        modifier *= self._get_final_damage_ability_modifier(
            attacker, defender, move, move_data, conditions
        )

        # 道具による最終ダメージ補正
        modifier *= self._get_final_damage_item_modifier(
            attacker, defender, move, move_data
        )

        # 技の特殊効果による補正
        modifier *= self._get_move_special_modifier(move, move_data, conditions)

        return modifier

    def _get_rank_multiplier(self, rank: int) -> float:
        """ランク補正の倍率を取得"""
        rank = max(-6, min(6, rank))  # -6〜+6に制限

        if rank >= 0:
            return (2 + rank) / 2
        else:
            return 2 / (2 - rank)

    def _get_attack_ability_multiplier(
        self, attacker: PokemonState, move_data, conditions: BattleConditions
    ) -> float:
        """攻撃特性による補正"""
        ability = attacker.ability
        multiplier = 1.0

        # ちからもち（物理攻撃2倍）
        if ability == "ちからもち" and move_data.is_physical:
            multiplier *= 2.0

        # ヨガパワー（物理攻撃2倍）
        if ability == "ヨガパワー" and move_data.is_physical:
            multiplier *= 2.0

        # こんじょう（状態異常時物理攻撃1.5倍）
        if (
            ability == "こんじょう"
            and move_data.is_physical
            and attacker.status_ailment != StatusAilment.NONE
        ):
            multiplier *= 1.5

        # サンパワー（晴れ時特殊攻撃1.5倍）
        if (
            ability == "サンパワー"
            and move_data.is_special
            and conditions.weather == WeatherCondition.SUN
        ):
            multiplier *= 1.5

        # ソーラーパワー（晴れ時特殊攻撃1.5倍）
        if (
            ability == "ソーラーパワー"
            and move_data.is_special
            and conditions.weather == WeatherCondition.SUN
        ):
            multiplier *= 1.5

        # はりきり（物理攻撃1.5倍）
        if ability == "はりきり" and move_data.is_physical:
            multiplier *= 1.5

        # スロースタート（5ターン間攻撃・素早さ半減）
        # TODO: ターン管理が必要

        # もうか（HP1/3以下でほのお技威力1.5倍）
        if (
            ability == "もうか"
            and move_data.move_type == "ほのお"
            and attacker.hp_ratio <= 1 / 3
        ):
            multiplier *= 1.5

        # しんりょく（HP1/3以下でくさ技威力1.5倍）
        if (
            ability == "しんりょく"
            and move_data.move_type == "くさ"
            and attacker.hp_ratio <= 1 / 3
        ):
            multiplier *= 1.5

        # げきりゅう（HP1/3以下でみず技威力1.5倍）
        if (
            ability == "げきりゅう"
            and move_data.move_type == "みず"
            and attacker.hp_ratio <= 1 / 3
        ):
            multiplier *= 1.5

        # むしのしらせ（HP1/3以下でむし技威力1.5倍）
        if (
            ability == "むしのしらせ"
            and move_data.move_type == "むし"
            and attacker.hp_ratio <= 1 / 3
        ):
            multiplier *= 1.5

        # ひひいろのこどう（晴れ時攻撃1.33倍）
        if ability == "ひひいろのこどう" and conditions.weather == WeatherCondition.SUN:
            multiplier *= 5461 / 4096  # ≈ 1.33x

        # ハドロンエンジン（エレキフィールド時特攻1.33倍）
        if (
            ability == "ハドロンエンジン"
            and not move_data.is_physical
            and conditions.terrain == TerrainCondition.ELECTRIC
        ):
            multiplier *= 5461 / 4096  # ≈ 1.33x

        # クォークチャージ（エレキフィールド時最も高い能力値1.3倍）
        if (
            ability == "クォークチャージ"
            and conditions.terrain == TerrainCondition.ELECTRIC
            and attacker.paradox_boost_stat
        ):
            print(
                f"クォークチャージ条件チェック: terrain={conditions.terrain}, paradox_stat={attacker.paradox_boost_stat}, is_physical={move_data.is_physical}"
            )
            # 指定された能力値が攻撃系の場合のみ適用
            if move_data.is_physical and attacker.paradox_boost_stat == "attack":
                print(
                    f"クォークチャージ攻撃補正適用: {multiplier} -> {multiplier * 1.3}"
                )
                multiplier *= 1.3
            elif (
                not move_data.is_physical and attacker.paradox_boost_stat == "sp_attack"
            ):
                print(
                    f"クォークチャージ特攻補正適用: {multiplier} -> {multiplier * 1.3}"
                )
                multiplier *= 1.3

        # 古代活性（晴れ時最も高い能力値1.3倍）
        if (
            ability == "こだいかっせい"
            and conditions.weather == WeatherCondition.SUN
            and attacker.paradox_boost_stat
        ):
            # 指定された能力値が攻撃系の場合のみ適用
            if move_data.is_physical and attacker.paradox_boost_stat == "attack":
                multiplier *= 1.3
            elif (
                not move_data.is_physical and attacker.paradox_boost_stat == "sp_attack"
            ):
                multiplier *= 1.3

        # すいほう（みず技威力2倍）
        if ability == "すいほう" and move_data.move_type == "みず":
            multiplier *= 2.0

        # ごりむちゅう（物理攻撃1.5倍）
        if ability == "ごりむちゅう" and move_data.is_physical:
            multiplier *= 1.5

        # フェアリースキン（ノーマル技がフェアリータイプになり威力1.2倍）
        if ability == "フェアリースキン" and move_data.move_type == "ノーマル":
            # Note: タイプ変更は別途TypeCalculatorで処理
            multiplier *= 1.2

        # スカイスキン（ノーマル技がひこうタイプになり威力1.2倍）
        if ability == "スカイスキン" and move_data.move_type == "ノーマル":
            multiplier *= 1.2

        # エレキスキン（ノーマル技がでんきタイプになり威力1.2倍）
        if ability == "エレキスキン" and move_data.move_type == "ノーマル":
            multiplier *= 1.2

        # フリーズスキン（ノーマル技がこおりタイプになり威力1.2倍）
        if ability == "フリーズスキン" and move_data.move_type == "ノーマル":
            multiplier *= 1.2

        return multiplier

    def _get_defense_ability_multiplier(
        self, defender: PokemonState, move_data, conditions: BattleConditions
    ) -> float:
        """防御特性による補正"""
        ability = defender.ability
        multiplier = 1.0
        # ファーコート（物理防御2倍）
        if ability == "ファーコート" and move_data.is_physical:
            multiplier *= 2.0

        # マルチスケイル（HP満タン時ダメージ半減）
        if ability == "マルチスケイル" and defender.hp_ratio >= 1.0:
            multiplier *= 2.0  # 防御側なので防御実数値を2倍

        # シャドーシールド（HP満タン時ダメージ半減）
        if ability == "シャドーシールド" and defender.hp_ratio >= 1.0:
            multiplier *= 2.0

        # ふしぎなうろこ（状態異常時防御1.5倍）
        if (
            ability == "ふしぎなうろこ"
            and move_data.is_physical
            and defender.status_ailment != StatusAilment.NONE
        ):
            multiplier *= 1.5

        # あついしぼう、たいねつ等のタイプ技軽減は TypeCalculator で処理済み

        # クォークチャージ（エレキフィールド時最も高い能力値1.3倍）
        if (
            ability == "クォークチャージ"
            and conditions.terrain == TerrainCondition.ELECTRIC
            and defender.paradox_boost_stat
        ):
            # 指定された能力値が防御系の場合のみ適用
            if move_data.is_physical and defender.paradox_boost_stat == "defense":
                multiplier *= 1.3
            elif (
                not move_data.is_physical
                and defender.paradox_boost_stat == "sp_defense"
            ):
                multiplier *= 1.3

        # 古代活性（晴れ時最も高い能力値1.3倍）
        if (
            ability == "こだいかっせい"
            and conditions.weather == WeatherCondition.SUN
            and defender.paradox_boost_stat
        ):
            # 指定された能力値が防御系の場合のみ適用
            if move_data.is_physical and defender.paradox_boost_stat == "defense":
                multiplier *= 1.3
            elif (
                not move_data.is_physical
                and defender.paradox_boost_stat == "sp_defense"
            ):
                multiplier *= 1.3

        # きせき（進化前ポケモンの防御・特防1.5倍）
        # TODO: 進化前判定が必要

        return multiplier

    def _get_attack_item_multiplier(
        self, attacker: PokemonState, move_data, conditions: BattleConditions
    ) -> float:
        """攻撃道具による補正"""
        item = attacker.item
        if not item:
            return 1.0

        multiplier = 1.0

        # こだわりハチマキ（物理攻撃1.5倍）
        if item == "こだわりハチマキ" and move_data.is_physical:
            multiplier *= 1.5

        # こだわりメガネ（特殊攻撃1.5倍）
        if item == "こだわりメガネ" and move_data.is_special:
            multiplier *= 1.5

        # タイプ強化アイテム
        item_data = self.data_loader.get_item_data(item)
        if item_data and item_data.boost_type == move_data.move_type:
            multiplier *= 1.2

        return multiplier

    def _get_defense_item_multiplier(
        self, defender: PokemonState, move_data, conditions: BattleConditions
    ) -> float:
        """防御道具による補正"""
        item = defender.item
        if not item:
            return 1.0

        multiplier = 1.0

        # しんかのきせき（進化前ポケモンの防御・特防1.5倍）
        if item == "しんかのきせき":
            # TODO: 進化前判定が必要
            multiplier *= 1.5

        # とつげきチョッキ（特防1.5倍、変化技使用不可）
        if item == "とつげきチョッキ" and move_data.is_special:
            multiplier *= 1.5

        # メタルパウダー（メタモン専用、防御2倍）
        if (
            item == "メタルパウダー"
            and defender.species == "メタモン"
            and move_data.is_physical
        ):
            multiplier *= 2.0

        return multiplier

    def _get_attack_status_multiplier(self, attacker: PokemonState, move_data) -> float:
        """状態異常による攻撃補正"""
        # やけど状態での物理攻撃半減（特性「こんじょう」等は除く）
        if (
            attacker.status_ailment == StatusAilment.BURN
            and move_data.is_physical
            and attacker.ability not in ["こんじょう", "からかい"]
        ):
            return 0.5

        return 1.0

    def _get_wall_multiplier(self, move_data, conditions: BattleConditions) -> float:
        """壁による防御補正"""
        # リフレクター（物理技半減）
        if conditions.reflect and move_data.is_physical:
            return 2.0  # 防御実数値を2倍

        # ひかりのかべ（特殊技半減）
        if conditions.light_screen and move_data.is_special:
            return 2.0

        # オーロラベール（物理・特殊両方半減）
        if conditions.aurora_veil:
            return 2.0

        return 1.0

    def _get_weather_power_modifier(
        self, move_data, conditions: BattleConditions
    ) -> float:
        """天気による威力補正"""
        multiplier = 1.0

        if conditions.weather == WeatherCondition.SUN:
            if move_data.move_type == "ほのお":
                multiplier *= 1.5
            elif move_data.move_type == "みず":
                multiplier *= 0.5
        elif conditions.weather == WeatherCondition.RAIN:
            if move_data.move_type == "みず":
                multiplier *= 1.5
            elif move_data.move_type == "ほのお":
                multiplier *= 0.5
        elif conditions.weather == WeatherCondition.SANDSTORM:
            if move_data.move_type == "いわ":
                # すなあらしで特防1.5倍（いわタイプ）は別処理
                pass

        return multiplier

    def _get_terrain_power_modifier(
        self, move_data, conditions: BattleConditions
    ) -> float:
        """テラインによる威力補正"""
        multiplier = 1.0

        if (
            conditions.terrain == TerrainCondition.ELECTRIC
            and move_data.move_type == "でんき"
        ):
            multiplier *= 1.3
        elif (
            conditions.terrain == TerrainCondition.GRASSY
            and move_data.move_type == "くさ"
        ):
            multiplier *= 1.3
        elif (
            conditions.terrain == TerrainCondition.PSYCHIC
            and move_data.move_type == "エスパー"
        ):
            multiplier *= 1.3
        elif (
            conditions.terrain == TerrainCondition.MISTY
            and move_data.move_type == "フェアリー"
        ):
            multiplier *= 1.3

        return multiplier

    def _get_power_ability_modifier(
        self,
        attacker: PokemonState,
        defender: PokemonState,
        move: MoveInput,
        move_data,
        conditions: BattleConditions,
    ) -> float:
        """特性による威力補正"""
        ability = attacker.ability
        multiplier = 1.0

        # てきおうりょく（タイプ一致技の威力2倍→1.5倍からさらに1.33倍）
        if ability == "てきおうりょく":
            attacker_species = self.data_loader.get_pokemon_data(attacker.species)
            if attacker_species and move_data.move_type in attacker_species.types:
                multiplier *= 4 / 3  # 1.5 → 2.0にするため4/3倍

        # アナライズ（後攻時威力1.3倍）
        # TODO: 行動順判定が必要

        # テクニシャン（威力60以下の技1.5倍）
        if ability == "テクニシャン" and move_data.power <= 60:
            multiplier *= 1.5

        # すなのちから（すなあらし時じめん・いわ・はがね技1.3倍）
        if (
            ability == "すなのちから"
            and conditions.weather == WeatherCondition.SANDSTORM
            and move_data.move_type in ["じめん", "いわ", "はがね"]
        ):
            multiplier *= 1.3

        # いわはこび（いわ技威力1.5倍）
        if ability == "いわはこび" and move_data.move_type == "いわ":
            multiplier *= 1.5

        # トランジスタ（でんき技威力1.3倍）
        if ability == "トランジスタ" and move_data.move_type == "でんき":
            multiplier *= 1.3

        # りゅうのあぎと（ドラゴン技威力1.5倍）
        if ability == "りゅうのあぎと" and move_data.move_type == "ドラゴン":
            multiplier *= 1.5

        # はがねつかい（はがね技威力1.5倍）
        if ability == "はがねつかい" and move_data.move_type == "はがね":
            multiplier *= 1.5

        # パンクロック（音技威力1.3倍）
        if ability == "パンクロック":
            # TODO: 音技判定が必要（技名での判定）
            sound_moves = [
                "ハイパーボイス",
                "ばくおんぱ",
                "りんしょう",
                "いびき",
                "エコーボイス",
            ]
            if move_data.name in sound_moves:
                multiplier *= 1.3

        # どくぼうそう（毒状態時物理技威力1.5倍）
        if (
            ability == "どくぼうそう"
            and move_data.is_physical
            and attacker.status_ailment == StatusAilment.POISON
        ):
            multiplier *= 1.5

        # ねつぼうそう（やけど状態時特殊技威力1.5倍）
        if (
            ability == "ねつぼうそう"
            and move_data.is_special
            and attacker.status_ailment == StatusAilment.BURN
        ):
            multiplier *= 1.5

        return multiplier

    def _get_power_item_modifier(self, attacker: PokemonState, move_data) -> float:
        """道具による威力補正"""
        item = attacker.item
        if not item:
            return 1.0

        multiplier = 1.0

        # いのちのたま（全技威力1.3倍、反動あり）
        if item == "いのちのたま":
            multiplier *= 1.3

        # たつじんのおび（効果抜群技威力1.2倍）
        # TODO: タイプ相性情報が必要

        # ノーマルジュエル等（該当タイプの技威力1.3倍、一度のみ）
        # TODO: 消費アイテム管理が必要

        return multiplier

    def _get_power_situation_modifier(
        self,
        attacker: PokemonState,
        defender: PokemonState,
        move: MoveInput,
        move_data,
        conditions: BattleConditions,
    ) -> float:
        """状況による威力補正"""
        multiplier = 1.0

        # ウェザーボール（天気により威力・タイプ変化）
        if (
            move.name == "ウェザーボール"
            and conditions.weather != WeatherCondition.NONE
        ):
            multiplier *= 2.0

        # ジャイロボール（相手より遅いほど威力アップ）
        if move.name == "ジャイロボール":
            # TODO: 素早さ比較が必要
            pass

        # エレキボール（相手より速いほど威力アップ）
        if move.name == "エレキボール":
            # TODO: 素早さ比較が必要
            pass

        # アクセルブレイク/イナズマドライブ（効果バツグンの時威力1.3倍）
        if move.name == "アクセルブレイク" or move.name == "イナズマドライブ":
            # タイプ相性を確認（効果バツグンかどうか）
            type_effectiveness = self._check_type_effectiveness_for_move(
                attacker, defender, move_data
            )
            if type_effectiveness > 1.0:  # 効果バツグンの場合
                multiplier *= 5461 / 4096  # ≈ 1.33x

        return multiplier

    def _check_type_effectiveness_for_move(
        self, attacker: PokemonState, defender: PokemonState, move_data
    ) -> float:
        """技のタイプ相性を簡易チェック"""
        # TypeCalculatorを使用してタイプ相性を取得
        from src.damage_calculator_api.calculators.type_calculator import TypeCalculator

        type_calc = TypeCalculator()

        # 簡易的なダミーのMoveInputとBattleConditionsを作成
        from src.damage_calculator_api.models.pokemon_models import (
            BattleConditions,
            MoveInput,
        )

        dummy_move = MoveInput(name=move_data.name)
        dummy_conditions = BattleConditions()

        return type_calc.calculate_type_effectiveness(
            attacker, defender, dummy_move, move_data, dummy_conditions
        )

    def _get_final_damage_ability_modifier(
        self,
        attacker: PokemonState,
        defender: PokemonState,
        move: MoveInput,
        move_data,
        conditions: BattleConditions,
    ) -> float:
        """特性による最終ダメージ補正"""
        multiplier = 1.0

        # スナイパー（急所時ダメージ1.5倍→2.25倍）
        if attacker.ability == "スナイパー" and move.is_critical:
            multiplier *= 4 / 3

        # いろめがね（効果今ひとつ技を等倍にする）
        if attacker.ability == "いろめがね":
            type_effectiveness = self._check_type_effectiveness_for_move(
                attacker, defender, move_data
            )
            if type_effectiveness < 1.0:  # 効果今ひとつの場合
                # 効果今ひとつを等倍にするため、逆数を掛ける
                multiplier *= 1.0 / type_effectiveness

        return multiplier

    def _get_final_damage_item_modifier(
        self, attacker: PokemonState, defender: PokemonState, move: MoveInput, move_data
    ) -> float:
        """道具による最終ダメージ補正"""
        multiplier = 1.0

        # メトロノーム（連続使用で威力アップ）
        # TODO: 連続使用回数管理が必要

        return multiplier

    def _get_move_special_modifier(
        self, move: MoveInput, move_data, conditions: BattleConditions
    ) -> float:
        """技の特殊効果による補正"""
        multiplier = 1.0

        # 2回攻撃技（ダブルアタック等）
        # TODO: 技の特殊効果管理が必要

        return multiplier

    def _get_attack_other_multiplier(
        self, attacker: PokemonState, move_data, conditions: BattleConditions
    ) -> float:
        """その他の攻撃補正"""
        multiplier = 1.0

        # おいかぜ（味方の素早さ2倍）
        if conditions.tailwind:
            # 攻撃に直接影響なし
            pass

        return multiplier

    def _get_defense_other_multiplier(
        self, defender: PokemonState, move_data, conditions: BattleConditions
    ) -> float:
        """その他の防御補正"""
        multiplier = 1.0

        # すなあらしでいわタイプの特防1.5倍
        if conditions.weather == WeatherCondition.SANDSTORM and move_data.is_special:
            defender_species = self.data_loader.get_pokemon_data(defender.species)
            if defender_species and "いわ" in defender_species.types:
                multiplier *= 1.5

        return multiplier

    def _get_attack_disaster_multiplier(
        self, attacker: PokemonState, move_data, opponent: PokemonState
    ) -> float:
        """相手の災いの特性による攻撃実数値補正"""
        if opponent is None:
            return 1.0

        multiplier = 1.0
        opponent_ability = opponent.ability

        # わざわいのうつわ（相手の特殊攻撃25%減）
        if opponent_ability == "わざわいのうつわ" and not move_data.is_physical:
            multiplier *= 3072 / 4096  # ≈ 0.75x

        # わざわいのおふだ（相手の物理攻撃25%減）
        if opponent_ability == "わざわいのおふだ" and move_data.is_physical:
            multiplier *= 3072 / 4096  # ≈ 0.75x

        return multiplier

    def _get_defense_disaster_multiplier(
        self, defender: PokemonState, move_data, opponent: PokemonState
    ) -> float:
        """相手の災いの特性による防御実数値補正"""
        if opponent is None:
            return 1.0

        multiplier = 1.0
        opponent_ability = opponent.ability

        # わざわいのつるぎ（攻撃側がこの特性を持つ時、防御側の物理防御25%減）
        if opponent_ability == "わざわいのつるぎ" and move_data.is_physical:
            multiplier *= 3072 / 4096  # ≈ 0.75x

        # わざわいのたま（攻撃側がこの特性を持つ時、防御側の特殊防御25%減）
        if opponent_ability == "わざわいのたま" and not move_data.is_physical:
            multiplier *= 3072 / 4096  # ≈ 0.75x

        return multiplier
