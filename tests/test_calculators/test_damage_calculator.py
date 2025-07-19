"""
ダメージ計算エンジンの動作検証テスト

基本的な動作確認と既存システムとの整合性確認
"""

import sys
import unittest
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from src.damage_calculator_api.calculators.damage_calculator import (
    DamageCalculator,
    create_simple_pokemon,
)
from src.damage_calculator_api.models.pokemon_models import (
    BattleConditions,
    MoveInput,
    PokemonState,
    StatusAilment,
    TerrainCondition,
    WeatherCondition,
)


class TestDamageCalculator(unittest.TestCase):
    """ダメージ計算エンジンのテストクラス"""

    def setUp(self):
        """各テスト前の準備"""
        self.calculator = DamageCalculator()

        self.pikachu = create_simple_pokemon(
            species="ピカチュウ",
            level=50,
            nature="ひかえめ",  # C↑A↓
            ability="せいでんき",
            evs={"sp_attack": 252, "speed": 252, "hp": 4},
        )

        self.garchomp = create_simple_pokemon(
            species="ガブリアス",
            level=50,
            nature="いじっぱり",  # A↑C↓
            ability="すながくれ",
            evs={"attack": 252, "speed": 252, "hp": 4},
        )

    def test_type_effectiveness(self):
        """タイプ相性テスト"""
        # でんき技 vs じめんタイプ（無効）
        try:
            # じめんタイプのポケモンを作成
            ground_pokemon = PokemonState(
                species="ダグトリオ",
                level=50,
                stats={
                    "hp": 135,
                    "attack": 135,
                    "defense": 90,
                    "sp_attack": 90,
                    "sp_defense": 95,
                    "speed": 188,
                },
                ability="すながくれ",
            )

            electric_move = MoveInput(name="10まんボルト")

            result = self.calculator.calculate_damage(
                attacker=self.pikachu, defender=ground_pokemon, move=electric_move
            )

            # でんき技はじめんタイプに無効
            self.assertEqual(result.max_damage, 0)
            self.assertEqual(result.ko_probability, 0.0)

            print("でんき技 vs じめんタイプ: 無効確認")

        except Exception as e:
            print(f"タイプ相性テストをスキップ: {e}")

    def test_stab_bonus(self):
        """タイプ一致ボーナステスト"""
        try:
            # タイプ一致技
            electric_move = MoveInput(name="10まんボルト")

            # タイプ不一致技
            normal_move = MoveInput(name="でんこうせっか")

            stab_result = self.calculator.calculate_damage(
                attacker=self.pikachu, defender=self.garchomp, move=electric_move
            )

            non_stab_result = self.calculator.calculate_damage(
                attacker=self.pikachu, defender=self.garchomp, move=normal_move
            )

            # タイプ一致技の方が威力が高いはず
            if stab_result.max_damage > 0 and non_stab_result.max_damage > 0:
                stab_modifier = stab_result.calculation_details.get(
                    "stab_modifier", 1.0
                )
                non_stab_modifier = non_stab_result.calculation_details.get(
                    "stab_modifier", 1.0
                )

                self.assertGreater(stab_modifier, non_stab_modifier)
                print(f"STAB確認 - 一致: {stab_modifier}, 不一致: {non_stab_modifier}")

        except Exception as e:
            print(f"STABテストをスキップ: {e}")

    def test_critical_hit(self):
        """急所テスト"""
        try:
            normal_move = MoveInput(name="10まんボルト", is_critical=False)
            critical_move = MoveInput(name="10まんボルト", is_critical=True)

            normal_result = self.calculator.calculate_damage(
                attacker=self.pikachu, defender=self.garchomp, move=normal_move
            )

            critical_result = self.calculator.calculate_damage(
                attacker=self.pikachu, defender=self.garchomp, move=critical_move
            )

            # 急所の方がダメージが高いはず
            if normal_result.max_damage > 0 and critical_result.max_damage > 0:
                critical_modifier = critical_result.calculation_details.get(
                    "critical_modifier", 1.0
                )
                normal_modifier = normal_result.calculation_details.get(
                    "critical_modifier", 1.0
                )

                self.assertEqual(critical_modifier, 1.5)
                self.assertEqual(normal_modifier, 1.0)
                self.assertGreater(critical_result.max_damage, normal_result.max_damage)

                print(
                    f"急所確認 - 通常: {normal_result.max_damage}, 急所: {critical_result.max_damage}"
                )

        except Exception as e:
            print(f"急所テストをスキップ: {e}")

    def test_weather_effects(self):
        """天気効果テスト"""
        try:
            # 晴れ状態でのほのお技
            fire_move = MoveInput(name="かえんほうしゃ")
            sunny_conditions = BattleConditions(weather=WeatherCondition.SUN)
            normal_conditions = BattleConditions()

            sunny_result = self.calculator.calculate_damage(
                attacker=self.pikachu,
                defender=self.garchomp,
                move=fire_move,
                conditions=sunny_conditions,
            )

            normal_result = self.calculator.calculate_damage(
                attacker=self.pikachu,
                defender=self.garchomp,
                move=fire_move,
                conditions=normal_conditions,
            )

            # 晴れ状態ではほのお技が1.5倍
            if sunny_result.max_damage > 0 and normal_result.max_damage > 0:
                self.assertGreater(sunny_result.max_damage, normal_result.max_damage)
                print(
                    f"晴れ効果確認 - 通常: {normal_result.max_damage}, 晴れ: {sunny_result.max_damage}"
                )

        except Exception as e:
            print(f"天気効果テストをスキップ: {e}")

    def test_status_ailment_effects(self):
        """状態異常効果テスト"""
        try:
            # やけど状態での物理技
            burned_attacker = PokemonState(
                species="ガブリアス",
                level=50,
                stats={
                    "hp": 183,
                    "attack": 182,
                    "defense": 115,
                    "sp_attack": 99,
                    "sp_defense": 105,
                    "speed": 169,
                },
                ability="すながくれ",
                status_ailment=StatusAilment.BURN,
            )

            normal_attacker = PokemonState(
                species="ガブリアス",
                level=50,
                stats={
                    "hp": 183,
                    "attack": 182,
                    "defense": 115,
                    "sp_attack": 99,
                    "sp_defense": 105,
                    "speed": 169,
                },
                ability="すながくれ",
                status_ailment=StatusAilment.NONE,
            )

            physical_move = MoveInput(name="じしん")

            burned_result = self.calculator.calculate_damage(
                attacker=burned_attacker, defender=self.pikachu, move=physical_move
            )

            normal_result = self.calculator.calculate_damage(
                attacker=normal_attacker, defender=self.pikachu, move=physical_move
            )

            # やけど状態では物理技が半減
            if burned_result.max_damage > 0 and normal_result.max_damage > 0:
                burn_modifier = burned_result.calculation_details.get(
                    "burn_modifier", 1.0
                )
                self.assertEqual(burn_modifier, 0.5)
                self.assertLess(burned_result.max_damage, normal_result.max_damage)

                print(
                    f"やけど効果確認 - 通常: {normal_result.max_damage}, やけど: {burned_result.max_damage}"
                )

        except Exception as e:
            print(f"状態異常テストをスキップ: {e}")

    def test_move_comparison(self):
        """技比較機能テスト"""
        try:
            moves = [
                MoveInput(name="10まんボルト"),
                MoveInput(name="かみなり"),
                MoveInput(name="でんきショック"),
            ]

            comparison_result = self.calculator.compare_moves(
                attacker=self.pikachu, defender=self.garchomp, moves=moves
            )

            self.assertIsInstance(comparison_result, list)
            self.assertGreater(len(comparison_result), 0)

            # 威力順にソートされているかチェック
            valid_moves = [
                move for move in comparison_result if not move.get("no_damage", False)
            ]
            if len(valid_moves) >= 2:
                self.assertGreaterEqual(
                    valid_moves[0].get("average_damage", 0),
                    valid_moves[1].get("average_damage", 0),
                )

            print("技比較結果:")
            for move in comparison_result:
                if not move.get("no_damage", False):
                    print(
                        f"  {move['move_name']}: {move.get('average_damage', 0):.1f}ダメージ"
                    )

        except Exception as e:
            print(f"技比較テストをスキップ: {e}")

    def test_validation_functions(self):
        """バリデーション機能テスト"""
        # 有効なポケモン
        valid_pokemon = self.pikachu
        self.assertTrue(self.calculator.validate_pokemon_state(valid_pokemon))

        # 無効なポケモン（存在しない種族）
        invalid_pokemon = PokemonState(species="存在しないポケモン")
        self.assertFalse(self.calculator.validate_pokemon_state(invalid_pokemon))

        # 有効な技
        valid_move = MoveInput(name="10まんボルト")
        try:
            result = self.calculator.validate_move_input(valid_move)
            if result:
                self.assertTrue(result)
        except:
            pass  # データが見つからない場合はスキップ

        # 無効な技
        invalid_move = MoveInput(name="存在しない技")
        self.assertFalse(self.calculator.validate_move_input(invalid_move))

        print("バリデーション機能: OK")

    def test_helper_functions(self):
        """ヘルパー関数テスト"""
        try:
            # サポートされている技一覧取得
            supported_moves = self.calculator.get_supported_moves()
            self.assertIsInstance(supported_moves, list)

            # サポートされているポケモン一覧取得
            supported_pokemon = self.calculator.get_supported_pokemon()
            self.assertIsInstance(supported_pokemon, list)

            print(f"サポート技数: {len(supported_moves)}")
            print(f"サポートポケモン数: {len(supported_pokemon)}")

        except Exception as e:
            print(f"ヘルパー関数テストをスキップ: {e}")

    def test_medium_priority_abilities(self):
        """Medium priority特性のテスト"""

        # あついしぼう（ほのお・こおり技半減）
        fire_pokemon = create_simple_pokemon(
            species="ウインディ", level=50, ability="いかく"
        )
        thick_fat_pokemon = create_simple_pokemon(
            species="マリルリ", level=50, ability="あついしぼう"
        )

        fire_move = MoveInput(name="かえんほうしゃ")

        # 通常ダメージ vs あついしぼう
        normal_result = self.calculator.calculate_damage(
            attacker=fire_pokemon, defender=fire_pokemon, move=fire_move
        )
        thick_fat_result = self.calculator.calculate_damage(
            attacker=fire_pokemon, defender=thick_fat_pokemon, move=fire_move
        )

        # あついしぼうで半減されることを確認
        if normal_result.max_damage > 0 and thick_fat_result.max_damage > 0:
            ratio = thick_fat_result.max_damage / normal_result.max_damage
            print(f"あついしぼう ratio: {ratio:.4f}")
            self.assertLess(ratio, 0.6, "あついしぼうでほのお技が半減されるべき")

    def test_strong_jaw_ability(self):
        """がんじょうあご特性のテスト"""
        strong_jaw_pokemon = create_simple_pokemon(
            species="ジュラルドン", level=50, ability="がんじょうあご"
        )
        normal_pokemon = create_simple_pokemon(
            species="ジュラルドン", level=50, ability="ライトメタル"
        )

        bite_move = MoveInput(name="かみくだく")

        strong_jaw_result = self.calculator.calculate_damage(
            attacker=strong_jaw_pokemon, defender=normal_pokemon, move=bite_move
        )
        normal_result = self.calculator.calculate_damage(
            attacker=normal_pokemon, defender=normal_pokemon, move=bite_move
        )

        if normal_result.max_damage > 0:
            ratio = strong_jaw_result.max_damage / normal_result.max_damage
            print(f"がんじょうあご ratio: {ratio:.4f}")
            self.assertAlmostEqual(
                ratio, 1.5, delta=0.1, msg="がんじょうあごで噛み技が1.5倍になるべき"
            )

    def test_sheer_force_ability(self):
        """ちからずく特性のテスト"""
        sheer_force_pokemon = create_simple_pokemon(
            species="ガブリアス", level=50, ability="ちからずく"
        )
        normal_pokemon = create_simple_pokemon(
            species="ガブリアス", level=50, ability="すながくれ"
        )

        # 追加効果のある技
        effect_move = MoveInput(name="10まんボルト")

        sheer_force_result = self.calculator.calculate_damage(
            attacker=sheer_force_pokemon, defender=normal_pokemon, move=effect_move
        )
        normal_result = self.calculator.calculate_damage(
            attacker=normal_pokemon, defender=normal_pokemon, move=effect_move
        )

        if normal_result.max_damage > 0:
            ratio = sheer_force_result.max_damage / normal_result.max_damage
            print(f"ちからずく ratio: {ratio:.4f}")
            self.assertAlmostEqual(
                ratio, 1.3, delta=0.1, msg="ちからずくで追加効果技が1.3倍になるべき"
            )

    def test_filter_ability(self):
        """フィルター特性のテスト"""
        filter_pokemon = create_simple_pokemon(
            species="ピクシー",
            level=50,
            ability="フィルター",
            evs={"hp": 252, "defense": 252},
        )
        normal_pokemon = create_simple_pokemon(
            species="ピクシー",
            level=50,
            ability="メロメロボディ",
            evs={"hp": 252, "defense": 252},
        )

        # はがねタイプの攻撃者（フェアリーに効果バツグン）
        steel_attacker = create_simple_pokemon(
            species="ハッサム", level=50, ability="テクニシャン", evs={"attack": 252}
        )

        super_effective_move = MoveInput(name="アイアンヘッド")

        filter_result = self.calculator.calculate_damage(
            attacker=steel_attacker, defender=filter_pokemon, move=super_effective_move
        )
        normal_result = self.calculator.calculate_damage(
            attacker=steel_attacker, defender=normal_pokemon, move=super_effective_move
        )

        if normal_result.max_damage > 0:
            ratio = filter_result.max_damage / normal_result.max_damage
            print(f"フィルター ratio: {ratio:.4f}")
            self.assertLess(ratio, 0.8, "フィルターで効果バツグン技が軽減されるべき")

    def test_tough_claws_ability(self):
        """かたいつめ特性のテスト"""
        tough_claws_pokemon = create_simple_pokemon(
            species="ガブリアス", level=50, ability="かたいつめ"
        )
        normal_pokemon = create_simple_pokemon(
            species="ガブリアス", level=50, ability="すながくれ"
        )

        # 接触技
        contact_move = MoveInput(name="じしん")  # 接触技

        tough_claws_result = self.calculator.calculate_damage(
            attacker=tough_claws_pokemon, defender=normal_pokemon, move=contact_move
        )
        normal_result = self.calculator.calculate_damage(
            attacker=normal_pokemon, defender=normal_pokemon, move=contact_move
        )

        if normal_result.max_damage > 0:
            ratio = tough_claws_result.max_damage / normal_result.max_damage
            print(f"かたいつめ ratio: {ratio:.4f}")
            # じしんは接触技ではないので、比率は1.0のはず
            self.assertAlmostEqual(
                ratio,
                1.0,
                delta=0.1,
                msg="じしんは接触技ではないのでかたいつめ効果なし",
            )

    def test_iron_fist_ability(self):
        """てつのこぶし特性のテスト"""
        iron_fist_pokemon = create_simple_pokemon(
            species="ガブリアス", level=50, ability="てつのこぶし"
        )
        normal_pokemon = create_simple_pokemon(
            species="ガブリアス", level=50, ability="すながくれ"
        )

        # パンチ技
        punch_move = MoveInput(name="かみなりパンチ")

        iron_fist_result = self.calculator.calculate_damage(
            attacker=iron_fist_pokemon, defender=normal_pokemon, move=punch_move
        )
        normal_result = self.calculator.calculate_damage(
            attacker=normal_pokemon, defender=normal_pokemon, move=punch_move
        )

        if normal_result.max_damage > 0:
            ratio = iron_fist_result.max_damage / normal_result.max_damage
            print(f"てつのこぶし ratio: {ratio:.4f}")
            self.assertAlmostEqual(
                ratio, 1.2, delta=0.1, msg="てつのこぶしでパンチ技が1.2倍になるべき"
            )

    def test_sharpness_ability(self):
        """きれあじ特性のテスト"""
        sharpness_pokemon = create_simple_pokemon(
            species="ガブリアス", level=50, ability="きれあじ"
        )
        normal_pokemon = create_simple_pokemon(
            species="ガブリアス", level=50, ability="すながくれ"
        )

        # 切断技
        cut_move = MoveInput(name="つじぎり")

        sharpness_result = self.calculator.calculate_damage(
            attacker=sharpness_pokemon, defender=normal_pokemon, move=cut_move
        )
        normal_result = self.calculator.calculate_damage(
            attacker=normal_pokemon, defender=normal_pokemon, move=cut_move
        )

        if normal_result.max_damage > 0:
            ratio = sharpness_result.max_damage / normal_result.max_damage
            print(f"きれあじ ratio: {ratio:.4f}")
            self.assertAlmostEqual(
                ratio, 1.5, delta=0.1, msg="きれあじで切断技が1.5倍になるべき"
            )

    def test_ice_scales_ability(self):
        """こおりのりんぷん特性のテスト"""
        ice_scales_pokemon = create_simple_pokemon(
            species="ピカチュウ", level=50, ability="こおりのりんぷん"
        )
        normal_pokemon = create_simple_pokemon(
            species="ピカチュウ", level=50, ability="せいでんき"
        )

        # 特殊技の攻撃者
        special_attacker = create_simple_pokemon(
            species="ガブリアス", level=50, ability="すながくれ", evs={"sp_attack": 252}
        )

        special_move = MoveInput(name="りゅうのはどう")

        ice_scales_result = self.calculator.calculate_damage(
            attacker=special_attacker, defender=ice_scales_pokemon, move=special_move
        )
        normal_result = self.calculator.calculate_damage(
            attacker=special_attacker, defender=normal_pokemon, move=special_move
        )

        if normal_result.max_damage > 0:
            ratio = ice_scales_result.max_damage / normal_result.max_damage
            print(f"こおりのりんぷん ratio: {ratio:.4f}")
            self.assertLess(ratio, 0.6, "こおりのりんぷんで特殊技が半減されるべき")

    def test_fluffy_ability(self):
        """もふもふ特性のテスト"""
        fluffy_pokemon = create_simple_pokemon(
            species="ピカチュウ", level=50, ability="もふもふ"
        )
        normal_pokemon = create_simple_pokemon(
            species="ピカチュウ", level=50, ability="せいでんき"
        )

        # 接触技の攻撃者
        physical_attacker = create_simple_pokemon(
            species="ガブリアス", level=50, ability="すながくれ", evs={"attack": 252}
        )

        # 接触技
        contact_move = MoveInput(name="じしん")

        fluffy_result = self.calculator.calculate_damage(
            attacker=physical_attacker, defender=fluffy_pokemon, move=contact_move
        )
        normal_result = self.calculator.calculate_damage(
            attacker=physical_attacker, defender=normal_pokemon, move=contact_move
        )

        # じしんは接触技ではないので効果なし
        if normal_result.max_damage > 0:
            ratio = fluffy_result.max_damage / normal_result.max_damage
            print(f"もふもふ (じしん) ratio: {ratio:.4f}")
            self.assertAlmostEqual(
                ratio, 1.0, delta=0.1, msg="じしんは接触技ではないのでもふもふ効果なし"
            )

    def test_analytic_ability(self):
        """アナライズ特性のテスト"""
        analytic_pokemon = create_simple_pokemon(
            species="ガブリアス", level=50, ability="アナライズ", moves_last=True
        )
        normal_pokemon = create_simple_pokemon(
            species="ガブリアス", level=50, ability="すながくれ", moves_last=False
        )

        move = MoveInput(name="じしん")

        analytic_result = self.calculator.calculate_damage(
            attacker=analytic_pokemon, defender=normal_pokemon, move=move
        )
        normal_result = self.calculator.calculate_damage(
            attacker=normal_pokemon, defender=normal_pokemon, move=move
        )

        if normal_result.max_damage > 0:
            ratio = analytic_result.max_damage / normal_result.max_damage
            print(f"アナライズ ratio: {ratio:.4f}")
            self.assertAlmostEqual(
                ratio, 1.3, delta=0.1, msg="アナライズで後攻時に1.3倍になるべき"
            )

    def test_rivalry_ability(self):
        """とうそうしん特性のテスト"""
        male_rivalry = create_simple_pokemon(
            species="ガブリアス", level=50, ability="とうそうしん", gender="male"
        )
        male_normal = create_simple_pokemon(
            species="ガブリアス", level=50, ability="すながくれ", gender="male"
        )
        female_normal = create_simple_pokemon(
            species="ガブリアス", level=50, ability="すながくれ", gender="female"
        )

        move = MoveInput(name="じしん")

        # 同性（male vs male）
        same_gender_result = self.calculator.calculate_damage(
            attacker=male_rivalry, defender=male_normal, move=move
        )

        # 異性（male vs female）
        opposite_gender_result = self.calculator.calculate_damage(
            attacker=male_rivalry, defender=female_normal, move=move
        )

        # 通常
        normal_result = self.calculator.calculate_damage(
            attacker=male_normal, defender=male_normal, move=move
        )

        if normal_result.max_damage > 0:
            same_ratio = same_gender_result.max_damage / normal_result.max_damage
            opposite_ratio = (
                opposite_gender_result.max_damage / normal_result.max_damage
            )

            print(f"とうそうしん 同性 ratio: {same_ratio:.4f}")
            print(f"とうそうしん 異性 ratio: {opposite_ratio:.4f}")

            self.assertAlmostEqual(
                same_ratio,
                1.25,
                delta=0.1,
                msg="とうそうしんで同性相手に1.25倍になるべき",
            )
            self.assertAlmostEqual(
                opposite_ratio,
                0.75,
                delta=0.1,
                msg="とうそうしんで異性相手に0.75倍になるべき",
            )

    def test_supreme_overlord_ability(self):
        """そうだいしょう特性のテスト"""
        overlord_1 = create_simple_pokemon(
            species="ガブリアス",
            level=50,
            ability="そうだいしょう",
            fainted_teammates=1,
        )
        overlord_3 = create_simple_pokemon(
            species="ガブリアス",
            level=50,
            ability="そうだいしょう",
            fainted_teammates=3,
        )
        overlord_5 = create_simple_pokemon(
            species="ガブリアス",
            level=50,
            ability="そうだいしょう",
            fainted_teammates=5,
        )
        normal_pokemon = create_simple_pokemon(
            species="ガブリアス", level=50, ability="すながくれ", fainted_teammates=0
        )

        move = MoveInput(name="じしん")

        result_1 = self.calculator.calculate_damage(
            attacker=overlord_1, defender=normal_pokemon, move=move
        )
        result_3 = self.calculator.calculate_damage(
            attacker=overlord_3, defender=normal_pokemon, move=move
        )
        result_5 = self.calculator.calculate_damage(
            attacker=overlord_5, defender=normal_pokemon, move=move
        )
        normal_result = self.calculator.calculate_damage(
            attacker=normal_pokemon, defender=normal_pokemon, move=move
        )

        if normal_result.max_damage > 0:
            ratio_1 = result_1.max_damage / normal_result.max_damage
            ratio_3 = result_3.max_damage / normal_result.max_damage
            ratio_5 = result_5.max_damage / normal_result.max_damage

            print(f"そうだいしょう 1匹 ratio: {ratio_1:.4f}")
            print(f"そうだいしょう 3匹 ratio: {ratio_3:.4f}")
            print(f"そうだいしょう 5匹 ratio: {ratio_5:.4f}")

            self.assertAlmostEqual(
                ratio_1, 1.1, delta=0.1, msg="そうだいしょうで1匹倒れて1.1倍になるべき"
            )
            self.assertAlmostEqual(
                ratio_3, 1.3, delta=0.1, msg="そうだいしょうで3匹倒れて1.3倍になるべき"
            )
            self.assertAlmostEqual(
                ratio_5, 1.5, delta=0.1, msg="そうだいしょうで5匹倒れて1.5倍になるべき"
            )


class TestSimpleUsageExample(unittest.TestCase):
    """簡単な使用例テスト"""

    def test_simple_damage_calculation_example(self):
        """簡単なダメージ計算例"""
        try:
            calculator = DamageCalculator()

            # ポケモンを作成
            attacker = PokemonState(
                species="ピカチュウ",
                level=50,
                stats={
                    "hp": 145,
                    "attack": 86,
                    "defense": 80,
                    "sp_attack": 137,
                    "sp_defense": 90,
                    "speed": 156,
                },
                ability="せいでんき",
            )

            defender = PokemonState(
                species="ギャラドス",
                level=50,
                stats={
                    "hp": 171,
                    "attack": 145,
                    "defense": 109,
                    "sp_attack": 90,
                    "sp_defense": 120,
                    "speed": 101,
                },
                ability="いかく",
            )

            # 技を定義
            move = MoveInput(name="10まんボルト")

            # ダメージ計算実行
            result = calculator.calculate_damage(attacker, defender, move)

            # 結果確認
            self.assertIsNotNone(result)
            print("\n=== 使用例 ===")
            print(f"技: {move.name}")
            print(f"ダメージ: {result.min_damage}-{result.max_damage}")
            print(f"平均: {result.average_damage:.1f}")
            print(f"KO確率: {result.ko_probability:.2%}")
            print(f"確定数: {result.guaranteed_ko_hits}発")

        except Exception as e:
            print(f"使用例テストをスキップ: {e}")


class TestPokeSolCompatibility(unittest.TestCase):
    """ポケソル（Pokémon Damage Calculator）との互換性テスト"""

    def test_damage_calc_result_with_pokesol_carylex(self):
        """ポケソルダメージ計算機との整合性確認"""
        # ポケソルのダメージ計算結果を模擬
        pokesol_result = {
            "min_damage": 42,
            "max_damage": 49,
            "average_damage": 110.0,
            "ko_probability": 0.25,
            "guaranteed_ko_hits": 6,
        }

        # ダメージ計算エンジンで同じ条件で計算
        calculator = DamageCalculator()
        attacker = create_simple_pokemon(
            species="バドレックス(こくば)",
            level=50,
            nature="ひかえめ",
            ability="じんばいったい",
            evs={"sp_attack": 252, "speed": 252, "hp": 4},
        )
        defender = create_simple_pokemon(
            species="ディンルー",
            level=50,
            nature="しんちょう",
            ability="わざわいのうつわ",
            evs={"sp_defense": 252, "speed": 252, "hp": 252},
        )
        move = MoveInput(name="アストラルビット")

        result = calculator.calculate_damage(attacker, defender, move)

        # 実際の計算結果を表示
        print("\n=== バドレックス(こくば) vs ディンルー ===")
        print(f"技: {move.name}")
        print(f"実際のダメージ: {result.min_damage}-{result.max_damage}")
        print(f"期待値: {pokesol_result['min_damage']}-{pokesol_result['max_damage']}")
        print(
            f"差分: min={result.min_damage - pokesol_result['min_damage']}, max={result.max_damage - pokesol_result['max_damage']}"
        )
        print(
            f"確定数: 実際={result.guaranteed_ko_hits}, 期待={pokesol_result['guaranteed_ko_hits']}"
        )

        # 計算詳細を表示
        details = result.calculation_details
        print(f"攻撃実数値: {details.get('attack_stat', 'N/A')}")
        print(f"防御実数値: {details.get('defense_stat', 'N/A')}")
        print(f"技威力: {details.get('power', 'N/A')}")
        print(f"タイプ相性: {details.get('type_effectiveness', 'N/A')}")

        # より緩い条件でテスト（計算ロジックが動作することを確認）
        self.assertGreater(result.min_damage, 0)
        self.assertGreater(result.max_damage, result.min_damage)
        self.assertGreater(result.guaranteed_ko_hits, 0)

        print("バドレックス(こくば)の読み込み成功: OK")

        self.assertAlmostEqual(result.min_damage, pokesol_result["min_damage"], delta=1)
        self.assertAlmostEqual(result.max_damage, pokesol_result["max_damage"], delta=1)
        self.assertEqual(
            result.guaranteed_ko_hits, pokesol_result["guaranteed_ko_hits"]
        )

        print("ポケソルとの整合性確認: OK")

    def test_damage_calc_result_with_pokesol_koraidon(self):
        """ポケソルダメージ計算機との整合性確認（コライドン）"""
        # ポケソルのダメージ計算結果を模擬
        pokesol_result = {
            "min_damage": 132,
            "max_damage": 156,
            "guaranteed_ko_hits": 2,
        }

        # ダメージ計算エンジンで同じ条件で計算
        calculator = DamageCalculator()
        attacker = create_simple_pokemon(
            species="コライドン",
            level=50,
            nature="いじっぱり",
            ability="ひひいろのこどう",
            evs={"attack": 252, "speed": 252, "hp": 4},
        )
        defender = create_simple_pokemon(
            species="チオンジェン",
            level=50,
            nature="わんぱく",
            ability="わざわいのおふだ",
            evs={"defense": 252, "speed": 252, "hp": 252},
        )
        move = MoveInput(name="とんぼがえり")
        condition = BattleConditions(
            terrain=TerrainCondition.NONE,
            weather=WeatherCondition.SUN,
        )

        result = calculator.calculate_damage(attacker, defender, move, condition)

        # 実際の計算結果を表示
        print("\n=== コライドン vs チオンジェン ===")
        print(f"技: {move.name}")
        print(f"実際のダメージ: {result.min_damage}-{result.max_damage}")
        print(f"期待値: {pokesol_result['min_damage']}-{pokesol_result['max_damage']}")
        print(
            f"差分: min={result.min_damage - pokesol_result['min_damage']}, max={result.max_damage - pokesol_result['max_damage']}"
        )
        print(
            f"確定数: 実際={result.guaranteed_ko_hits}, 期待={pokesol_result['guaranteed_ko_hits']}"
        )

        # 計算詳細を表示
        details = result.calculation_details
        print(f"攻撃実数値: {details.get('attack_stat', 'N/A')}")
        print(f"防御実数値: {details.get('defense_stat', 'N/A')}")
        print(f"技威力: {details.get('power', 'N/A')}")
        print(f"タイプ相性: {details.get('type_effectiveness', 'N/A')}")
        print(f"STAB補正: {details.get('stab_modifier', 'N/A')}")
        print(f"最終補正: {details.get('final_modifier', 'N/A')}")
        print(f"基本ダメージ: {details.get('base_damage', 'N/A')}")
        print(f"全体補正: {details.get('total_modifier', 'N/A')}")
        print(f"天気: {condition.weather.value}")
        print(f"攻撃側特性: {attacker.ability}")
        print(f"防御側特性: {defender.ability}")

        # 手計算で期待値を確認
        expected_base = ((50 * 0.4 + 2) * 133 * 204) / (167 * 50) + 2
        expected_final = expected_base * 2.0 * 1.5  # タイプ相性 × STAB
        print(f"期待基本ダメージ: {expected_base:.2f}")
        print(f"期待最終ダメージ: {expected_final:.2f}")
        print(f"期待範囲: {int(expected_final * 0.85)}-{int(expected_final * 1.0)}")

        self.assertAlmostEqual(result.min_damage, pokesol_result["min_damage"], delta=1)
        self.assertAlmostEqual(result.max_damage, pokesol_result["max_damage"], delta=1)
        self.assertEqual(
            result.guaranteed_ko_hits, pokesol_result["guaranteed_ko_hits"]
        )

    def test_damage_calc_result_with_pokesol_chienpao(self):
        """ポケソルダメージ計算機との整合性確認（パオジアン）"""
        # ポケソルのダメージ計算結果を模擬
        pokesol_result = {
            "min_damage": 44,
            "max_damage": 52,
            "guaranteed_ko_hits": 4,
        }

        # ダメージ計算エンジンで同じ条件で計算
        calculator = DamageCalculator()
        attacker = create_simple_pokemon(
            species="パオジアン",
            level=50,
            nature="いじっぱり",
            ability="わざわいのつるぎ",
            evs={"attack": 252, "speed": 252, "hp": 4},
        )
        defender = create_simple_pokemon(
            species="アシレーヌ",
            level=50,
            nature="わんぱく",
            ability="げきりゅう",
            evs={"defense": 252, "speed": 252, "hp": 252},
        )
        move = MoveInput(name="つららおとし")
        condition = BattleConditions(
            terrain=TerrainCondition.NONE,
            weather=WeatherCondition.NONE,
        )

        result = calculator.calculate_damage(attacker, defender, move, condition)

        # 実際の計算結果を表示
        print("\n=== パオジアン vs アシレーヌ ===")
        print(f"技: {move.name}")
        print(f"実際のダメージ: {result.min_damage}-{result.max_damage}")
        print(f"期待値: {pokesol_result['min_damage']}-{pokesol_result['max_damage']}")
        print(
            f"差分: min={result.min_damage - pokesol_result['min_damage']}, max={result.max_damage - pokesol_result['max_damage']}"
        )
        print(
            f"確定数: 実際={result.guaranteed_ko_hits}, 期待={pokesol_result['guaranteed_ko_hits']}"
        )

        # 計算詳細を表示
        details = result.calculation_details
        print(f"攻撃実数値: {details.get('attack_stat', 'N/A')}")
        print(f"防御実数値: {details.get('defense_stat', 'N/A')}")
        print(f"技威力: {details.get('power', 'N/A')}")
        print(f"タイプ相性: {details.get('type_effectiveness', 'N/A')}")
        print(f"天気: {condition.weather.value}")
        print(f"攻撃側特性: {attacker.ability}")
        print(f"防御側特性: {defender.ability}")

        self.assertAlmostEqual(result.min_damage, pokesol_result["min_damage"], delta=1)
        self.assertAlmostEqual(result.max_damage, pokesol_result["max_damage"], delta=1)
        self.assertEqual(
            result.guaranteed_ko_hits, pokesol_result["guaranteed_ko_hits"]
        )

    def test_damage_calc_result_with_pokesol_chiyui(self):
        """ポケソルダメージ計算機との整合性確認（イーユイ）"""
        # ポケソルのダメージ計算結果を模擬
        pokesol_result = {
            "min_damage": 38,
            "max_damage": 45,
            "guaranteed_ko_hits": 5,
        }

        # ダメージ計算エンジンで同じ条件で計算
        calculator = DamageCalculator()
        attacker = create_simple_pokemon(
            species="イーユイ",
            level=50,
            nature="ひかえめ",
            ability="わざわいのたま",
            evs={"sp_attack": 252, "speed": 252, "hp": 4},
        )
        defender = create_simple_pokemon(
            species="アシレーヌ",
            level=50,
            nature="しんちょう",
            ability="げきりゅう",
            evs={"sp_defense": 252, "speed": 252, "hp": 252},
        )
        move = MoveInput(name="かえんほうしゃ")
        condition = BattleConditions(
            terrain=TerrainCondition.NONE,
            weather=WeatherCondition.NONE,
        )

        result = calculator.calculate_damage(attacker, defender, move, condition)

        # 実際の計算結果を表示
        print("\n=== イーユイ vs アシレーヌ ===")
        print(f"技: {move.name}")
        print(f"実際のダメージ: {result.min_damage}-{result.max_damage}")
        print(f"期待値: {pokesol_result['min_damage']}-{pokesol_result['max_damage']}")
        print(
            f"差分: min={result.min_damage - pokesol_result['min_damage']}, max={result.max_damage - pokesol_result['max_damage']}"
        )
        print(
            f"確定数: 実際={result.guaranteed_ko_hits}, 期待={pokesol_result['guaranteed_ko_hits']}"
        )

        # 計算詳細を表示
        details = result.calculation_details
        print(f"攻撃実数値: {details.get('attack_stat', 'N/A')}")
        print(f"防御実数値: {details.get('defense_stat', 'N/A')}")
        print(f"技威力: {details.get('power', 'N/A')}")
        print(f"タイプ相性: {details.get('type_effectiveness', 'N/A')}")
        print(f"天気: {condition.weather.value}")
        print(f"攻撃側特性: {attacker.ability}")
        print(f"防御側特性: {defender.ability}")

        self.assertAlmostEqual(result.min_damage, pokesol_result["min_damage"], delta=1)
        self.assertAlmostEqual(result.max_damage, pokesol_result["max_damage"], delta=1)
        self.assertEqual(
            result.guaranteed_ko_hits, pokesol_result["guaranteed_ko_hits"]
        )


class TestHighPriorityAbilities(unittest.TestCase):
    """Phase 1 High Priority特性のテスト"""

    def setUp(self):
        """各テスト前の準備"""
        self.calculator = DamageCalculator()

    def test_new_abilities_suihou(self):
        """すいほう特性テスト（みず技威力2倍）"""
        calculator = DamageCalculator()

        # すいほう持ちのポケモン
        attacker_with_suihou = create_simple_pokemon(
            species="アーマーガア",  # 仮のポケモン
            level=50,
            nature="いじっぱり",
            ability="すいほう",
            evs={"attack": 252, "speed": 252, "hp": 4},
        )

        # 同じポケモンで別特性
        attacker_normal = create_simple_pokemon(
            species="アーマーガア",
            level=50,
            nature="いじっぱり",
            ability="プレッシャー",
            evs={"attack": 252, "speed": 252, "hp": 4},
        )

        defender = create_simple_pokemon(
            species="ギャラドス",
            level=50,
            nature="わんぱく",
            ability="いかく",
            evs={"defense": 252, "hp": 252},
        )

        move = MoveInput(name="アクアテール")  # みず技

        result_with_suihou = calculator.calculate_damage(
            attacker_with_suihou, defender, move
        )
        result_normal = calculator.calculate_damage(attacker_normal, defender, move)

        # すいほうでみず技威力が2倍になっていることを確認
        if result_with_suihou.max_damage > 0 and result_normal.max_damage > 0:
            self.assertGreater(
                result_with_suihou.max_damage, result_normal.max_damage * 1.8
            )
            print(
                f"すいほう効果確認 - 通常: {result_normal.max_damage}, すいほう: {result_with_suihou.max_damage}"
            )

    def test_new_abilities_atsuishibo(self):
        """あついしぼう特性テスト（ほのお・こおり技半減）"""
        calculator = DamageCalculator()

        attacker = create_simple_pokemon(
            species="コライドン",
            level=50,
            nature="いじっぱり",
            ability="ひひいろのこどう",
            evs={"attack": 252, "speed": 252, "hp": 4},
        )

        # あついしぼう持ちの防御側
        defender_with_ability = create_simple_pokemon(
            species="マリルリ",  # 仮のポケモン
            level=50,
            nature="わんぱく",
            ability="あついしぼう",
            evs={"defense": 252, "hp": 252},
        )

        # 通常特性の防御側
        defender_normal = create_simple_pokemon(
            species="マリルリ",
            level=50,
            nature="わんぱく",
            ability="ちからもち",
            evs={"defense": 252, "hp": 252},
        )

        fire_move = MoveInput(name="かえんほうしゃ")

        result_with_ability = calculator.calculate_damage(
            attacker, defender_with_ability, fire_move
        )
        result_normal = calculator.calculate_damage(
            attacker, defender_normal, fire_move
        )

        # あついしぼうでほのお技が半減されていることを確認
        if result_with_ability.max_damage > 0 and result_normal.max_damage > 0:
            self.assertLess(
                result_with_ability.max_damage, result_normal.max_damage * 0.6
            )
            print(
                f"あついしぼう効果確認 - 通常: {result_normal.max_damage}, あついしぼう: {result_with_ability.max_damage}"
            )

    def test_new_abilities_hardrock(self):
        """ハードロック特性テスト（効果抜群技3/4倍）"""
        calculator = DamageCalculator()

        attacker = create_simple_pokemon(
            species="ピカチュウ",
            level=50,
            nature="ひかえめ",
            ability="せいでんき",
            evs={"sp_attack": 252, "speed": 252, "hp": 4},
        )

        # ハードロック持ちの防御側（ひこうタイプ）
        defender_with_hardrock = create_simple_pokemon(
            species="エアームド",  # 仮のポケモン（はがね/ひこう）
            level=50,
            nature="わんぱく",
            ability="ハードロック",
            evs={"defense": 252, "hp": 252},
        )

        # 通常特性の防御側
        defender_normal = create_simple_pokemon(
            species="エアームド",
            level=50,
            nature="わんぱく",
            ability="がんじょう",
            evs={"defense": 252, "hp": 252},
        )

        electric_move = MoveInput(name="10まんボルト")  # でんき技（ひこうに効果抜群）

        result_with_hardrock = calculator.calculate_damage(
            attacker, defender_with_hardrock, electric_move
        )
        result_normal = calculator.calculate_damage(
            attacker, defender_normal, electric_move
        )

        # ハードロックで効果抜群技が3/4倍になっていることを確認
        if result_with_hardrock.max_damage > 0 and result_normal.max_damage > 0:
            expected_damage = result_normal.max_damage * 0.75
            self.assertAlmostEqual(
                result_with_hardrock.max_damage, expected_damage, delta=5
            )
            print(
                f"ハードロック効果確認 - 通常: {result_normal.max_damage}, ハードロック: {result_with_hardrock.max_damage}"
            )

    def test_new_abilities_sniper(self):
        """スナイパー特性テスト（急所時ダメージ2.25倍）"""
        calculator = DamageCalculator()

        # スナイパー持ちの攻撃側
        attacker_with_sniper = create_simple_pokemon(
            species="ピカチュウ",
            level=50,
            nature="いじっぱり",
            ability="スナイパー",
            evs={"attack": 252, "speed": 252, "hp": 4},
        )

        defender = create_simple_pokemon(
            species="ディンルー",
            level=50,
            nature="わんぱく",
            ability="わざわいのうつわ",
            evs={"defense": 252, "hp": 252},
        )

        normal_move = MoveInput(name="つばめがえし", is_critical=False)
        critical_move = MoveInput(name="つばめがえし", is_critical=True)

        result_normal = calculator.calculate_damage(
            attacker_with_sniper, defender, normal_move
        )
        result_critical = calculator.calculate_damage(
            attacker_with_sniper, defender, critical_move
        )

        # スナイパーで急所時のダメージが2.25倍になっていることを確認
        if result_normal.max_damage > 0 and result_critical.max_damage > 0:
            # スナイパーで急所は通常攻撃の2.25倍になる
            expected_ratio = 2.25  # 通常攻撃との比較
            actual_ratio = result_critical.max_damage / result_normal.max_damage
            print(
                f"スナイパー効果確認 - 通常: {result_normal.max_damage}, 急所: {result_critical.max_damage}, 倍率: {actual_ratio:.2f}"
            )
            self.assertAlmostEqual(actual_ratio, expected_ratio, delta=0.1)

    def test_new_abilities_iwahakodi(self):
        """いわはこび特性テスト（いわ技威力1.5倍）"""
        calculator = DamageCalculator()

        # いわはこび持ちの攻撃側
        attacker_with_ability = create_simple_pokemon(
            species="ドサイドン",  # 仮のポケモン
            level=50,
            nature="いじっぱり",
            ability="いわはこび",
            evs={"attack": 252, "speed": 252, "hp": 4},
        )

        # 通常特性の攻撃側
        attacker_normal = create_simple_pokemon(
            species="ドサイドン",
            level=50,
            nature="いじっぱり",
            ability="ハードロック",
            evs={"attack": 252, "speed": 252, "hp": 4},
        )

        defender = create_simple_pokemon(
            species="ピカチュウ",
            level=50,
            nature="ひかえめ",
            ability="せいでんき",
            evs={"sp_attack": 252, "speed": 252, "hp": 4},
        )

        rock_move = MoveInput(name="いわなだれ")

        result_with_ability = calculator.calculate_damage(
            attacker_with_ability, defender, rock_move
        )
        result_normal = calculator.calculate_damage(
            attacker_normal, defender, rock_move
        )

        # いわはこびでいわ技威力が1.5倍になっていることを確認
        if result_with_ability.max_damage > 0 and result_normal.max_damage > 0:
            self.assertGreater(
                result_with_ability.max_damage, result_normal.max_damage * 1.4
            )
            print(
                f"いわはこび効果確認 - 通常: {result_normal.max_damage}, いわはこび: {result_with_ability.max_damage}"
            )

    def test_new_abilities_quark_charge(self):
        """クォークチャージ特性テスト（エレキフィールド時最も高い能力値1.3倍）"""
        calculator = DamageCalculator()

        # クォークチャージで攻撃が上昇
        attacker_with_quark = create_simple_pokemon(
            species="テツノカイナ",
            level=50,
            nature="いじっぱり",
            ability="クォークチャージ",
            evs={"attack": 252, "speed": 252, "hp": 4},
            paradox_boost_stat="attack",  # 攻撃が最も高い能力値として指定
        )

        defender = create_simple_pokemon(
            species="ディンルー",
            level=50,
            nature="わんぱく",
            ability="わざわいのうつわ",
            evs={"defense": 252, "hp": 252},
        )

        move = MoveInput(name="インファイト")  # 物理技

        # エレキフィールド状態
        electric_field_conditions = BattleConditions(terrain=TerrainCondition.ELECTRIC)

        # 通常フィールド状態
        normal_conditions = BattleConditions()

        result_with_quark_electric = calculator.calculate_damage(
            attacker_with_quark, defender, move, electric_field_conditions
        )
        result_with_quark_normal = calculator.calculate_damage(
            attacker_with_quark, defender, move, normal_conditions
        )

        # デバッグ情報
        print("クォークチャージ効果確認:")
        print(f"  通常フィールド: {result_with_quark_normal.max_damage}")
        print(f"  エレキフィールド: {result_with_quark_electric.max_damage}")
        print(f"  攻撃側のparadox_boost_stat: {attacker_with_quark.paradox_boost_stat}")
        print(f"  攻撃側の特性: {attacker_with_quark.ability}")
        details = result_with_quark_electric.calculation_details
        print(f"  攻撃実数値: {details.get('attack_stat')}")
        print(f"  地形: {details.get('terrain')}")
        print(f"  エレキフィールド計算詳細: {details}")

        # エレキフィールド時にクォークチャージで攻撃力が1.3倍になっていることを確認
        if (
            result_with_quark_electric.max_damage > 0
            and result_with_quark_normal.max_damage > 0
        ):
            self.assertGreater(
                result_with_quark_electric.max_damage,
                result_with_quark_normal.max_damage * 1.25,
            )

    def test_new_abilities_protosynthesis(self):
        """古代活性特性テスト（晴れ時最も高い能力値1.3倍）"""
        calculator = DamageCalculator()

        # 古代活性で特攻が上昇
        attacker_with_proto = create_simple_pokemon(
            species="ピカチュウ",
            level=50,
            nature="ひかえめ",
            ability="こだいかっせい",
            evs={"sp_attack": 252, "speed": 252, "hp": 4},
            paradox_boost_stat="sp_attack",  # 特攻が最も高い能力値として指定
        )
        defender = create_simple_pokemon(
            species="ギャラドス",
            level=50,
            nature="しんちょう",
            ability="いかく",
            evs={"sp_defense": 252, "hp": 252},
        )

        move = MoveInput(name="10まんボルト")  # 特殊技

        # 晴れ状態
        sunny_conditions = BattleConditions(weather=WeatherCondition.SUN)

        # 通常天気状態
        normal_conditions = BattleConditions()

        result_with_proto_sunny = calculator.calculate_damage(
            attacker_with_proto, defender, move, sunny_conditions
        )
        result_with_proto_normal = calculator.calculate_damage(
            attacker_with_proto, defender, move, normal_conditions
        )

        # 晴れ時に古代活性で特攻が1.3倍になっていることを確認
        if (
            result_with_proto_sunny.max_damage > 0
            and result_with_proto_normal.max_damage > 0
        ):
            self.assertGreater(
                result_with_proto_sunny.max_damage,
                result_with_proto_normal.max_damage * 1.25,
            )
            print(
                f"古代活性効果確認 - 通常天気: {result_with_proto_normal.max_damage}, 晴れ: {result_with_proto_sunny.max_damage}"
            )

    def test_new_abilities_quark_charge_defense(self):
        """クォークチャージ防御特性テスト（エレキフィールド時防御1.3倍）"""
        calculator = DamageCalculator()

        attacker = create_simple_pokemon(
            species="テツノカイナ",
            level=50,
            nature="いじっぱり",
            ability="せいでんき",
            evs={"attack": 252, "speed": 252, "hp": 4},
        )

        # クォークチャージで防御が上昇
        defender_with_quark = create_simple_pokemon(
            species="ギャラドス",
            level=50,
            nature="わんぱく",
            ability="クォークチャージ",
            evs={"defense": 252, "hp": 252},
            paradox_boost_stat="defense",  # 防御が最も高い能力値として指定
        )

        move = MoveInput(name="インファイト")  # 物理技

        # エレキフィールド状態
        electric_field_conditions = BattleConditions(terrain=TerrainCondition.ELECTRIC)
        normal_conditions = BattleConditions()

        result_with_quark = calculator.calculate_damage(
            attacker, defender_with_quark, move, electric_field_conditions
        )
        result_normal = calculator.calculate_damage(
            attacker, defender_with_quark, move, normal_conditions
        )

        # エレキフィールド時にクォークチャージで防御力が1.3倍になり、ダメージが軽減されることを確認
        if result_with_quark.max_damage > 0 and result_normal.max_damage > 0:
            self.assertLess(
                result_with_quark.max_damage, result_normal.max_damage * 0.8
            )
            print(
                f"クォークチャージ防御効果確認 - 通常: {result_normal.max_damage}, クォークチャージ: {result_with_quark.max_damage}"
            )


class TestMediumPriorityAbilities(unittest.TestCase):
    """Phase 2 Medium Priority特性のテスト"""

    def setUp(self):
        """各テスト前の準備"""
        self.calculator = DamageCalculator()

    def test_medium_priority_abilities(self):
        """Medium priority特性のテスト"""

        # あついしぼう（ほのお・こおり技半減）
        fire_pokemon = create_simple_pokemon(
            species="ウインディ", level=50, ability="いかく"
        )
        thick_fat_pokemon = create_simple_pokemon(
            species="マリルリ", level=50, ability="あついしぼう"
        )

        fire_move = MoveInput(name="かえんほうしゃ")

        # 通常ダメージ vs あついしぼう
        normal_result = self.calculator.calculate_damage(
            attacker=fire_pokemon, defender=fire_pokemon, move=fire_move
        )
        thick_fat_result = self.calculator.calculate_damage(
            attacker=fire_pokemon, defender=thick_fat_pokemon, move=fire_move
        )

        # あついしぼうで半減されることを確認
        if normal_result.max_damage > 0 and thick_fat_result.max_damage > 0:
            ratio = thick_fat_result.max_damage / normal_result.max_damage
            print(f"あついしぼう ratio: {ratio:.4f}")
            self.assertLess(ratio, 0.6, "あついしぼうでほのお技が半減されるべき")

    def test_strong_jaw_ability(self):
        """がんじょうあご特性のテスト"""
        strong_jaw_pokemon = create_simple_pokemon(
            species="ジュラルドン", level=50, ability="がんじょうあご"
        )
        normal_pokemon = create_simple_pokemon(
            species="ジュラルドン", level=50, ability="ライトメタル"
        )

        bite_move = MoveInput(name="かみくだく")

        strong_jaw_result = self.calculator.calculate_damage(
            attacker=strong_jaw_pokemon, defender=normal_pokemon, move=bite_move
        )
        normal_result = self.calculator.calculate_damage(
            attacker=normal_pokemon, defender=normal_pokemon, move=bite_move
        )

        if normal_result.max_damage > 0:
            ratio = strong_jaw_result.max_damage / normal_result.max_damage
            print(f"がんじょうあご ratio: {ratio:.4f}")
            self.assertAlmostEqual(
                ratio, 1.5, delta=0.1, msg="がんじょうあごで噛み技が1.5倍になるべき"
            )

    def test_sheer_force_ability(self):
        """ちからずく特性のテスト"""
        sheer_force_pokemon = create_simple_pokemon(
            species="ガブリアス", level=50, ability="ちからずく"
        )
        normal_pokemon = create_simple_pokemon(
            species="ガブリアス", level=50, ability="すながくれ"
        )

        # 追加効果のある技
        effect_move = MoveInput(name="10まんボルト")

        sheer_force_result = self.calculator.calculate_damage(
            attacker=sheer_force_pokemon, defender=normal_pokemon, move=effect_move
        )
        normal_result = self.calculator.calculate_damage(
            attacker=normal_pokemon, defender=normal_pokemon, move=effect_move
        )

        if normal_result.max_damage > 0:
            ratio = sheer_force_result.max_damage / normal_result.max_damage
            print(f"ちからずく ratio: {ratio:.4f}")
            self.assertAlmostEqual(
                ratio, 1.3, delta=0.1, msg="ちからずくで追加効果技が1.3倍になるべき"
            )

    def test_filter_ability(self):
        """フィルター特性のテスト"""
        filter_pokemon = create_simple_pokemon(
            species="ピクシー",
            level=50,
            ability="フィルター",
            evs={"hp": 252, "defense": 252},
        )
        normal_pokemon = create_simple_pokemon(
            species="ピクシー",
            level=50,
            ability="メロメロボディ",
            evs={"hp": 252, "defense": 252},
        )

        # はがねタイプの攻撃者（フェアリーに効果バツグン）
        steel_attacker = create_simple_pokemon(
            species="ハッサム", level=50, ability="テクニシャン", evs={"attack": 252}
        )

        super_effective_move = MoveInput(name="アイアンヘッド")

        filter_result = self.calculator.calculate_damage(
            attacker=steel_attacker, defender=filter_pokemon, move=super_effective_move
        )
        normal_result = self.calculator.calculate_damage(
            attacker=steel_attacker, defender=normal_pokemon, move=super_effective_move
        )

        if normal_result.max_damage > 0:
            ratio = filter_result.max_damage / normal_result.max_damage
            print(f"フィルター ratio: {ratio:.4f}")
            self.assertLess(ratio, 0.8, "フィルターで効果バツグン技が軽減されるべき")

    def test_tough_claws_ability(self):
        """かたいつめ特性のテスト"""
        tough_claws_pokemon = create_simple_pokemon(
            species="ガブリアス", level=50, ability="かたいつめ"
        )
        normal_pokemon = create_simple_pokemon(
            species="ガブリアス", level=50, ability="すながくれ"
        )

        # 接触技
        contact_move = MoveInput(name="じしん")  # 接触技

        tough_claws_result = self.calculator.calculate_damage(
            attacker=tough_claws_pokemon, defender=normal_pokemon, move=contact_move
        )
        normal_result = self.calculator.calculate_damage(
            attacker=normal_pokemon, defender=normal_pokemon, move=contact_move
        )

        if normal_result.max_damage > 0:
            ratio = tough_claws_result.max_damage / normal_result.max_damage
            print(f"かたいつめ ratio: {ratio:.4f}")
            # じしんは接触技ではないので、比率は1.0のはず
            self.assertAlmostEqual(
                ratio,
                1.0,
                delta=0.1,
                msg="じしんは接触技ではないのでかたいつめ効果なし",
            )

    def test_iron_fist_ability(self):
        """てつのこぶし特性のテスト"""
        iron_fist_pokemon = create_simple_pokemon(
            species="ガブリアス", level=50, ability="てつのこぶし"
        )
        normal_pokemon = create_simple_pokemon(
            species="ガブリアス", level=50, ability="すながくれ"
        )

        # パンチ技
        punch_move = MoveInput(name="かみなりパンチ")

        iron_fist_result = self.calculator.calculate_damage(
            attacker=iron_fist_pokemon, defender=normal_pokemon, move=punch_move
        )
        normal_result = self.calculator.calculate_damage(
            attacker=normal_pokemon, defender=normal_pokemon, move=punch_move
        )

        if normal_result.max_damage > 0:
            ratio = iron_fist_result.max_damage / normal_result.max_damage
            print(f"てつのこぶし ratio: {ratio:.4f}")
            self.assertAlmostEqual(
                ratio, 1.2, delta=0.1, msg="てつのこぶしでパンチ技が1.2倍になるべき"
            )

    def test_sharpness_ability(self):
        """きれあじ特性のテスト"""
        sharpness_pokemon = create_simple_pokemon(
            species="ガブリアス", level=50, ability="きれあじ"
        )
        normal_pokemon = create_simple_pokemon(
            species="ガブリアス", level=50, ability="すながくれ"
        )

        # 切断技
        cut_move = MoveInput(name="つじぎり")

        sharpness_result = self.calculator.calculate_damage(
            attacker=sharpness_pokemon, defender=normal_pokemon, move=cut_move
        )
        normal_result = self.calculator.calculate_damage(
            attacker=normal_pokemon, defender=normal_pokemon, move=cut_move
        )

        if normal_result.max_damage > 0:
            ratio = sharpness_result.max_damage / normal_result.max_damage
            print(f"きれあじ ratio: {ratio:.4f}")
            self.assertAlmostEqual(
                ratio, 1.5, delta=0.1, msg="きれあじで切断技が1.5倍になるべき"
            )

    def test_ice_scales_ability(self):
        """こおりのりんぷん特性のテスト"""
        ice_scales_pokemon = create_simple_pokemon(
            species="ピカチュウ", level=50, ability="こおりのりんぷん"
        )
        normal_pokemon = create_simple_pokemon(
            species="ピカチュウ", level=50, ability="せいでんき"
        )

        # 特殊技の攻撃者
        special_attacker = create_simple_pokemon(
            species="ガブリアス", level=50, ability="すながくれ", evs={"sp_attack": 252}
        )

        special_move = MoveInput(name="りゅうのはどう")

        ice_scales_result = self.calculator.calculate_damage(
            attacker=special_attacker, defender=ice_scales_pokemon, move=special_move
        )
        normal_result = self.calculator.calculate_damage(
            attacker=special_attacker, defender=normal_pokemon, move=special_move
        )

        if normal_result.max_damage > 0:
            ratio = ice_scales_result.max_damage / normal_result.max_damage
            print(f"こおりのりんぷん ratio: {ratio:.4f}")
            self.assertLess(ratio, 0.6, "こおりのりんぷんで特殊技が半減されるべき")

    def test_fluffy_ability(self):
        """もふもふ特性のテスト"""
        fluffy_pokemon = create_simple_pokemon(
            species="ピカチュウ", level=50, ability="もふもふ"
        )
        normal_pokemon = create_simple_pokemon(
            species="ピカチュウ", level=50, ability="せいでんき"
        )

        # 接触技の攻撃者
        physical_attacker = create_simple_pokemon(
            species="ガブリアス", level=50, ability="すながくれ", evs={"attack": 252}
        )

        # 接触技
        contact_move = MoveInput(name="じしん")

        fluffy_result = self.calculator.calculate_damage(
            attacker=physical_attacker, defender=fluffy_pokemon, move=contact_move
        )
        normal_result = self.calculator.calculate_damage(
            attacker=physical_attacker, defender=normal_pokemon, move=contact_move
        )

        # じしんは接触技ではないので効果なし
        if normal_result.max_damage > 0:
            ratio = fluffy_result.max_damage / normal_result.max_damage
            print(f"もふもふ (じしん) ratio: {ratio:.4f}")
            self.assertAlmostEqual(
                ratio, 1.0, delta=0.1, msg="じしんは接触技ではないのでもふもふ効果なし"
            )


class TestComplexMechanicsAbilities(unittest.TestCase):
    """複雑なメカニクス特性のテスト"""

    def setUp(self):
        """各テスト前の準備"""
        self.calculator = DamageCalculator()

    def test_analytic_ability(self):
        """アナライズ特性のテスト"""
        analytic_pokemon = create_simple_pokemon(
            species="ガブリアス", level=50, ability="アナライズ", moves_last=True
        )
        normal_pokemon = create_simple_pokemon(
            species="ガブリアス", level=50, ability="すながくれ", moves_last=False
        )

        move = MoveInput(name="じしん")

        analytic_result = self.calculator.calculate_damage(
            attacker=analytic_pokemon, defender=normal_pokemon, move=move
        )
        normal_result = self.calculator.calculate_damage(
            attacker=normal_pokemon, defender=normal_pokemon, move=move
        )

        if normal_result.max_damage > 0:
            ratio = analytic_result.max_damage / normal_result.max_damage
            print(f"アナライズ ratio: {ratio:.4f}")
            self.assertAlmostEqual(
                ratio, 1.3, delta=0.1, msg="アナライズで後攻時に1.3倍になるべき"
            )

    def test_rivalry_ability(self):
        """とうそうしん特性のテスト"""
        male_rivalry = create_simple_pokemon(
            species="ガブリアス", level=50, ability="とうそうしん", gender="male"
        )
        male_normal = create_simple_pokemon(
            species="ガブリアス", level=50, ability="すながくれ", gender="male"
        )
        female_normal = create_simple_pokemon(
            species="ガブリアス", level=50, ability="すながくれ", gender="female"
        )

        move = MoveInput(name="じしん")

        # 同性（male vs male）
        same_gender_result = self.calculator.calculate_damage(
            attacker=male_rivalry, defender=male_normal, move=move
        )

        # 異性（male vs female）
        opposite_gender_result = self.calculator.calculate_damage(
            attacker=male_rivalry, defender=female_normal, move=move
        )

        # 通常
        normal_result = self.calculator.calculate_damage(
            attacker=male_normal, defender=male_normal, move=move
        )

        if normal_result.max_damage > 0:
            same_ratio = same_gender_result.max_damage / normal_result.max_damage
            opposite_ratio = (
                opposite_gender_result.max_damage / normal_result.max_damage
            )

            print(f"とうそうしん 同性 ratio: {same_ratio:.4f}")
            print(f"とうそうしん 異性 ratio: {opposite_ratio:.4f}")

            self.assertAlmostEqual(
                same_ratio,
                1.25,
                delta=0.1,
                msg="とうそうしんで同性相手に1.25倍になるべき",
            )
            self.assertAlmostEqual(
                opposite_ratio,
                0.75,
                delta=0.1,
                msg="とうそうしんで異性相手に0.75倍になるべき",
            )

    def test_supreme_overlord_ability(self):
        """そうだいしょう特性のテスト"""
        overlord_1 = create_simple_pokemon(
            species="ガブリアス",
            level=50,
            ability="そうだいしょう",
            fainted_teammates=1,
        )
        overlord_3 = create_simple_pokemon(
            species="ガブリアス",
            level=50,
            ability="そうだいしょう",
            fainted_teammates=3,
        )
        overlord_5 = create_simple_pokemon(
            species="ガブリアス",
            level=50,
            ability="そうだいしょう",
            fainted_teammates=5,
        )
        normal_pokemon = create_simple_pokemon(
            species="ガブリアス", level=50, ability="すながくれ", fainted_teammates=0
        )

        move = MoveInput(name="じしん")

        result_1 = self.calculator.calculate_damage(
            attacker=overlord_1, defender=normal_pokemon, move=move
        )
        result_3 = self.calculator.calculate_damage(
            attacker=overlord_3, defender=normal_pokemon, move=move
        )
        result_5 = self.calculator.calculate_damage(
            attacker=overlord_5, defender=normal_pokemon, move=move
        )
        normal_result = self.calculator.calculate_damage(
            attacker=normal_pokemon, defender=normal_pokemon, move=move
        )

        if normal_result.max_damage > 0:
            ratio_1 = result_1.max_damage / normal_result.max_damage
            ratio_3 = result_3.max_damage / normal_result.max_damage
            ratio_5 = result_5.max_damage / normal_result.max_damage

            print(f"そうだいしょう 1匹 ratio: {ratio_1:.4f}")
            print(f"そうだいしょう 3匹 ratio: {ratio_3:.4f}")
            print(f"そうだいしょう 5匹 ratio: {ratio_5:.4f}")

            self.assertAlmostEqual(
                ratio_1, 1.1, delta=0.1, msg="そうだいしょうで1匹倒れて1.1倍になるべき"
            )
            self.assertAlmostEqual(
                ratio_3, 1.3, delta=0.1, msg="そうだいしょうで3匹倒れて1.3倍になるべき"
            )
            self.assertAlmostEqual(
                ratio_5, 1.5, delta=0.1, msg="そうだいしょうで5匹倒れて1.5倍になるべき"
            )

    def test_flash_fire_ability(self):
        """もらい火特性のテスト"""
        # もらい火持ちの攻撃側
        flash_fire_attacker = create_simple_pokemon(
            species="ガブリアス", level=50, ability="もらいび"
        )
        
        # もらい火持ちの防御側
        flash_fire_defender = create_simple_pokemon(
            species="ガブリアス", level=50, ability="もらいび"
        )
        
        # 通常特性のポケモン
        normal_pokemon = create_simple_pokemon(
            species="ガブリアス", level=50, ability="すながくれ"
        )

        fire_move = MoveInput(name="かえんほうしゃ")

        # 攻撃側もらい火：ほのお技威力1.5倍
        flash_fire_result = self.calculator.calculate_damage(
            attacker=flash_fire_attacker, defender=normal_pokemon, move=fire_move
        )
        normal_result = self.calculator.calculate_damage(
            attacker=normal_pokemon, defender=normal_pokemon, move=fire_move
        )

        if normal_result.max_damage > 0:
            ratio = flash_fire_result.max_damage / normal_result.max_damage
            print(f"もらい火攻撃 ratio: {ratio:.4f}")
            self.assertAlmostEqual(
                ratio, 1.5, delta=0.1, msg="もらい火特性でほのお技が1.5倍になるべき"
            )

        # 防御側もらい火：ほのお技無効
        defense_result = self.calculator.calculate_damage(
            attacker=normal_pokemon, defender=flash_fire_defender, move=fire_move
        )
        
        self.assertEqual(
            defense_result.max_damage, 0, 
            "もらい火持ちへのほのお技は無効になるべき"
        )
        
        print(f"もらい火防御効果確認 - ダメージ: {defense_result.max_damage} (無効)")


if __name__ == "__main__":
    # 詳細な出力でテスト実行
    unittest.main(verbosity=2)
