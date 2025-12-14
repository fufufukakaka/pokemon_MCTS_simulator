"""
MoveEffectivenessCalculator のテスト

特に以下のケースを検証:
1. ふうせんによる地面技無効
2. おうごんのからだによる変化技無効
3. タイプ相性による無効（ノーマル→ゴースト等）
4. 特性による無効化（ふゆう、ちくでん等）
"""

import unittest
from pathlib import Path

# Pokemon.init() を呼ぶためにまずデータをロード
from src.pokemon_battle_sim.pokemon import Pokemon

# テスト前にPokemonを初期化
Pokemon.init()

from src.rebel.move_effectiveness import (
    MoveEffectivenessCalculator,
    MoveEffectivenessResult,
    encode_move_effectiveness_flags,
)


class TestMoveEffectivenessCalculator(unittest.TestCase):
    """MoveEffectivenessCalculator のユニットテスト"""

    def setUp(self):
        self.calculator = MoveEffectivenessCalculator()

    def test_balloon_blocks_ground(self):
        """ふうせんは地面技を無効化"""
        # ひこうタイプを持たないポケモンでテスト
        result = self.calculator.check_move_effectiveness(
            move_name="じしん",
            defender_types=["はがね", "ゴースト"],  # サーフゴーのタイプ
            defender_ability=None,
            defender_item="ふうせん",
            gravity=False,
        )
        self.assertFalse(result.is_effective)
        self.assertEqual(result.effectiveness, 0.0)
        self.assertEqual(result.reason, "balloon_blocks_ground")

    def test_balloon_with_gravity(self):
        """重力下ではふうせんは地面技を防げない"""
        result = self.calculator.check_move_effectiveness(
            move_name="じしん",
            defender_types=["はがね", "ゴースト"],  # サーフゴーのタイプ
            defender_ability=None,
            defender_item="ふうせん",
            gravity=True,
        )
        self.assertTrue(result.is_effective)
        self.assertGreater(result.effectiveness, 0.0)

    def test_golden_body_blocks_status(self):
        """おうごんのからだは変化技を無効化"""
        result = self.calculator.check_move_effectiveness(
            move_name="どくどく",
            defender_types=["はがね", "ゴースト"],  # サーフゴーのタイプ
            defender_ability="おうごんのからだ",
            defender_item=None,
            gravity=False,
        )
        self.assertFalse(result.is_effective)
        self.assertEqual(result.effectiveness, 0.0)
        self.assertEqual(result.reason, "golden_body_blocks_status")

    def test_golden_body_allows_damaging_moves(self):
        """おうごんのからだはダメージ技は通す"""
        result = self.calculator.check_move_effectiveness(
            move_name="シャドーボール",
            defender_types=["はがね", "ゴースト"],
            defender_ability="おうごんのからだ",
            defender_item=None,
            gravity=False,
        )
        self.assertTrue(result.is_effective)
        self.assertGreater(result.effectiveness, 0.0)

    def test_type_immunity_normal_ghost(self):
        """ノーマル技はゴーストタイプに無効"""
        result = self.calculator.check_move_effectiveness(
            move_name="すてみタックル",
            defender_types=["ゴースト"],
            defender_ability=None,
            defender_item=None,
            gravity=False,
        )
        self.assertFalse(result.is_effective)
        self.assertEqual(result.effectiveness, 0.0)
        self.assertEqual(result.reason, "type_immunity")

    def test_type_immunity_ground_flying(self):
        """地面技はひこうタイプに無効"""
        result = self.calculator.check_move_effectiveness(
            move_name="じしん",
            defender_types=["ひこう"],
            defender_ability=None,
            defender_item=None,
            gravity=False,
        )
        self.assertFalse(result.is_effective)
        self.assertEqual(result.effectiveness, 0.0)
        # タイプ相性で無効化される（type_immunity）か、
        # ひこうタイプチェックで無効化される（flying_type_blocks_ground）のどちらか
        self.assertIn(result.reason, ["type_immunity", "flying_type_blocks_ground"])

    def test_levitate_blocks_ground(self):
        """ふゆう特性は地面技を無効化"""
        result = self.calculator.check_move_effectiveness(
            move_name="じしん",
            defender_types=["ゴースト"],  # ゲンガーのタイプ（第一タイプのみ）
            defender_ability="ふゆう",
            defender_item=None,
            gravity=False,
        )
        self.assertFalse(result.is_effective)
        self.assertEqual(result.reason, "levitate_blocks_ground")

    def test_volt_absorb_blocks_electric(self):
        """ちくでんは電気技を無効化"""
        result = self.calculator.check_move_effectiveness(
            move_name="10まんボルト",
            defender_types=["みず"],
            defender_ability="ちくでん",
            defender_item=None,
            gravity=False,
        )
        self.assertFalse(result.is_effective)
        self.assertEqual(result.reason, "ability_blocks_electric")

    def test_flash_fire_blocks_fire(self):
        """もらいびは炎技を無効化"""
        result = self.calculator.check_move_effectiveness(
            move_name="かえんほうしゃ",
            defender_types=["ほのお"],
            defender_ability="もらいび",
            defender_item=None,
            gravity=False,
        )
        self.assertFalse(result.is_effective)
        self.assertEqual(result.reason, "ability_blocks_fire")

    def test_sap_sipper_blocks_grass(self):
        """そうしょくは草技を無効化"""
        result = self.calculator.check_move_effectiveness(
            move_name="リーフブレード",
            defender_types=["ノーマル"],
            defender_ability="そうしょく",
            defender_item=None,
            gravity=False,
        )
        self.assertFalse(result.is_effective)
        self.assertEqual(result.reason, "ability_blocks_grass")

    def test_super_effective(self):
        """効果抜群のケース"""
        result = self.calculator.check_move_effectiveness(
            move_name="シャドーボール",
            defender_types=["エスパー"],
            defender_ability=None,
            defender_item=None,
            gravity=False,
        )
        self.assertTrue(result.is_effective)
        self.assertEqual(result.effectiveness, 2.0)

    def test_not_very_effective(self):
        """効果いまひとつのケース"""
        result = self.calculator.check_move_effectiveness(
            move_name="かえんほうしゃ",
            defender_types=["みず"],
            defender_ability=None,
            defender_item=None,
            gravity=False,
        )
        self.assertTrue(result.is_effective)
        self.assertEqual(result.effectiveness, 0.5)

    def test_has_effective_move_true(self):
        """有効な技がある場合"""
        has_effective = self.calculator.has_effective_move(
            moves=["じしん", "シャドーボール", "まもる", "どくどく"],
            defender_types=["はがね", "ゴースト"],
            defender_ability="おうごんのからだ",
            defender_item="ふうせん",
            gravity=False,
        )
        # シャドーボールは有効（ゴーストにゴースト技は抜群）
        self.assertTrue(has_effective)

    def test_has_effective_move_false(self):
        """有効な技がない場合（グライオン vs ふうせんおうごんのからだサーフゴー）"""
        has_effective = self.calculator.has_effective_move(
            moves=["じしん", "どくどく", "まもる", "みがわり"],
            defender_types=["はがね", "ゴースト"],
            defender_ability="おうごんのからだ",
            defender_item="ふうせん",
            gravity=False,
        )
        # じしん: ふうせんで無効
        # どくどく: おうごんのからだで無効
        # まもる: 変化技でおうごんのからだで無効
        # みがわり: 変化技でおうごんのからだで無効
        self.assertFalse(has_effective)

    def test_count_effective_moves(self):
        """有効な技の数をカウント"""
        count = self.calculator.count_effective_moves(
            moves=["じしん", "シャドーボール", "まもる", "どくどく"],
            defender_types=["はがね", "ゴースト"],
            defender_ability="おうごんのからだ",
            defender_item="ふうせん",
            gravity=False,
        )
        # シャドーボールのみ有効
        self.assertEqual(count, 1)


class TestEncodeMoveFlagsFunction(unittest.TestCase):
    """encode_move_effectiveness_flags 関数のテスト"""

    def test_encode_all_effective(self):
        """すべての技が有効なケース"""
        flags = encode_move_effectiveness_flags(
            attacker_moves=["10まんボルト", "れいとうビーム", "シャドーボール", "サイコキネシス"],
            defender_types=["みず"],
            defender_ability=None,
            defender_item=None,
            gravity=False,
        )
        # 長さ: 4(有効フラグ) + 4(有効性) + 1(有効技あり) = 9
        self.assertEqual(len(flags), 9)
        # すべての技が有効
        self.assertEqual(flags[0], 1.0)
        self.assertEqual(flags[1], 1.0)
        self.assertEqual(flags[2], 1.0)
        self.assertEqual(flags[3], 1.0)
        # 有効技あり
        self.assertEqual(flags[8], 1.0)

    def test_encode_no_effective(self):
        """有効な技がないケース"""
        flags = encode_move_effectiveness_flags(
            attacker_moves=["じしん", "どくどく", "まもる", "みがわり"],
            defender_types=["はがね", "ゴースト"],
            defender_ability="おうごんのからだ",
            defender_item="ふうせん",
            gravity=False,
        )
        # すべての技が無効
        self.assertEqual(flags[0], 0.0)
        self.assertEqual(flags[1], 0.0)
        self.assertEqual(flags[2], 0.0)
        self.assertEqual(flags[3], 0.0)
        # 有効技なし
        self.assertEqual(flags[8], 0.0)


class TestRealBattleScenario(unittest.TestCase):
    """実際のバトルシナリオのテスト"""

    def setUp(self):
        self.calculator = MoveEffectivenessCalculator()

    def test_gliscor_vs_balloon_gholdengo(self):
        """
        グライオン vs ふうせんサーフゴー

        グライオンの技:
        - まもる (変化技) → おうごんのからだで無効
        - みがわり (変化技) → おうごんのからだで無効
        - じしん (地面技) → ふうせんで無効
        - どくどく (変化技) → おうごんのからだで無効

        結果: グライオンの全技が無効 → 詰み
        """
        gliscor_moves = ["まもる", "みがわり", "じしん", "どくどく"]
        gholdengo_types = ["はがね", "ゴースト"]
        gholdengo_ability = "おうごんのからだ"
        gholdengo_item = "ふうせん"

        results = self.calculator.check_all_moves_effectiveness(
            moves=gliscor_moves,
            defender_types=gholdengo_types,
            defender_ability=gholdengo_ability,
            defender_item=gholdengo_item,
            gravity=False,
        )

        # すべての技が無効
        for result in results:
            self.assertFalse(
                result.is_effective,
                f"{result.move_name} should be ineffective, but reason={result.reason}"
            )

        # 有効な技が0
        self.assertFalse(
            self.calculator.has_effective_move(
                gliscor_moves,
                gholdengo_types,
                gholdengo_ability,
                gholdengo_item,
                False,
            )
        )

    def test_gholdengo_vs_gliscor(self):
        """
        サーフゴー vs グライオン

        サーフゴーの技:
        - ゴールドラッシュ (鋼技) → 等倍
        - シャドーボール (ゴースト技) → 等倍

        結果: サーフゴーは攻撃可能
        """
        gholdengo_moves = ["ゴールドラッシュ", "シャドーボール"]
        gliscor_types = ["じめん", "ひこう"]

        results = self.calculator.check_all_moves_effectiveness(
            moves=gholdengo_moves,
            defender_types=gliscor_types,
            defender_ability=None,
            defender_item=None,
            gravity=False,
        )

        # 少なくとも1つは有効
        self.assertTrue(any(r.is_effective for r in results))


if __name__ == "__main__":
    unittest.main()
