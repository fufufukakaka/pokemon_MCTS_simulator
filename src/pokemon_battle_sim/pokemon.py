import json
import warnings
from datetime import datetime, timedelta, timezone

from src.pokemon_battle_sim.utils import to_hankaku


class Pokemon:
    """
    ポケモンの個体を表現するクラス。ポケモンのデータ全般をクラス変数に持つ。

    クラス変数 (抜粋)
    ----------------------------------------
    Pokemon.zukan: dict
        key: ポケモン名。
        value: タイプ、特性、種族値、ゲーム上の表示名、体重。
        (例)
        Pokemon.zukan['オーガポン(かまど)'] = {
            'type': ['くさ', 'ほのお'],
            'ability': ['かたやぶり'],
            'base': [80, 120, 84, 60, 96, 110],
            'display_name': 'オーガポン',
            'weight': 39.8
        }

    Pokemon.zukan_name: dict
        key: ゲーム上の表示名。
        value: ポケモン名のリスト。
        (例)
        Pokemon.zukan_name['ウーラオス'] = ['ウーラオス(れんげき)', 'ウーラオス(いちげき)']

    Pokemon.home: dict
        key: ポケモン名。
        value: ランクマッチの使用率データ。
        (例)
        Pokemon.home['カイリュー'] = {
            'move': [['しんそく', 'じしん', 'りゅうのまい', 'はねやすめ', 'げきりん', 'スケイルショット', 'アンコール', 'アイアンヘッド', 'けたぐり', 'でんじは'],
                    [78.7, 78.0, 42.5, 39.4, 30.8, 24.5, 23.3, 16.2, 10.0, 9.9]],
            'ability': [['マルチスケイル', 'せいしんりょく'],
                    [99.8, 0.2]],
            'item': [['こだわりハチマキ', 'いかさまダイス', 'ゴツゴツメット', 'たべのこし', 'あつぞこブーツ', 'じゃくてんほけん', 'とつげきチョッキ', 'ラムのみ', 'シルクのスカーフ', 'おんみつマント'],
                    [33.9, 21.5, 18.3, 8.0, 4.3, 3.4, 2.2, 1.9, 1.7, 1.6]],
            'Ttype': [['ノーマル', 'じめん', 'はがね', 'ひこう', 'フェアリー', 'ほのお', 'でんき', 'みず', 'どく', 'ドラゴン'],
                    [65.0, 11.3, 11.0, 4.7, 4.5, 1.2, 0.7, 0.6, 0.4, 0.2]]
        }

    Pokemon.nature_corrections: dict
        key: 性格。
        value: 性格補正値リスト。
        (例)
        Pokemon.nature_corrections['いじっぱり'] = [1.0, 1.1, 1.0, 0.9, 1.0, 1.0]

    Pokemon.type_id = {
        'ノーマル': 0, 'ほのお': 1, 'みず': 2, 'でんき': 3, 'くさ': 4, 'こおり': 5, 'かくとう': 6,
        'どく': 7, 'じめん': 8, 'ひこう': 9, 'エスパー': 10, 'むし': 11, 'いわ': 12, 'ゴースト': 13,
        'ドラゴン': 14, 'あく': 15, 'はがね': 16, 'フェアリー': 17, 'ステラ': 18
    }

    Pokemon.type_corrections: list
        (例) どくタイプの技でくさおタイプに攻撃したときのタイプ補正値。
        Pokemon.type_corrections[7][4] = 2.0

    Pokemon.abilities: list[str]
        全ての特性。

    Pokemon.items: dict
        key: アイテム名。
        value: なげつける威力。

    Pokemon.all_moves: dict
        key: わざ名。
        value: わざのタイプ、分類、威力、命中率、PP。
        (例)
        Pokemon.all_moves['しんそく'] = {
            'type': 'ノーマル',
            'class': 'phy',
            'power': 80,
            'hit': 100,
            'pp': 8
        }

    Pokemon.combo_hit: dict
        key: 連続技。
        value: [最小ヒット数, 最大ヒット数]。

    インスタンス変数 (抜粋)
    ----------------------------------------
    self.__name: str
        ポケモン名。

    self.__display_name: str
        ゲーム上の表示名。

    self.__types: list[str]
        タイプ。

    self.__weight: float
        体重。

    self.sex: int
        性別。Pokemon.MALE or Pokemon.FEMALE or Pokemon.NONSEXUAL。

    self.__level: int
        レベル。

    self.__nature: str
        性格。

    self.__org_ability: str
        もとの特性。

    self.ability: str
        現在の特性。試合中に技や特性により変更される可能性がある。

    self.item: str
        所持しているアイテム。

    self.lost_item: str
        失ったアイテム。

    self.Ttype: str
        テラスタイプ。

    self.terastal: bool
        テラスタルしていればTrue。

    self.__status: list[int]
        ステータス。[H,A,B,C,D,S]。

    self.__base: list[int]
        種族値。[H,A,B,C,D,S]。

    self.__indiv: list[int]
        個体値。[H,A,B,C,D,S]。

    self.__effort: list[int]
        努力値。[H,A,B,C,D,S]。

    self.__hp: int
        残りHP。

    self.__hp_ratio: float
        残りHP割合。

    self.sub_hp: int
        みがわりの残りHP。

    self.__moves: list[str]
        わざ。最大10個。

    self.last_pp_move: str
        最後にPPを消費した技。

    self.last_used_move: str
        最後に出た技。

    self.pp: list[int]
        わざの残りPP。

    self.rank: list[int]
        能力ランク。[H,A,B,C,D,S,命中,回避]。

    self.ailment: str
        状態異常。

    self.condition: dict
        状態変化。

    self.boost_index: int
        クォークチャージ、こだいかっせい、ブーストエナジーにより上昇した能力番号。
    """

    zukan = {}
    zukan_name = {}
    form_diff = {}  # {表示名: フォルム差 (='type' or 'ability')}
    japanese_display_name = {}  # {各言語の表示名: 日本語の表示名}
    foreign_display_names = {}  # {日本語の表示名: [全言語の表示名]}
    home = {}

    type_file_code = {}  # {テラスタイプ: 画像コード}
    template_file_code = {}  # {ポケモン名: テンプレート画像コード}

    nature_corrections = {}
    type_id = {}
    type_corrections = []

    abilities = []
    ability_category = {}  # {分類: [該当する特性]}

    items = {}
    item_buff_type = {}  # {タイプ強化アイテム: 強化タイプ}
    item_debuff_type = {}  # {タイプ半減きのみ: 半減タイプ}
    item_correction = {}  # {アイテム: 威力補正}
    consumable_items = []  # 消耗アイテム

    all_moves = {}
    move_category = {}  # {分類: [該当する技]}
    move_value = {}  # {分類: {技: 値}}
    move_priority = {}  # {技: 優先度}
    combo_hit = {}
    move_effect = {}  # {技: 追加効果dict}

    stone_weather = {
        "sunny": "あついいわ",
        "rainy": "しめったいわ",
        "snow": "つめたいいわ",
        "sandstorm": "さらさらいわ",
    }
    plate_type = {
        "まっさらプレート": "ノーマル",
        "ひのたまプレート": "ほのお",
        "しずくプレート": "みず",
        "みどりのプレート": "くさ",
        "いかずちプレート": "でんき",
        "つららのプレート": "こおり",
        "こぶしのプレート": "かくとう",
        "もうどくプレート": "どく",
        "だいちのプレート": "じめん",
        "あおぞらプレート": "ひこう",
        "ふしぎのプレート": "エスパー",
        "たまむしプレート": "むし",
        "がんせきプレート": "いわ",
        "もののけプレート": "ゴースト",
        "りゅうのプレート": "ドラゴン",
        "こわもてプレート": "あく",
        "こうてつプレート": "はがね",
        "せいれいプレート": "フェアリー",
    }

    ailments = ("PSN", "PAR", "BRN", "SLP", "FLZ")
    weathers = ("sunny", "rainy", "snow", "sandstorm")
    fields = ("elecfield", "glassfield", "psycofield", "mistfield")

    # 性別
    MALE = 1
    FEMALE = -1
    NONSEXUAL = 0

    status_label = ("H", "A", "B", "C", "D", "S", "命中", "回避")
    status_label_hiragana = [
        "HP",
        "こうげき",
        "ぼうぎょ",
        "とくこう",
        "とくぼう",
        "すばやさ",
        "めいちゅう",
        "かいひ",
    ]
    status_label_kanji = [
        "HP",
        "攻撃",
        "防御",
        "特攻",
        "特防",
        "素早さ",
        "命中",
        "回避",
    ]

    JPN = {
        "PSN": "どく",
        "PAR": "まひ",
        "BRN": "やけど",
        "SLP": "ねむり",
        "FLZ": "こおり",
        "confusion": "こんらん",
        "critical": "急所ランク",
        "aquaring": "アクアリング",
        "healblock": "かいふくふうじ",
        "magnetrise": "でんじふゆう",
        "noroi": "呪い",
        "horobi": "ほろびのうた",
        "yadorigi": "やどりぎのタネ",
        "ame_mamire": "あめまみれ",
        "encore": "アンコール",
        "anti_air": "うちおとす",
        "kanashibari": "かなしばり",
        "shiozuke": "しおづけ",
        "jigokuzuki": "じごくづき",
        "charge": "じゅうでん",
        "stock": "たくわえる",
        "chohatsu": "ちょうはつ",
        "change_block": "にげられない",
        "nemuke": "ねむけ",
        "neoharu": "ねをはる",
        "bind": "バインド",
        "meromero": "メロメロ",
        "badpoison": "もうどく",
        "sunny": "はれ",
        "rainy": "あめ",
        "snow": "ゆき",
        "sandstorm": "すなあらし",
        "elecfield": "エレキフィールド",
        "glassfield": "グラスフィールド",
        "psycofield": "サイコフィールド",
        "mistfield": "ミストフィールド",
        "gravity": "じゅうりょく",
        "trickroom": "トリックルーム",
        "reflector": "リフレクター",
        "lightwall": "ひかりのかべ",
        "oikaze": "おいかぜ",
        "safeguard": "しんぴのまもり",
        "whitemist": "しろいきり",
        "makibishi": "まきびし",
        "dokubishi": "どくびし",
        "stealthrock": "ステルスロック",
        "nebanet": "ねばねばネット",
        "wish": "ねがいごと",
    }

    def __init__(self, name: str = "ピカチュウ", use_template: bool = True):
        """{name}のポケモンを生成する。{use_template}=Trueならテンプレートを適用して初期化する"""
        self.sex = Pokemon.NONSEXUAL
        self.__level = 50
        self.__nature = "まじめ"
        if name in Pokemon.home:
            self.org_ability = Pokemon.home[name]["ability"][0][0]
        else:
            self.org_ability = Pokemon.zukan[name]["ability"][0]
        self.item = ""
        self.lost_item = ""

        self.__status = [0] * 6
        self.__indiv = [31] * 6
        self.__effort = [0] * 6
        self.__hp = 0
        self.__hp_ratio = 1

        self.name = name
        self.Ttype = self.__types[0]

        self.__moves = []
        self.reset_game()

        if use_template:
            self.apply_template()

    def reset_game(self):
        """ポケモンを試合開始前の状態に初期化する"""
        self.come_back()
        self.ailment = ""
        self.terastal = False
        self.__hp = self.__status[0]
        self.__hp_ratio = 1
        self.pp = [Pokemon.all_moves[m]["pp"] if m else 0 for m in self.__moves]
        self.sleep_count = 0  # ねむり状態の残りターン

        # ばけのかわリセット
        if "ばけのかわ" in self.ability:
            self.ability = self.__org_ability

        # おもかげやどし解除
        if "オーガポン" in self.name:
            self.org_ability = Pokemon.zukan[self.name]["ability"][0]

        # フォルムリセット
        if self.name == "イルカマン(マイティ)":
            self.name == "イルカマン(ナイーブ)"
            self.update_status()

        if self.name == "テラパゴス(ステラ)":
            self.name = "テラパゴス(テラスタル)"
            self.org_ability = "テラスシェル"
            self.update_status()

    def come_back(self):
        """ポケモンを控えに戻したときの状態に初期化する"""
        self.rank = [0] * 8
        self.last_pp_move = ""
        self.last_used_move = ""
        self.inaccessible = 0
        self.lockon = False
        self.lost_types = []
        self.added_types = []
        self.sub_hp = 0
        self.boost_index = 0
        self.acted_turn = 0  # 行動したターン数
        self.n_attacked = 0  # 被弾回数
        self.fixed_move = ""  # こだわっている技
        self.hide_move = ""  # 隠れている技(を使っている状態)
        self.BE_activated = False  # ブーストエナジーが発動していればTrue
        self.rank_dropped = False  # ランク下降していればTrue
        self.berserk_triggered = False  # ぎゃくじょうの発動条件を満たしていればTrue

        self.condition = {
            "confusion": 0,  # こんらん 残りターン
            "critical": 0,  # 急所ランク上昇
            "aquaring": 0,  # アクアリング
            "healblock": 0,  # かいふくふうじ 残りターン
            "magnetrise": 0,  # でんじふゆう 残りターン
            "noroi": 0,  # のろい
            "horobi": 0,  # ほろびのうたカウント
            "yadorigi": 0,  # やどりぎのタネ
            # 以上がバトンタッチ対象
            "ame_mamire": 0,  # あめまみれ 残りターン
            "encore": 0,  # アンコール 残りターン
            "anti_air": 0,  # うちおとす
            "kanashibari": 0,  # かなしばり 残りターン
            "shiozuke": 0,  # しおづけ
            "jigokuzuki": 0,  # じごくづき 残りターン
            "charge": 0,  # じゅうでん
            "stock": 0,  # たくわえるカウント
            "chohatsu": 0,  # ちょうはつ 残りターン
            "change_block": 0,  # にげられない
            "nemuke": 0,  # ねむけ 残りターン
            "neoharu": 0,  # ねをはる
            "michizure": 0,  # みちづれ
            "meromero": 0,  # メロメロ
            "badpoison": 0,  # もうどくカウント
            "bind": 0,  # バインド (残りターン)+0.1*(ダメージ割合)
        }

        # 特性の処理
        if self.ability == "さいせいりょく" and self.condition["healblock"] == 0:
            self.hp = min(self.__status[0], self.hp + int(self.__status[0] / 3))
        elif self.ability == "しぜんかいふく":
            self.ailment = ""

        if "ばけのかわ" not in self.ability:
            self.ability = self.__org_ability

    def update_status(self, keep_damage=False):
        """ステータスを更新する。
        {keep_damage}=Trueならステータス更新前に受けていたダメージを更新後にも適用し、FalseならHPを全回復する。
        """
        nc = Pokemon.nature_corrections[self.__nature]
        damage = self.__status[0] - self.__hp

        self.__status[0] = (
            int(
                (self.base[0] * 2 + self.__indiv[0] + int(self.__effort[0] / 4))
                * self.__level
                / 100
            )
            + self.__level
            + 10
        )
        for i in range(1, 6):
            self.__status[i] = int(
                (
                    int(
                        (self.base[i] * 2 + self.__indiv[i] + int(self.__effort[i] / 4))
                        * self.__level
                        / 100
                    )
                    + 5
                )
                * nc[i]
            )

        self.hp = int(self.__status[0] * self.__hp_ratio)
        if keep_damage:
            self.hp = self.hp - damage

    def apply_template(self):
        """ポケモンの型を設定する"""
        if self.__name in Pokemon.home:
            self.__nature = Pokemon.home[self.__name]["nature"][0][0]
            self.org_ability = Pokemon.home[self.__name]["ability"][0][0]
            self.Ttype = Pokemon.home[self.__name]["Ttype"][0][0]
            self.moves = Pokemon.home[self.__name]["move"][0][:4]

    # Getter
    @property
    def name(self):
        return self.__name

    @property
    def display_name(self):
        return self.__display_name

    @property
    def level(self):
        return self.__level

    @property
    def weight(self):
        w = self.__weight
        match self.ability:
            case "ライトメタル":
                w = int(w * 0.5 * 10) / 10
            case "ヘヴィメタル":
                w *= 2
        if self.item == "かるいし":
            w = int(w * 0.5 * 10) / 10
        return w

    @property
    def nature(self):
        return self.__nature

    @property
    def types(self):
        result = self.__types.copy()
        if self.terastal:
            if self.Ttype != "ステラ":
                result = [self.Ttype]
        else:
            if self.__name == "アルセウス":
                result = [
                    (
                        Pokemon.plate_type[self.item]
                        if self.item in Pokemon.plate_type
                        else "ノーマル"
                    )
                ]
            else:
                result = [t for t in result if t not in self.lost_types]
                result += self.added_types
        return result

    @property
    def org_types(self):
        return self.__types.copy()

    @property
    def org_ability(self):
        return self.__org_ability

    @property
    def status(self):
        return self.__status.copy()

    @property
    def base(self):
        return self.__base.copy()

    @property
    def indiv(self):
        return self.__indiv.copy()

    @property
    def effort(self):
        return self.__effort.copy()

    @property
    def moves(self):
        return self.__moves.copy()

    @property
    def hp(self):
        return self.__hp

    @property
    def hp_ratio(self):
        return self.__hp_ratio

    # Setter
    @name.setter
    def name(self, name: str):
        if name not in Pokemon.zukan:
            warnings.warn(f"{name} is not in Pokemon.zukan")
        else:
            self.__name = name
            self.__display_name = Pokemon.zukan[self.__name]["display_name"]
            self.__types = Pokemon.zukan[self.__name]["type"].copy()
            self.__base = Pokemon.zukan[self.__name]["base"].copy()
            self.__weight = Pokemon.zukan[self.__name]["weight"]
            self.update_status()

    def change_form(self, name: str):
        if name not in Pokemon.zukan:
            warnings.warn(f"{name} is not in Pokemon.zukan")
        else:
            self.__name = name
            self.__display_name = Pokemon.zukan[self.__name]["display_name"]
            self.__types = Pokemon.zukan[self.__name]["type"].copy()
            self.__base = Pokemon.zukan[self.__name]["base"].copy()
            self.__weight = Pokemon.zukan[self.__name]["weight"]
            self.update_status(keep_damage=True)

            if (
                self.__name == "ザシアン(けんのおう)"
                and "アイアンヘッド" in self.__moves
            ):
                ind = self.__moves.index("アイアンヘッド")
                self.set_move(ind, "きょじゅうざん")

            if (
                self.__name == "ザマゼンタ(たてのおう)"
                and "アイアンヘッド" in self.__moves
            ):
                ind = self.__moves.index("アイアンヘッド")
                self.set_move(ind, "きょじゅうだん")

    @level.setter
    def level(self, level: int):
        self.__level = level
        self.update_status()

    @nature.setter
    def nature(self, nature: str):
        self.__nature = nature
        self.update_status()

    @org_ability.setter
    def org_ability(self, ability: str):
        self.__org_ability = self.ability = ability

    @status.setter
    def status(self, status: list[int]):
        nc = Pokemon.nature_corrections[self.__nature]
        for i in range(6):
            for eff in range(0, 256, 4):
                if i == 0:
                    v = (
                        int(
                            (self.__base[0] * 2 + self.__indiv[0] + int(eff / 4))
                            * self.__level
                            / 100
                        )
                        + self.__level
                        + 10
                    )
                else:
                    v = int(
                        (
                            int(
                                (self.__base[i] * 2 + self.__indiv[i] + int(eff / 4))
                                * self.__level
                                / 100
                            )
                            + 5
                        )
                        * nc[i]
                    )
                if v == status[i]:
                    self.__effort[i] = eff
                    self.__status[i] = v
                    break

    def set_status(self, index: int, value: int) -> bool:
        nc = Pokemon.nature_corrections[self.__nature]
        for eff in range(0, 256, 4):
            if index == 0:
                v = (
                    int(
                        (self.__base[0] * 2 + self.__indiv[0] + int(eff / 4))
                        * self.__level
                        / 100
                    )
                    + self.__level
                    + 10
                )
            else:
                v = int(
                    (
                        int(
                            (
                                self.__base[index] * 2
                                + self.__indiv[index]
                                + int(eff / 4)
                            )
                            * self.__level
                            / 100
                        )
                        + 5
                    )
                    * nc[index]
                )
            if v == value:
                self.__effort[index] = eff
                self.__status[index] = v
                return True
        return False

    @indiv.setter
    def indiv(self, indiv: list[int]):
        self.__indiv = indiv
        self.update_status()

    @effort.setter
    def effort(self, effort: list[int]):
        self.__effort = effort
        self.update_status()

    def set_effort(self, index: int, value: list[int]):
        self.__effort[index] = value
        self.update_status()

    @moves.setter
    def moves(self, moves: list[str]):
        self.__moves, self.pp = [], []
        for move in moves:
            if not move or move in self.__moves:
                continue
            elif move not in Pokemon.all_moves:
                warnings.warn(f"{move} is not in Pokemon.all_moves")
            else:
                self.__moves.append(move)
                self.pp.append(Pokemon.all_moves[move]["pp"])

            if len(self.__moves) == 10:
                break

    def set_move(self, index: int, move: str):
        """技を追加してPPを初期化する"""
        if index not in range(10) or not move or move in self.__moves:
            return
        elif move not in Pokemon.all_moves:
            warnings.warn(f"{move} is not in Pokemon.all_moves")
        else:
            self.__moves[index] = move
            self.pp[index] = Pokemon.all_moves[move]["pp"]

    def add_move(self, move: str):
        """技を追加してPPを初期化する"""
        if not move or move in self.__moves or len(self.__moves) == 10:
            return
        elif move not in Pokemon.all_moves:
            warnings.warn(f"{move} is not in Pokemon.all_moves")
        else:
            self.__moves.append(move)
            self.pp.append(Pokemon.all_moves[move]["pp"])

    @hp.setter
    def hp(self, hp: int):
        self.__hp = hp
        self.__hp_ratio = self.__hp / self.__status[0]

    @hp_ratio.setter
    def hp_ratio(self, hp_ratio: int):
        self.__hp_ratio = hp_ratio
        self.__hp = int(hp_ratio * self.__status[0])
        if hp_ratio and self.__hp == 0:
            self.__hp = 1

    def use_terastal(self) -> bool:
        if self.terastal:
            return False

        self.terastal = True

        if "オーガポン" in self.name:
            self.org_ability = "おもかげやどし"
        elif "テラパゴス" in self.name:
            self.change_form("テラパゴス(ステラ)")

        return True

    def has_protected_ability(self) -> bool:
        """特性が上書きされない状態ならTrueを返す"""
        return (
            self.ability in Pokemon.ability_category["protected"]
            or self.item == "とくせいガード"
        )

    def is_blowable(self) -> bool:
        """強制交代されうる状態ならTrueを返す"""
        return (
            self.ability not in ["きゅうばん", "ばんけん"]
            and not self.condition["neoharu"]
        )

    def contacts(self, move: str) -> bool:
        """{move}を使用したときに直接攻撃ならTrueを返す"""
        return (
            move in Pokemon.move_category["contact"]
            and self.ability != "えんかく"
            and self.item != "ぼうごパッド"
            and not (
                move in Pokemon.move_category["punch"] and self.item == "パンチグローブ"
            )
        )

    def item_removable(self):
        """アイテムを奪われない状態ならTrueを返す"""
        if (
            self.ability == "ねんちゃく"
            or self.item == "ブーストエナジー"
            or "オーガポン(" in self.__name
            or ("ザシアン" in self.__name and self.item == "くちたけん")
            or ("ザマゼンタ" in self.__name and self.item == "くちたたて")
            or ("ディアルガ" in self.__name and "こんごうだま" in self.item)
            or ("パルキア" in self.__name and "しらたま" in self.item)
            or ("ギラティナ" in self.__name and "はっきんだま" in self.item)
            or (self.__name == "アルセウス" and "プレート" in self.item)
            or (self.__name == "ゲノセクト" and "カセット" in self.item)
        ):
            return False
        return True

    def show(self):
        print(f"\tName      {self.__name}")
        print(f"\tNature    {self.__nature}")
        print(f"\tAbility   {self.ability}")
        print(f"\tItem      {self.item} ({self.lost_item})")
        print(f"\tTerastal  {self.Ttype} {self.terastal}")
        print(f"\tMoves     {self.__moves}")
        print(f"\tEffort    {self.__effort}")
        print(f"\tStatus    {self.__status}")
        print(f"\tHP        {self.hp}")
        print()

    def rank_correction(self, index: int) -> float:
        """ランク補正値を返す。
        Parameters
        ----------
        index: int
            0,1,2,3,4,5,6,7
            H,A,B,C,D,S,命中,回避
        """
        if self.rank[index] >= 0:
            return (self.rank[index] + 2) / 2
        else:
            return 2 / (2 - self.rank[index])

    def move_class(self, move: str) -> str:
        """{move}を使用したときの技の分類を返す"""
        if move in ["テラバースト", "テラクラスター"] and self.terastal:
            effA = self.__status[1] * self.rank_correction(1)
            effC = self.__status[3] * self.rank_correction(3)
            return "phy" if effA >= effC else "spe"
        return Pokemon.all_moves[move]["class"]

    def last_pp_move_index(self) -> int:
        """最後にPPを消費した技のindexを返す"""
        return (
            self.__moves.index(self.last_pp_move)
            if self.last_pp_move and self.last_pp_move in self.__moves
            else None
        )

    def energy_boost(self, boost: bool = True):
        """{boost}=Trueならブーストエナジーにより能力を上昇させ、Falseなら元に戻す"""
        if boost:
            ls = [v * self.rank_correction(i) for i, v in enumerate(self.__status[1:])]
            self.boost_index = ls.index(max(ls)) + 1
        else:
            self.boost_index = 0

    def fruit_recovery(self, hp_dict: dict) -> dict:
        """きのみによる回復後のHP dictを返す。リーサル計算用の関数"""
        result = {}
        for hp in hp_dict:
            if hp == "0" or hp[-2:] == ".0":
                push(result, hp, hp_dict[hp])
            elif self.item in ["オレンのみ", "オボンのみ"]:
                if float(hp) <= 0.5 * self.__status[0]:
                    recovery = (
                        int(self.__status[0] / 4) if self.item == "オボンのみ" else 10
                    )
                    key = str(min(self.hp, int(float(hp)) + recovery)) + ".0"
                    push(result, key, hp_dict[hp])
                else:
                    push(result, hp, hp_dict[hp])
            elif self.item in [
                "フィラのみ",
                "ウイのみ",
                "マゴのみ",
                "バンジのみ",
                "イアのみ",
            ]:
                if float(hp) / self.__status[0] <= (
                    0.5 if self.ability == "くいしんぼう" else 0.25
                ):
                    key = str(int(float(hp)) + int(self.__status[0] / 3)) + ".0"
                    push(result, key, hp_dict[hp])
                else:
                    push(result, hp, hp_dict[hp])
        return result

    def damage_text(self, damage: dict, lethal_num: int, lethal_prob: float) -> str:
        """リーサル計算結果から 'd1~d2 (p1~p2 %) 確n' 形式の文字列を生成する"""
        damages = [int(k) for k in list(damage.keys())]
        min_damage, max_damage = min(damages), max(damages)

        result = f"{min_damage}~{max_damage} ({100*min_damage/self.__status[0]:.1f}~{100*max_damage/self.__status[0]:.1f}%)"
        if lethal_prob == 1:
            result += f" 確{lethal_num}"
        elif lethal_prob > 0:
            result += f" 乱{lethal_num}({100*lethal_prob:.2f}%)"
        return result

    def find(pokemon_list, name: str = "", display_name: str = ""):
        """{pokemon_list}から条件に合致したPokemonインスタンスを返す"""
        for p in pokemon_list:
            if name == p.name or display_name == p.display_name:
                return p

    def index(pokemon_list, name: str = "", display_name: str = ""):
        """{pokemon_list}から条件に合致したPokemonインスタンスのindexを返す"""
        for i, p in enumerate(pokemon_list):
            if name == p.name or display_name == p.display_name:
                return i

    def rank2str(rank_list: list[int]):
        """能力ランクから 'A+1 S+1' 形式の文字列を返す"""
        s = ""
        for i, v in enumerate(rank_list):
            if rank_list[i]:
                s += f" {Pokemon.status_label[i]}{'+'*(v > 0)}{v}"
        return s[1:]

    def calculate_status(
        name: str, nature: str, efforts: list[int], indivs: list[int] = [31] * 6
    ) -> list[int]:
        p = Pokemon(name)
        p.nature = nature
        p.indiv = indivs
        p.effort = efforts
        p.update_status()
        return p.status

    def init(season=None):
        """ライブラリを初期化する"""

        # シーズンが指定されていなければ、最新のシーズンを取得する
        if season is None:
            dt_now = datetime.now(timezone(timedelta(hours=+9), "JST"))
            y, m, d = dt_now.year, dt_now.month, dt_now.day
            season = max(12 * (y - 2022) + m - 11 - (d == 1), 1)

        # タイプ画像コードの読み込み
        with open("data/terastal/codelist.txt", encoding="utf-8") as fin:
            for line in fin:
                data = line.split()
                Pokemon.type_file_code[data[1]] = data[0]
            # print(Pokemon.type_file_code)

        # テンプレート画像コードの読み込み
        with open("data/template/codelist.txt", encoding="utf-8") as fin:
            for line in fin:
                data = line.split()
                Pokemon.template_file_code[data[0]] = data[1]
            # print(Pokemon.template_file_code)

        # 図鑑の読み込み
        with open("data/zukan.txt", encoding="utf-8") as fin:
            next(fin)
            for line in fin:
                data = line.split()
                name = to_hankaku(data[1])
                Pokemon.zukan[name] = {}
                Pokemon.zukan[name]["type"] = [s for s in data[2:4] if s != "-"]
                Pokemon.zukan[name]["ability"] = [s for s in data[4:8] if s != "-"]
                Pokemon.zukan[name]["base"] = list(map(int, data[8:14]))

                for s in Pokemon.zukan[name]["ability"]:
                    Pokemon.abilities.append(s)

                # 表示名の設定
                display_name = name
                if "ロトム" in name:
                    display_name = "ロトム"
                else:
                    if "(" in name:
                        display_name = name[: display_name.find("(")]
                    display_name = display_name.replace("パルデア", "")
                    display_name = display_name.replace("ヒスイ", "")
                    display_name = display_name.replace("ガラル", "")
                    display_name = display_name.replace("アローラ", "")
                    display_name = display_name.replace("ホワイト", "")
                    display_name = display_name.replace("ブラック", "")
                Pokemon.zukan[name]["display_name"] = display_name

                if display_name not in Pokemon.zukan_name:
                    Pokemon.zukan_name[display_name] = [name]
                elif name not in Pokemon.zukan_name[display_name]:
                    Pokemon.zukan_name[display_name].append(name)
                    # フォルム違いの差分を記録
                    for key in ["type", "ability"]:
                        if (
                            Pokemon.zukan[Pokemon.zukan_name[display_name][0]][key]
                            != Pokemon.zukan[name][key]
                        ):
                            Pokemon.form_diff[display_name] = key
                            break

            Pokemon.abilities = list(set(Pokemon.abilities))
            Pokemon.abilities.sort()

            # print(Pokemon.zukan)
            # print(Pokemon.abilities)
            # print(Pokemon.zukan_name)
            # print(Pokemon.form_diff)

        # 外国語名の読み込み
        with open("data/foreign_name.txt", encoding="utf-8") as fin:
            next(fin)
            for line in fin:
                data = list(map(to_hankaku, line.split()))
                for i in range(len(data)):
                    Pokemon.japanese_display_name[to_hankaku(data[i])] = to_hankaku(
                        data[0]
                    )
                Pokemon.foreign_display_names[to_hankaku(data[0])] = [
                    to_hankaku(s) for s in data
                ]
            # print(Pokemon.japanese_display_name)
            # print(Pokemon.foreign_display_names)

        # 体重の読み込み
        with open("data/weight.txt", encoding="utf-8") as fin:
            next(fin)
            for line in fin:
                data = line.split()
                Pokemon.zukan[to_hankaku(data[0])]["weight"] = float(data[1])

        # 特性の読み込み
        with open("data/ability_category.txt", encoding="utf-8") as fin:
            for line in fin:
                data = list(map(to_hankaku, line.split()))
                Pokemon.ability_category[data[0]] = data[1:]
                if "ばけのかわ" in Pokemon.ability_category[data[0]]:
                    Pokemon.ability_category[data[0]].append("ばけのかわ+")
                # print(data[0]), print(Pokemon.ability_category[data[0]])

        # アイテムの読み込み
        with open("data/item.txt", encoding="utf-8") as fin:
            next(fin)
            for line in fin:
                data = line.split()
                item = to_hankaku(data[0])
                Pokemon.items[item] = {"power": int(data[1])}  # なげつける威力
                if data[2] != "-":
                    Pokemon.item_buff_type[item] = data[2]
                if data[3] != "-":
                    Pokemon.item_debuff_type[item] = data[3]
                Pokemon.item_correction[item] = float(data[4])
                if int(data[5]):
                    Pokemon.consumable_items.append(item)

            Pokemon.item_correction[""] = 1
            # print(Pokemon.items)
            # print(Pokemon.item_correction)

        # 技の分類の読み込み
        with open("data/move_category.txt", encoding="utf-8") as fin:
            for line in fin:
                data = list(map(to_hankaku, line.split()))
                Pokemon.move_category[data[0]] = data[1:]
                # print(data[0]), print(Pokemon.move_category[data[0]])

        with open("data/move_value.txt", encoding="utf-8") as fin:
            for line in fin:
                data = line.split()
                Pokemon.move_value[data[0]] = {}
                for i in range(int(len(data[1:]) / 2)):
                    Pokemon.move_value[data[0]][to_hankaku(data[2 * i + 1])] = float(
                        data[2 * i + 2]
                    )
                # print(data[0], Pokemon.move_value[data[0]])

        # 技の読み込み
        with open("data/move.txt", encoding="utf-8") as fin:
            eng = {"物理": "phy", "特殊": "spe"}
            next(fin)
            for line in fin:
                data = line.split()
                move = to_hankaku(data[0])
                if "変化" in data[2]:
                    data[2] = "sta" + format(int(data[2][2:]), "04b")
                else:
                    data[2] = eng[data[2]]
                Pokemon.all_moves[move] = {
                    "type": data[1],  # タイプ
                    "class": data[2],  # 分類
                    "power": int(data[3]),  # 威力
                    "hit": int(data[4]),  # 命中率
                    "pp": int(int(data[5]) * 1.6),  # PP
                }

            # 威力変動技を初期化する
            for move in Pokemon.move_category["power_var"]:
                Pokemon.all_moves[move]["power"] = 1

        # 技の優先度の読み込み
        with open("data/move_priority.txt", encoding="utf-8") as fin:
            for line in fin:
                data = line.split()
                for move in data[1:]:
                    Pokemon.move_priority[to_hankaku(move)] = int(data[0])
            # print(Pokemon.move_priority)

        # 技の追加効果の読み込み
        with open("data/move_effect.txt", encoding="utf-8") as fin:
            next(fin)
            for line in fin:
                data = line.split()
                move = to_hankaku(data[0])
                Pokemon.move_effect[move] = {}
                Pokemon.move_effect[move]["object"] = int(data[1])
                Pokemon.move_effect[move]["prob"] = float(data[2])
                Pokemon.move_effect[move]["rank"] = [0] + list(map(int, data[3:10]))
                Pokemon.move_effect[move]["ailment"] = list(map(int, data[10:15]))
                Pokemon.move_effect[move]["confusion"] = int(data[15])
                Pokemon.move_effect[move]["flinch"] = float(data[16])
            # print(Pokemon.move_effect)

        # 連続技の読み込み
        with open("data/combo_move.txt", encoding="utf-8") as fin:
            for line in fin:
                data = line.split()
                Pokemon.combo_hit[to_hankaku(data[0])] = [int(data[1]), int(data[2])]
            # print(Pokemon.combo_hit)

        # 性格補正の読み込み
        with open("data/nature.txt", encoding="utf-8") as fin:
            for line in fin:
                data = line.split()
                Pokemon.nature_corrections[data[0]] = list(map(float, data[1:7]))
            # print(Pokemon.nature_corrections)

        # タイプ相性補正の読み込み
        with open("data/type.txt", encoding="utf-8") as fin:
            line = fin.readline()
            data = line.split()
            for i in range(len(data)):
                Pokemon.type_id[data[i]] = i
            for line in fin:
                data = line.split()
                Pokemon.type_corrections.append(list(map(float, data)))
            # print(Pokemon.type_id)
            # print(Pokemon.type_corrections)

        # ランクマッチの統計データの読み込み
        filename = "data/battle_data/season22.json"
        print(f"{filename}")
        with open(filename, encoding="utf-8") as fin:
            dict = json.load(fin)
            for org_name in dict:
                name = to_hankaku(org_name)
                Pokemon.home[name] = {}
                Pokemon.home[name]["nature"] = dict[org_name]["nature"]
                Pokemon.home[name]["ability"] = dict[org_name]["ability"]
                Pokemon.home[name]["item"] = dict[org_name]["item"]
                Pokemon.home[name]["Ttype"] = dict[org_name]["Ttype"]
                Pokemon.home[name]["move"] = dict[org_name]["move"]

                # 半角表記に統一する
                for key in ["ability", "item", "move"]:
                    for i, s in enumerate(Pokemon.home[name][key][0]):
                        Pokemon.home[name][key][0][i] = to_hankaku(s)

                # データの補完
                if not Pokemon.home[name]["nature"][0]:
                    Pokemon.home[name]["nature"] = [["まじめ"], [100]]
                if not Pokemon.home[name]["ability"][0]:
                    Pokemon.home[name]["ability"] = [
                        [Pokemon.zukan[name]["ability"][0]],
                        [100],
                    ]
                if not Pokemon.home[name]["item"][0]:
                    Pokemon.home[name]["item"] = [[""], [100]]
                if not Pokemon.home[name]["Ttype"][0]:
                    Pokemon.home[name]["Ttype"] = [
                        [Pokemon.zukan[name]["type"][0]],
                        [100],
                    ]

            # print(Pokemon.home.keys())
