import datetime
import json
import random

from src.mcts.mcts_battle import MyMCTSBattle
from src.pokemon_battle_sim.pokemon import Pokemon


# Pokemon クラスの定義
class BattlePokemon:
    def __init__(
        self,
        name,
        item=None,
        nature=None,
        ability=None,
        Ttype=None,
        moves=None,
        effort=None,
    ):
        self.name = name
        self.item = item
        self.nature = nature
        self.ability = ability
        self.Ttype = Ttype
        self.moves = moves or []
        self.effort = effort or []


# Trainer クラスの定義
class Trainer:
    def __init__(self, name, rank, rating, pokemons):
        self.name = name
        self.rank = rank
        self.rating = rating
        self.sim_rating = 1500
        self.pokemons = pokemons  # 6体のポケモンリスト

    def choose_team(self, team_size=3):
        """6体からランダムに3体を選出"""
        return random.sample(self.pokemons, team_size)


# JSON ファイルからトレーナーデータを読み込む関数
def load_trainers_from_json(filename):
    trainers = []
    with open(filename, "r", encoding="utf-8") as f:
        data = json.load(f)
        for td in data:
            # 各ポケモンのデータを Pokemon インスタンスに変換
            pokemons = []
            for p in td["pokemons"]:
                pokemon = Pokemon(p.get("name"))
                pokemon.item = p.get("item")
                pokemon.nature = p.get("nature")
                pokemon.ability = p.get("ability")
                pokemon.Ttype = p.get("Ttype")
                pokemon.moves = p.get("moves")
                pokemon.effort = p.get("effort")
                pokemons.append(pokemon)
            # トレーナーのデータから Trainer インスタンスを作成
            trainer = Trainer(
                name=td.get("name"),
                rank=td.get("rank"),
                rating=td.get("rating"),
                pokemons=pokemons,
            )
            trainers.append(trainer)
    return trainers


class Battle:
    def __init__(self, trainer_a: Trainer, trainer_b: Trainer):
        self.trainer_a = trainer_a
        self.trainer_b = trainer_b
        self.team_a = trainer_a.choose_team()
        self.team_b = trainer_b.choose_team()
        self.turn = 0
        self.log = []

    def simulate_turn(self):
        # 実際の対戦ロジックに合わせ、各ターンのアクションをシミュレート
        self.turn += 1
        action = f"ターン {self.turn}: {random.choice(['攻撃', '防御', '回避'])}"
        self.log.append(action)
        print(action)

    def simulate_battle(self):
        Pokemon.init()

        # 手持ちポケモンから3体選ぶ
        team_a = self.trainer_a.choose_team()
        team_b = self.trainer_b.choose_team()

        # ポケモンクラスにする
        team_a = [BattlePokemon(**vars(pokemon)) for pokemon in team_a]
        team_b = [BattlePokemon(**vars(pokemon)) for pokemon in team_b]

        battle = MyMCTSBattle()
        battle.selected[0] = team_a
        battle.selected[1] = team_b

        while battle.winner() is None:
            battle.proceed()
            print(f"ターン {battle.turn}")
            for pl in [0, 1]:
                print(f"Player {pl}: {battle.log[pl]}")

        print(f"勝者: Player {battle.winner()}")
        return self.trainer_a if battle.winner() == 0 else self.trainer_b

    def save_log(self):
        filename = f"battle_log_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(filename, "w", encoding="utf-8") as f:
            for entry in self.log:
                f.write(entry + "\n")
        print(f"ログを {filename} に保存しました。")


def match_trainers(trainers):
    # trainers は Trainer クラスのインスタンスリスト
    return random.sample(trainers, 2)


def update_elo(rating_a, rating_b, result_a, K=32):
    expected_a = 1 / (1 + 10 ** ((rating_b - rating_a) / 400))
    new_rating_a = rating_a + K * (result_a - expected_a)
    return new_rating_a


def main():
    # JSON ファイルからトレーナーデータを読み込む
    trainers = load_trainers_from_json("data/top_rankers/season_27.json")

    # マッチングして対戦開始
    trainer_a, trainer_b = match_trainers(trainers)
    print(f"{trainer_a.name} vs {trainer_b.name} の対戦開始!")

    battle = Battle(trainer_a, trainer_b)
    winner = battle.simulate_battle()
    print(f"勝者は {winner.name} です。")
    battle.save_log()

    # 対戦結果に応じた Elo レーティングの更新
    if winner == trainer_a:
        trainer_a.sim_rating = update_elo(trainer_a.sim_rating, trainer_b.sim_rating, 1)
        trainer_b.sim_rating = update_elo(trainer_b.sim_rating, trainer_a.sim_rating, 0)
    else:
        trainer_a.sim_rating = update_elo(trainer_a.sim_rating, trainer_b.sim_rating, 0)
        trainer_b.sim_rating = update_elo(trainer_b.sim_rating, trainer_a.sim_rating, 1)

    print(
        f"更新後のレーティング: {trainer_a.name}: {trainer_a.sim_rating}, {trainer_b.name}: {trainer_b.sim_rating}"
    )


if __name__ == "__main__":
    main()
