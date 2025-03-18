import datetime
import json
import random

from src.mcts.mcts_battle import MyMCTSBattle
from src.pokemon_battle_sim.pokemon import Pokemon


# Trainer クラスの定義
class Trainer:
    def __init__(self, name, rank, rating, pokemons, raw_pokemons):
        self.name = name
        self.rank = rank
        self.rating = rating
        self.sim_rating = 1500
        self.pokemons = pokemons  # 6体のポケモンリスト
        self.raw_pokemons = raw_pokemons  # インスタンス化する前のポケモンリスト

    def choose_team(self, team_size=3):
        """6体からランダムに3体を選出"""
        choiced_pokemons = random.sample(self.pokemons, team_size)
        if len(choiced_pokemons) < team_size:
            raise ValueError(f"選出ポケモンの数が足りません: {len(choiced_pokemons)}")
        return choiced_pokemons


# JSON ファイルからトレーナーデータを読み込む関数
def load_trainers_from_json(filename):
    trainers = []
    with open(filename, "r", encoding="utf-8") as f:
        data = json.load(f)
        for td in data:
            raw_pokemons = []
            for p in td["pokemons"]:
                raw_pokemons.append({
                    "name": p.get("name"),
                    "item": p.get("item"),
                    "nature": p.get("nature"),
                    "ability": p.get("ability"),
                    "Ttype": p.get("Ttype"),
                    "moves": p.get("moves"),
                    "effort": p.get("effort"),
                })
            trainer = Trainer(
                name=td.get("name"),
                rank=td.get("rank"),
                rating=td.get("rating"),
                pokemons=[],
                raw_pokemons=raw_pokemons,
            )
            trainers.append(trainer)
    return trainers


class SimulatedBattle:
    def __init__(self, trainer_a: Trainer, trainer_b: Trainer):
        self.trainer_a = trainer_a
        self.trainer_b = trainer_b
        self.turn = 0
        self.log = []

    def init_trainer_pokemons(self):
        Pokemon.init()

        # trainer_a と trainer_b のポケモンをインスタンス化
        for p in self.trainer_a.raw_pokemons:
            pokemon = Pokemon(p.get("name"))
            pokemon.item = p.get("item")
            pokemon.nature = p.get("nature")
            pokemon.ability = p.get("ability")
            pokemon.Ttype = p.get("Ttype")
            pokemon.moves = p.get("moves")
            pokemon.effort = p.get("effort")
            self.trainer_a.pokemons.append(pokemon)

        for p in self.trainer_b.raw_pokemons:
            pokemon = Pokemon(p.get("name"))
            pokemon.item = p.get("item")
            pokemon.nature = p.get("nature")
            pokemon.ability = p.get("ability")
            pokemon.Ttype = p.get("Ttype")
            pokemon.moves = p.get("moves")
            pokemon.effort = p.get("effort")
            self.trainer_b.pokemons.append(pokemon)

    def simulate_battle(self):
        self.init_trainer_pokemons()

        # 手持ちポケモンから3体選ぶ
        team_a = self.trainer_a.choose_team()
        team_b = self.trainer_b.choose_team()

        battle = MyMCTSBattle()
        battle.reset_game()

        battle.selected[0].append(team_a[0])
        battle.selected[0].append(team_a[1])
        battle.selected[0].append(team_a[2])

        battle.selected[1].append(team_b[0])
        battle.selected[1].append(team_b[1])
        battle.selected[1].append(team_b[2])

        while battle.winner() is None:
            battle.proceed()

            self.log.append(f"ターン {battle.turn}")
            print(f"ターン {battle.turn}")
            for pl in [0, 1]:
                self.log.append(f"Player {pl}: {battle.log[pl]}")
                print(f"Player {pl}: {battle.log[pl]}")
                print(f"Player {pl} pokemon HP list: {[v.hp_ratio for v in battle.selected[pl]]}")
                print(f"Player {pl} field pokemon rank: {battle.pokemon[pl].rank}")
            print(battle.damage_log)

        self.log.append(f"勝者: Player {battle.winner()}")

        return self.trainer_a if battle.winner() == 0 else self.trainer_b

    def save_log(self):
        filename = (
            f"logs/battle_log_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        )
        with open(filename, "w", encoding="utf-8") as f:
            for entry in self.log:
                f.write(entry + "\n")
        print(f"ログを {filename} に保存しました。")


def match_trainers(trainers, threshold=50):
    """
    trainers は Trainer クラスのインスタンスリスト（各インスタンスは sim_rating を rating 属性に持つ）
    threshold は対戦可能とみなすレーティング差の上限
    """
    if len(trainers) < 2:
        return None  # 対戦できるトレーナーがいない場合

    # 1人のトレーナーをランダムに選ぶ
    trainer_a = random.choice(trainers)

    # trainer_a と rating 差が threshold 以内の候補リストを作成
    candidates = [
        t
        for t in trainers
        if t != trainer_a and abs(t.sim_rating - trainer_a.sim_rating) <= threshold
    ]

    if candidates:
        # 候補があれば、その中からランダムに対戦相手を選ぶ
        trainer_b = random.choice(candidates)
    else:
        # 候補がいなければ、rating 差が最も小さいトレーナーを選ぶ
        trainer_b = min(
            [t for t in trainers if t != trainer_a],
            key=lambda t: abs(t.sim_rating - trainer_a.sim_rating),
        )

    return trainer_a, trainer_b


def update_elo(rating_a, rating_b, result_a, K=32):
    expected_a = 1 / (1 + 10 ** ((rating_b - rating_a) / 400))
    new_rating_a = rating_a + K * (result_a - expected_a)
    return new_rating_a


def main():
    # JSON ファイルからトレーナーデータを読み込む
    trainers = load_trainers_from_json("data/top_rankers/season_27.json")

    # マッチングして対戦開始。全部で 100 回の対戦を行う
    for _ in range(100):
        trainer_a, trainer_b = match_trainers(trainers)
        print(f"{trainer_a.name} vs {trainer_b.name} の対戦開始!")

        battle = SimulatedBattle(trainer_a, trainer_b)
        winner = battle.simulate_battle()
        print(f"勝者は {winner.name} です。")
        # battle.save_log()

        # 対戦結果に応じた Elo レーティングの更新
        if winner == trainer_a:
            trainer_a.sim_rating = update_elo(
                trainer_a.sim_rating, trainer_b.sim_rating, 1
            )
            trainer_b.sim_rating = update_elo(
                trainer_b.sim_rating, trainer_a.sim_rating, 0
            )
        else:
            trainer_a.sim_rating = update_elo(
                trainer_a.sim_rating, trainer_b.sim_rating, 0
            )
            trainer_b.sim_rating = update_elo(
                trainer_b.sim_rating, trainer_a.sim_rating, 1
            )

        print(
            f"更新後のレーティング: {trainer_a.name}: {trainer_a.sim_rating}, {trainer_b.name}: {trainer_b.sim_rating}"
        )

    # 全員の sim_rating と rating を表示
    for trainer in trainers:
        print(
            f"{trainer.name}: sim_rating={trainer.sim_rating}, rating={trainer.rating}"
        )


if __name__ == "__main__":
    main()
