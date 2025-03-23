import datetime
import json
import random

import click

from src.database_handler import DatabaseHandler
from src.mcts.mcts_battle import MyMCTSBattle
from src.models import Trainer
from src.notify_discord import send_discord_notification
from src.pokemon_battle_sim.pokemon import Pokemon


# JSON ファイルからトレーナーデータを読み込む関数
def load_trainers_from_json(filename: str):
    trainers = []
    with open(filename, "r", encoding="utf-8") as f:
        data = json.load(f)
        for td in data:
            raw_pokemons = []
            for p in td["pokemons"]:
                raw_pokemons.append(
                    {
                        "name": p.get("name"),
                        "item": p.get("item"),
                        "nature": p.get("nature"),
                        "ability": p.get("ability"),
                        "Ttype": p.get("Ttype"),
                        "moves": p.get("moves"),
                        "effort": p.get("effort"),
                    }
                )
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

        self.log.append(
            f"Player 0: {self.trainer_a.name} vs Player 1: {self.trainer_b.name}"
        )
        self.log.append(f"Player 0 team: {[p.name for p in self.trainer_a.pokemons]}")
        self.log.append(f"Player 1 team: {[p.name for p in self.trainer_b.pokemons]}")

        while battle.winner() is None:
            battle.proceed()

            self.log.append(f"ターン {battle.turn}")
            print(f"ターン {battle.turn}")
            for pl in [0, 1]:
                self.log.append(f"Player {pl}: {battle.log[pl]}")
                print(f"Player {pl}: {battle.log[pl]}")
            for pl in [0, 1]:
                print(
                    f"Player {pl} pokemon HP list: {[v.hp_ratio for v in battle.selected[pl]]}"
                )
                print(f"Player {pl} field pokemon rank: {battle.pokemon[pl].rank}")

                # if (
                #     battle.turn == 0
                #     and sum([v.hp_ratio for v in battle.selected[pl]]) < 3
                # ):
                #     # 開始時に HP が満タンでないポケモンがいる場合の調査
                #     import os

                #     os.system(
                #         "osascript -e 'display notification \"ポケモンシミュレータ_HP総量がリセットされていないケースを発見しました\"'"
                #     )
                #     import pdb

                #     pdb.set_trace()

            print(battle.damage_log)

        winner = battle.winner()
        self.log.append(
            f"勝者: Player {winner}, {self.trainer_a.name if winner == 0 else self.trainer_b.name}"
        )

        # トレーナーのポケモンをすべてリセット
        self.trainer_a.pokemons = []
        self.trainer_b.pokemons = []

        return self.trainer_a if winner == 0 else self.trainer_b

    def save_log(self):
        saved_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"logs/battle_log_{saved_time}.txt"
        with open(filename, "w", encoding="utf-8") as f:
            for entry in self.log:
                f.write(entry + "\n")
        print(f"ログを {filename} に保存しました。")
        return saved_time


def match_trainers(trainers, trainer_a=None, threshold=50, random_battle=False):
    """
    trainers は Trainer クラスのインスタンスリスト（各インスタンスは sim_rating を rating 属性に持つ）
    threshold は対戦可能とみなすレーティング差の上限
    """
    if len(trainers) < 2:
        return None  # 対戦できるトレーナーがいない場合

    # 1人のトレーナーをランダムに選ぶ
    if trainer_a is None:
        trainer_a = random.choice(trainers)

    if random_battle:
        # trainer_a を除いたトレーナーの中からランダムに trainer_b を選ぶ
        trainer_b = random.choice([t for t in trainers if t != trainer_a])
        return trainer_a, trainer_b

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


def pokemon_battle(
    database_handler: DatabaseHandler,
    trainers: list[Trainer],
    trainer_a: Trainer,
    random_battle: bool = False,
):
    trainer_a, trainer_b = match_trainers(
        trainers, trainer_a, random_battle=random_battle
    )
    print(f"{trainer_a.name} vs {trainer_b.name} の対戦開始!")

    battle = SimulatedBattle(trainer_a, trainer_b)
    winner = battle.simulate_battle()
    print(f"勝者は {winner.name} です。")
    saved_time = battle.save_log()

    # 対戦結果に応じた Elo レーティングの更新
    if winner == trainer_a:
        result_a, result_b = 1, 0
    else:
        result_a, result_b = 0, 1

    # ここで一旦、更新前のレーティングを変数に保持する
    rating_a_old = trainer_a.sim_rating
    rating_b_old = trainer_b.sim_rating

    trainer_a.sim_rating = update_elo(rating_a_old, rating_b_old, result_a)
    trainer_b.sim_rating = update_elo(rating_b_old, rating_a_old, result_b)

    print(
        f"更新後のレーティング: {trainer_a.name}: {trainer_a.sim_rating}, {trainer_b.name}: {trainer_b.sim_rating}"
    )

    # 対戦履歴とレーティングを保存する
    # 保存するとき、トレーナー名は 順位+名前 で保存する
    database_handler.insert_battle_history(
        trainer_a_name=f"{trainer_a.rank}_{trainer_a.name}",
        trainer_b_name=f"{trainer_b.rank}_{trainer_b.name}",
        trainer_a_rating=trainer_a.sim_rating,
        trainer_b_rating=trainer_b.sim_rating,
        winner_name=f"{winner.rank}_{winner.name}",
        log_saved_time=saved_time,
    )

    # トレーナーのレーティングを更新する
    database_handler.update_trainer_rating(trainer_a.rank, trainer_a.sim_rating)
    database_handler.update_trainer_rating(trainer_b.rank, trainer_b.sim_rating)


@click.command()
# 途中から再開するオプション
@click.option(
    "--resume",
    is_flag=True,
    help="Resume the simulation from the last saved state.",
)
def main(resume: bool):
    max_battle_count = 10

    # JSON ファイルからトレーナーデータを読み込む
    trainers = load_trainers_from_json("data/top_rankers/season_27.json")

    database_handler = DatabaseHandler()

    if resume:
        # トレーナーのレーティングをデータベースから取得する
        trainer_ratings = database_handler.load_trainer_ratings()
        for trainer in trainers:
            for tr in trainer_ratings:
                if trainer.rank == tr.rank:
                    trainer.sim_rating = tr.sim_rating
    else:
        # トレーナーのレーティングを初期化
        database_handler.create_rating_table(trainers)
        # 対戦履歴を初期化
        database_handler.initialize_battle_history()

    # 対戦の履歴を取得する
    battle_history = database_handler.load_battle_history()
    # トレーナーごとの対戦回数をカウント
    battle_count = {}
    for bh in battle_history:
        if bh.trainer_a_name not in battle_count:
            battle_count[bh.trainer_a_name] = 0
        battle_count[bh.trainer_a_name] += 1

    # マッチングして対戦開始。各プレイヤーごとに10回の対戦を行う
    for trainer_a in trainers:
        # 対戦回数が10回以上のトレーナーはスキップ
        battle_count_key = f"{trainer_a.rank}_{trainer_a.name}"
        resumed_battle_count = battle_count.get(battle_count_key, 0)
        if resumed_battle_count >= 10:
            print(f"{battle_count_key} はすでに対戦済みです。")
            continue

        # 対戦回数が10回未満のトレーナーは対戦を行う
        for _ in range(max_battle_count - resumed_battle_count):
            pokemon_battle(
                database_handler=database_handler,
                trainers=trainers,
                trainer_a=trainer_a,
                random_battle=True,
            )

        send_discord_notification(f"🔥{trainer_a.name} の対戦がすべて終了しました🎱")

    # 全員の sim_rating と rating を表示
    for trainer in trainers:
        print(
            f"{trainer.name}: sim_rating={trainer.sim_rating}, rating={trainer.rating}"
        )


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"エラーが発生しました: {e}")
        send_discord_notification("🔥ポケモンシミュレータでエラーが起きています🎱")
