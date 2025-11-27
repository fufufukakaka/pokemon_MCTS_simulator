import json
import random
from pathlib import Path
from typing import Iterable, List

import click

from src.llm.static_dataset import build_example_from_battle, example_to_json
from src.models import Trainer
from src.pokemon_battle_sim.battle import Battle
from src.pokemon_battle_sim.pokemon import Pokemon


def load_trainers_from_json(filename: str) -> List[Trainer]:
    """`scripts/matching.py` と同等のロジックでトレーナー情報を読み込む。"""
    from src.models import Trainer

    trainers: List[Trainer] = []
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


def init_trainer_pokemons(trainer: Trainer) -> None:
    """`SimulatedBattle.init_trainer_pokemons` の単純版。"""
    for p in trainer.raw_pokemons:
        pokemon = Pokemon(p.get("name"))
        pokemon.item = p.get("item")
        pokemon.nature = p.get("nature")
        pokemon.ability = p.get("ability")
        pokemon.Ttype = p.get("Ttype")
        pokemon.moves = p.get("moves")
        pokemon.effort = p.get("effort")
        trainer.pokemons.append(pokemon)


def sample_battles_from_trainers(
    trainers: List[Trainer], num_battles: int, seed: int | None = None
) -> Iterable[Battle]:
    """
    トレーナーの構築からランダムに 3vs3 の対面をサンプリングし、
    1 ターン目想定のシンプルな Battle を構築する。
    """
    rng = random.Random(seed)

    Pokemon.init()

    # 一度だけポケモンをインスタンス化しておく
    for tr in trainers:
        if not tr.pokemons:
            init_trainer_pokemons(tr)

    for _ in range(num_battles):
        t0, t1 = rng.sample(trainers, 2)
        team_a = t0.choose_team()
        team_b = t1.choose_team()

        battle = Battle()
        battle.reset_game()

        # シンプルに 3 体を選出し、先頭同士が対面している状態を作る
        battle.selected[0] = team_a[:3]
        battle.selected[1] = team_b[:3]
        battle.pokemon[0] = team_a[0]
        battle.pokemon[1] = team_b[0]
        battle.turn = 0

        yield battle


@click.command()
@click.option(
    "--trainer-json",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=Path("data/top_rankers/season_27.json"),
    show_default=True,
    help="トレーナー構築データの JSON パス",
)
@click.option(
    "--output",
    type=click.Path(dir_okay=False, path_type=Path),
    default=Path("data/llm_static_dataset.jsonl"),
    show_default=True,
    help="出力先 JSONL ファイル",
)
@click.option(
    "--num-battles",
    type=int,
    default=1000,
    show_default=True,
    help="サンプリングする 3vs3 バトル数（1 バトルから 2 サンプル生成）",
)
@click.option(
    "--seed",
    type=int,
    default=12345,
    show_default=True,
    help="乱数シード",
)
def main(trainer_json: Path, output: Path, num_battles: int, seed: int) -> None:
    """
    ダメージ計算 API を用いた静的教師データセットを生成する。

    使い方:
        poetry run python scripts/generate_llm_static_dataset.py \\
            --num-battles 1000 \\
            --output data/llm_static_dataset.jsonl
    """
    trainers = load_trainers_from_json(str(trainer_json))

    battles = sample_battles_from_trainers(trainers, num_battles=num_battles, seed=seed)

    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output.open("w", encoding="utf-8") as f:
        for battle in battles:
            # 両プレイヤー視点で 1 サンプルずつ生成
            for pl in (0, 1):
                ex = build_example_from_battle(battle, pl)
                f.write(example_to_json(ex) + "\n")
                count += 1

    click.echo(f"wrote {count} examples to {output}")


if __name__ == "__main__":
    main()


