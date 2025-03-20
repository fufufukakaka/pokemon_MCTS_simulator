import random


class Trainer:
    def __init__(self, name: str, rank: int, rating: int, pokemons, raw_pokemons: list[dict[str, str | int]]):
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
