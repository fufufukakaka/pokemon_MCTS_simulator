from typing import List, Dict, Optional

from src.pokemon_battle_sim.pokemon import Pokemon


class Damage:
    """
    Attributes:
    ----------------------------------------
    turn: int
        The turn when the damage occurred.

    attack_player: int
        The player who initiated the attack.

    pokemon: List[Optional[Pokemon]]
        Instances of Pokemon on the field.

    move: Optional[str]
        The move used for the attack.

    damage: Optional[int]
        The amount of damage dealt.

    damage_ratio: Optional[float]
        The ratio of damage.

    critical: bool
        True if the hit was critical.

    stellar: List[List[str]]
        List of types that can be enhanced by Stellar Terastal.

    condition: Dict
        The battle conditions at the time of damage.
    """

    def __init__(self):
        self.turn: int = 0
        self.attack_player: int = 0
        self.index: List[Optional[int]] = [None, None]
        self.pokemon: List[Optional[Pokemon]] = [None, None]
        self.move: Optional[str] = None
        self.damage: Optional[int] = None
        self.damage_ratio: Optional[float] = None
        self.critical: bool = False
        self.stellar: List[List[str]] = [[], []]
        self.condition: Dict = {}
