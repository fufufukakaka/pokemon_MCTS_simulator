import requests
import random
import numpy as np

class BattleState:
    def __init__(self, attacker, defender):
        self.attacker = attacker  # 攻撃側ポケモンの情報
        self.defender = defender  # 防御側ポケモンの情報
        self.history = []  # 行動履歴

    def get_legal_actions(self):
        """可能な行動（技4つ + 交代）を取得"""
        #return self.attacker["moves"] + ["switch"]
        return self.attacker["moves"]

    def perform_action(self, action):
        """アクションを実行し、状態を遷移"""
        if action == "switch":
            # 交代処理（仮のロジック）
            new_attacker = self.attacker.copy()
            new_defender = self.defender.copy()
            return BattleState(new_attacker, new_defender)

        # APIでダメージ計算
        damage = self.calculate_damage(self.attacker, self.defender, action)
        new_defender = self.defender.copy()
        new_defender["hp"] -= damage  # HPを減らす

        return BattleState(self.attacker, new_defender)

    def calculate_damage(self, attacker, defender, move):
        """ダメージ計算APIを呼び出す"""
        response = requests.post("http://localhost:3000/calculate", json={
            "attacker": attacker,
            "defender": defender,
            "move": move
        })
        damages: int | list[int] = response.json()["damage"]
        if damages == 0:
            max_damges = 0
        else:
            max_damges = max(damages)
        return max_damges

    def is_terminal(self):
        """試合が終了したか（どちらかのHPが0）"""
        return self.defender["hp"] <= 0 or self.attacker["hp"] <= 0

    def get_reward(self):
        """試合の結果を評価（ダメージ量も考慮）"""
        if self.defender["hp"] <= 0:
            return 1  # 勝利
        elif self.attacker["hp"] <= 0:
            return -1  # 敗北
        else:
            return (self.initial_defender_hp - self.defender["hp"]) / self.initial_defender_hp


class MCTSNode:
    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent
        self.children = {}
        self.visits = 0
        self.value = 0

    def select(self):
        """UCT（探索と活用のバランスを取る）で子ノードを選択"""
        best_score = -np.inf
        best_action = None

        for action, child in self.children.items():
            uct_score = child.value / (child.visits + 1) + np.sqrt(np.log(self.visits + 1) / (child.visits + 1))
            if uct_score > best_score:
                best_score = uct_score
                best_action = action

        return self.children[best_action]

    def expand(self):
        """新しいノードを追加"""
        actions = self.state.get_legal_actions()
        for action in actions:
            if action not in self.children:
                new_state = self.state.perform_action(action)
                self.children[action] = MCTSNode(new_state, parent=self)

    def simulate(self):
        """ランダムに試合を進める"""
        current_state = self.state
        while not current_state.is_terminal():
            action = random.choice(current_state.get_legal_actions())
            current_state = current_state.perform_action(action)
        return current_state.get_reward()

    def backpropagate(self, reward):
        """結果を親ノードに反映"""
        self.visits += 1
        self.value += reward
        if self.parent:
            self.parent.backpropagate(reward)


def mcts_search(state, iterations=100):
    root = MCTSNode(state)

    for _ in range(iterations):
        node = root
        while node.children:
            node = node.select()
        if not node.state.is_terminal():
            node.expand()
        reward = node.simulate()
        node.backpropagate(reward)

    # 最も訪問回数の多いアクションを選択
    best_action = max(root.children, key=lambda action: root.children[action].visits)
    return best_action


if __name__ == "__main__":
    # 例：ゲンガー vs ラッキーのバトル
    attacker = {
        "name": "Gengar",
        "item": "Choice Specs",
        "nature": "Timid",
        "evs": {"spa": 252},
        "hp": 135,  # 仮のHP
        "moves": ["Shadow Ball", "Sludge Bomb", "Thunderbolt", "Focus Blast"]
    }

    defender = {
        "name": "Chansey",
        "item": "Eviolite",
        "nature": "Calm",
        "evs": {"hp": 252, "spd": 252},
        "hp": 357  # 仮のHP
    }

    # MCTSで最適な行動を選ぶ
    battle_state = BattleState(attacker, defender)
    best_action = mcts_search(battle_state, iterations=500)

    print(f"推奨行動: {best_action}")
