"""
Team Selector

チーム選出を行うユーティリティクラス。
Team Selection Networkを使った選出や、ランダム選出を提供する。
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Optional, Protocol

import torch

from .team_selection_encoder import TeamSelectionEncoder
from .team_selection_network import TeamSelectionNetwork


class TeamSelectorProtocol(Protocol):
    """Team Selector のプロトコル"""

    def select(
        self,
        my_team: list[dict],
        opp_team: list[dict],
        num_select: int = 3,
    ) -> list[dict]:
        """チームから指定数のポケモンを選出する"""
        ...


class RandomTeamSelector:
    """ランダムに選出するセレクター"""

    def select(
        self,
        my_team: list[dict],
        opp_team: list[dict],
        num_select: int = 3,
    ) -> list[dict]:
        """ランダムに num_select 匹を選出"""
        if len(my_team) <= num_select:
            return my_team
        indices = random.sample(range(len(my_team)), num_select)
        return [my_team[i] for i in indices]


class TopNTeamSelector:
    """先頭N匹を選出するセレクター（従来の動作）"""

    def select(
        self,
        my_team: list[dict],
        opp_team: list[dict],
        num_select: int = 3,
    ) -> list[dict]:
        """先頭 num_select 匹を選出"""
        return my_team[:num_select]


class NNTeamSelector:
    """
    Neural Network を使用して選出するセレクター

    Team Selection Networkを使って、相手チームを見て最適な選出を行う。
    """

    def __init__(
        self,
        model: TeamSelectionNetwork,
        encoder: TeamSelectionEncoder,
        device: str = "cpu",
        temperature: float = 1.0,
        deterministic: bool = True,
    ):
        """
        Args:
            model: Team Selection Network
            encoder: Team Selection Encoder
            device: 実行デバイス
            temperature: サンプリング温度（deterministic=Falseの場合）
            deterministic: Trueなら確率最大の選出、Falseならサンプリング
        """
        self.model = model
        self.encoder = encoder
        self.device = device
        self.temperature = temperature
        self.deterministic = deterministic

        self.model.eval()
        self.model.to(device)

    def select(
        self,
        my_team: list[dict],
        opp_team: list[dict],
        num_select: int = 3,
    ) -> list[dict]:
        """
        NNを使ってチームを選出

        Args:
            my_team: 自チーム（最大6匹）
            opp_team: 相手チーム（最大6匹）
            num_select: 選出数

        Returns:
            選出されたポケモンのリスト
        """
        if len(my_team) <= num_select:
            return my_team

        # エンコード
        my_team_tensor = self.encoder.encode_team(my_team).unsqueeze(0).to(self.device)
        opp_team_tensor = self.encoder.encode_team(opp_team).unsqueeze(0).to(self.device)

        with torch.no_grad():
            indices, probs = self.model.select_team(
                my_team_tensor,
                opp_team_tensor,
                num_select=num_select,
                temperature=self.temperature,
                deterministic=self.deterministic,
            )

        # インデックスからポケモンを取得
        selected_indices = indices[0].cpu().tolist()
        return [my_team[i] for i in selected_indices if i < len(my_team)]

    def get_selection_probs(
        self,
        my_team: list[dict],
        opp_team: list[dict],
    ) -> list[float]:
        """
        各ポケモンの選出確率を取得

        Returns:
            各ポケモンの選出確率のリスト
        """
        my_team_tensor = self.encoder.encode_team(my_team).unsqueeze(0).to(self.device)
        opp_team_tensor = self.encoder.encode_team(opp_team).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(my_team_tensor, opp_team_tensor)
            probs = output["selection_probs"][0].cpu().tolist()

        return probs[: len(my_team)]


def load_team_selector(
    model_dir: str | Path,
    device: str = "cpu",
    temperature: float = 1.0,
    deterministic: bool = True,
) -> NNTeamSelector:
    """
    学習済みTeam Selectorを読み込む

    Args:
        model_dir: モデルディレクトリ
        device: 実行デバイス
        temperature: サンプリング温度
        deterministic: 確定的選出かどうか

    Returns:
        NNTeamSelector
    """
    import json

    model_dir = Path(model_dir)

    # エンコーダー読み込み
    encoder = TeamSelectionEncoder.load(model_dir / "encoder.json")

    # 設定読み込み
    with open(model_dir / "config.json", "r") as f:
        config = json.load(f)

    # モデル作成・読み込み
    from .team_selection_network import TeamSelectionNetworkConfig

    network_config = TeamSelectionNetworkConfig(
        pokemon_feature_dim=config.get("pokemon_feature_dim", 15),
        pokemon_embed_dim=config.get("pokemon_embed_dim", 128),
        hidden_dim=config.get("hidden_dim", 256),
        num_heads=config.get("num_heads", 4),
        num_layers=config.get("num_layers", 2),
        dropout=0.0,  # 推論時はドロップアウトなし
    )

    model = TeamSelectionNetwork(config=network_config)
    checkpoint = torch.load(model_dir / "best_model.pt", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return NNTeamSelector(
        model=model,
        encoder=encoder,
        device=device,
        temperature=temperature,
        deterministic=deterministic,
    )


class HybridTeamSelector:
    """
    ハイブリッドセレクター

    確率的にNNセレクターとランダムセレクターを切り替える。
    探索のための多様性を確保する。
    """

    def __init__(
        self,
        nn_selector: Optional[NNTeamSelector] = None,
        random_prob: float = 0.1,
    ):
        """
        Args:
            nn_selector: NNセレクター（Noneならランダムのみ）
            random_prob: ランダム選出する確率
        """
        self.nn_selector = nn_selector
        self.random_selector = RandomTeamSelector()
        self.random_prob = random_prob

    def select(
        self,
        my_team: list[dict],
        opp_team: list[dict],
        num_select: int = 3,
    ) -> list[dict]:
        """ハイブリッド選出"""
        if self.nn_selector is None or random.random() < self.random_prob:
            return self.random_selector.select(my_team, opp_team, num_select)
        return self.nn_selector.select(my_team, opp_team, num_select)
