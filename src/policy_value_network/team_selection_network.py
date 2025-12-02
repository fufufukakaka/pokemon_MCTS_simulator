"""
Team Selection Network

相手の6匹を見て、自分の6匹から最適な3匹を選出するネットワーク。

アーキテクチャ:
1. 各ポケモンを個別にエンコード（共有Embedding + MLP）
2. 自チームと相手チームをそれぞれSet Encoderで集約
3. Cross Attentionで相手チームを考慮した自チーム表現を生成
4. 各ポケモンの選出スコアを出力
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TeamSelectionNetworkConfig:
    """Team Selection Network の設定"""

    # 入力次元
    pokemon_feature_dim: int = 15

    # Embedding次元
    pokemon_embed_dim: int = 128
    hidden_dim: int = 256

    # Transformer設定
    num_heads: int = 4
    num_layers: int = 2
    dropout: float = 0.1

    # 出力
    team_size: int = 6
    select_size: int = 3  # 選出する数


class PokemonEmbedding(nn.Module):
    """ポケモンの特徴量を埋め込みベクトルに変換"""

    def __init__(self, input_dim: int, embed_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, num_pokemon, feature_dim]
        Returns:
            [batch, num_pokemon, embed_dim]
        """
        return self.net(x)


class SetEncoder(nn.Module):
    """
    順序不変なSet Encoder (Transformer Encoder)

    チーム内のポケモン間の関係を学習する。
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_layers: int,
        dropout: float = 0.1,
    ):
        super().__init__()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: [batch, num_pokemon, embed_dim]
            mask: [batch, num_pokemon] パディングマスク（Trueでマスク）
        Returns:
            [batch, num_pokemon, embed_dim]
        """
        return self.encoder(x, src_key_padding_mask=mask)


class CrossAttention(nn.Module):
    """
    Cross Attention

    自チームのポケモンが相手チームを見て情報を集約する。
    """

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            query: [batch, num_my_pokemon, embed_dim] 自チーム
            key_value: [batch, num_opp_pokemon, embed_dim] 相手チーム
            key_padding_mask: [batch, num_opp_pokemon]
        Returns:
            [batch, num_my_pokemon, embed_dim]
        """
        attn_out, _ = self.attn(
            query, key_value, key_value, key_padding_mask=key_padding_mask
        )
        return self.norm(query + self.dropout(attn_out))


class TeamSelectionNetwork(nn.Module):
    """
    Team Selection Network

    相手チームを見て、自チームから最適な3匹を選出する。

    出力:
    - selection_logits: [batch, 6] 各ポケモンの選出スコア
    - value: [batch, 1] この選出でのマッチアップ勝率予測（オプション）
    """

    def __init__(self, config: Optional[TeamSelectionNetworkConfig] = None):
        super().__init__()
        self.config = config or TeamSelectionNetworkConfig()

        # ポケモン埋め込み（自チーム・相手チーム共通）
        self.pokemon_embedding = PokemonEmbedding(
            input_dim=self.config.pokemon_feature_dim,
            embed_dim=self.config.pokemon_embed_dim,
            dropout=self.config.dropout,
        )

        # 自チームSet Encoder
        self.my_team_encoder = SetEncoder(
            embed_dim=self.config.pokemon_embed_dim,
            num_heads=self.config.num_heads,
            num_layers=self.config.num_layers,
            dropout=self.config.dropout,
        )

        # 相手チームSet Encoder
        self.opp_team_encoder = SetEncoder(
            embed_dim=self.config.pokemon_embed_dim,
            num_heads=self.config.num_heads,
            num_layers=self.config.num_layers,
            dropout=self.config.dropout,
        )

        # Cross Attention（自チーム→相手チームを見る）
        self.cross_attention = CrossAttention(
            embed_dim=self.config.pokemon_embed_dim,
            num_heads=self.config.num_heads,
            dropout=self.config.dropout,
        )

        # 選出スコア出力
        self.selection_head = nn.Sequential(
            nn.Linear(self.config.pokemon_embed_dim, self.config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.hidden_dim, 1),
        )

        # Value Head（勝率予測、オプショナル）
        self.value_head = nn.Sequential(
            nn.Linear(self.config.pokemon_embed_dim * 2, self.config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        my_team: torch.Tensor,
        opp_team: torch.Tensor,
        my_mask: Optional[torch.Tensor] = None,
        opp_mask: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            my_team: [batch, 6, feature_dim] 自チーム
            opp_team: [batch, 6, feature_dim] 相手チーム
            my_mask: [batch, 6] 自チームのパディングマスク
            opp_mask: [batch, 6] 相手チームのパディングマスク

        Returns:
            {
                "selection_logits": [batch, 6] 各ポケモンの選出ロジット
                "selection_probs": [batch, 6] 選出確率（softmax後）
                "value": [batch, 1] 勝率予測
            }
        """
        batch_size = my_team.size(0)

        # ポケモン埋め込み
        my_embed = self.pokemon_embedding(my_team)  # [batch, 6, embed_dim]
        opp_embed = self.pokemon_embedding(opp_team)  # [batch, 6, embed_dim]

        # Set Encoding（チーム内の関係を学習）
        my_encoded = self.my_team_encoder(my_embed, mask=my_mask)
        opp_encoded = self.opp_team_encoder(opp_embed, mask=opp_mask)

        # Cross Attention（相手チームを考慮）
        my_with_opp = self.cross_attention(
            my_encoded, opp_encoded, key_padding_mask=opp_mask
        )

        # 選出スコア
        selection_logits = self.selection_head(my_with_opp).squeeze(-1)  # [batch, 6]

        # マスクされたポケモンのスコアを-infに
        if my_mask is not None:
            selection_logits = selection_logits.masked_fill(my_mask, float("-inf"))

        selection_probs = F.softmax(selection_logits, dim=-1)

        # Value予測
        # チーム全体の表現を集約（mean pooling）
        if my_mask is not None:
            my_mask_expanded = my_mask.unsqueeze(-1).float()
            my_pooled = (my_with_opp * (1 - my_mask_expanded)).sum(dim=1) / (
                (1 - my_mask_expanded).sum(dim=1) + 1e-8
            )
        else:
            my_pooled = my_with_opp.mean(dim=1)

        if opp_mask is not None:
            opp_mask_expanded = opp_mask.unsqueeze(-1).float()
            opp_pooled = (opp_encoded * (1 - opp_mask_expanded)).sum(dim=1) / (
                (1 - opp_mask_expanded).sum(dim=1) + 1e-8
            )
        else:
            opp_pooled = opp_encoded.mean(dim=1)

        combined = torch.cat([my_pooled, opp_pooled], dim=-1)
        value = self.value_head(combined)

        return {
            "selection_logits": selection_logits,
            "selection_probs": selection_probs,
            "value": value,
        }

    def select_team(
        self,
        my_team: torch.Tensor,
        opp_team: torch.Tensor,
        num_select: int = 3,
        temperature: float = 1.0,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        チームを選出する

        Args:
            my_team: [batch, 6, feature_dim] or [6, feature_dim]
            opp_team: [batch, 6, feature_dim] or [6, feature_dim]
            num_select: 選出する数
            temperature: サンプリング温度
            deterministic: Trueなら確率最大の3匹を選択

        Returns:
            selected_indices: [batch, num_select] 選出されたインデックス
            selection_probs: [batch, 6] 選出確率
        """
        # バッチ次元を追加
        if my_team.dim() == 2:
            my_team = my_team.unsqueeze(0)
            opp_team = opp_team.unsqueeze(0)

        output = self.forward(my_team, opp_team)
        logits = output["selection_logits"] / temperature

        if deterministic:
            # 上位num_select個を選択
            _, indices = torch.topk(logits, num_select, dim=-1)
        else:
            # 確率的にサンプリング（重複なし）
            probs = F.softmax(logits, dim=-1)
            indices = torch.zeros(
                my_team.size(0), num_select, dtype=torch.long, device=my_team.device
            )

            for b in range(my_team.size(0)):
                remaining_probs = probs[b].clone()
                for i in range(num_select):
                    idx = torch.multinomial(remaining_probs, 1)
                    indices[b, i] = idx
                    remaining_probs[idx] = 0  # 選択済みは0に
                    remaining_probs = remaining_probs / (
                        remaining_probs.sum() + 1e-8
                    )  # 再正規化

        return indices, output["selection_probs"]


class TeamSelectionLoss(nn.Module):
    """
    Team Selection用の損失関数

    1. Selection Loss: 正解の選出との交差エントロピー
    2. Value Loss: 勝敗予測のMSE
    """

    def __init__(self, selection_weight: float = 1.0, value_weight: float = 0.5):
        super().__init__()
        self.selection_weight = selection_weight
        self.value_weight = value_weight

    def forward(
        self,
        selection_logits: torch.Tensor,
        selection_target: torch.Tensor,
        value_pred: torch.Tensor,
        value_target: torch.Tensor,
        selection_mask: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            selection_logits: [batch, 6] 選出ロジット
            selection_target: [batch, 6] 選出ラベル（0 or 1）
            value_pred: [batch, 1] 勝率予測
            value_target: [batch, 1] 実際の勝敗
            selection_mask: [batch, 6] マスク（1で有効）

        Returns:
            {"loss": total_loss, "selection_loss": ..., "value_loss": ...}
        """
        # Selection Loss（Binary Cross Entropy）
        if selection_mask is not None:
            # マスクされた部分を除外
            selection_logits_masked = selection_logits.clone()
            selection_logits_masked[selection_mask == 0] = -1e9

        selection_probs = torch.sigmoid(selection_logits)
        selection_loss = F.binary_cross_entropy(
            selection_probs, selection_target.float(), reduction="mean"
        )

        # Value Loss
        value_loss = F.mse_loss(value_pred, value_target)

        # Total Loss
        total_loss = (
            self.selection_weight * selection_loss + self.value_weight * value_loss
        )

        return {
            "loss": total_loss,
            "selection_loss": selection_loss,
            "value_loss": value_loss,
        }
