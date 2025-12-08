"""
Team Selection Network

相手の6匹を見て、自分の6匹から最適な3匹を選出するネットワーク。
また、選出した3匹のうち誰を先発にするかも予測する。

アーキテクチャ:
1. 各ポケモンを個別にエンコード（共有Embedding + MLP）
2. 自チームと相手チームをそれぞれSet Encoderで集約
3. Cross Attentionで相手チームを考慮した自チーム表現を生成
4. 各ポケモンの選出スコアを出力
5. 選出されたポケモンの中から先発スコアを出力
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

    # 先発予測を有効にするか
    predict_lead: bool = True


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

        # 先発スコア出力（選出されたポケモンの中から先発を決める）
        if self.config.predict_lead:
            self.lead_head = nn.Sequential(
                nn.Linear(self.config.pokemon_embed_dim, self.config.hidden_dim),
                nn.ReLU(),
                nn.Dropout(self.config.dropout),
                nn.Linear(self.config.hidden_dim, 1),
            )
        else:
            self.lead_head = None

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
        selection_mask: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            my_team: [batch, 6, feature_dim] 自チーム
            opp_team: [batch, 6, feature_dim] 相手チーム
            my_mask: [batch, 6] 自チームのパディングマスク
            opp_mask: [batch, 6] 相手チームのパディングマスク
            selection_mask: [batch, 6] 選出マスク（Trueで選出済み、先発予測時に使用）

        Returns:
            {
                "selection_logits": [batch, 6] 各ポケモンの選出ロジット
                "selection_probs": [batch, 6] 選出確率（softmax後）
                "lead_logits": [batch, 6] 各ポケモンの先発ロジット（選出されていないポケモンは-inf）
                "lead_probs": [batch, 6] 先発確率（選出されたポケモン内でsoftmax）
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

        # 先発スコア
        if self.lead_head is not None:
            lead_logits = self.lead_head(my_with_opp).squeeze(-1)  # [batch, 6]

            # 選出されていないポケモンは先発になれない
            # selection_mask が与えられた場合はそれを使う
            # 与えられない場合は、selection_probsから上位3匹を選出済みとみなす
            if selection_mask is not None:
                # selection_mask: True = 選出済み
                lead_mask = ~selection_mask  # True = 選出されていない（マスク対象）
            else:
                # selection_probsから上位3匹を選出済みとみなす
                _, top_indices = torch.topk(selection_probs, self.config.select_size, dim=-1)
                lead_mask = torch.ones_like(lead_logits, dtype=torch.bool)
                for b in range(batch_size):
                    lead_mask[b, top_indices[b]] = False

            # パディングマスクも適用
            if my_mask is not None:
                lead_mask = lead_mask | my_mask

            lead_logits = lead_logits.masked_fill(lead_mask, float("-inf"))
            lead_probs = F.softmax(lead_logits, dim=-1)
        else:
            lead_logits = torch.zeros_like(selection_logits)
            lead_probs = torch.zeros_like(selection_probs)

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
            "lead_logits": lead_logits,
            "lead_probs": lead_probs,
            "value": value,
        }

    def select_team(
        self,
        my_team: torch.Tensor,
        opp_team: torch.Tensor,
        num_select: int = 3,
        temperature: float = 1.0,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        チームを選出する

        Args:
            my_team: [batch, 6, feature_dim] or [6, feature_dim]
            opp_team: [batch, 6, feature_dim] or [6, feature_dim]
            num_select: 選出する数
            temperature: サンプリング温度
            deterministic: Trueなら確率最大の3匹を選択

        Returns:
            selected_indices: [batch, num_select] 選出されたインデックス（先発が[0]）
            selection_probs: [batch, 6] 選出確率
            lead_index: [batch] 先発のインデックス
            lead_probs: [batch, 6] 先発確率
        """
        # バッチ次元を追加
        if my_team.dim() == 2:
            my_team = my_team.unsqueeze(0)
            opp_team = opp_team.unsqueeze(0)

        batch_size = my_team.size(0)

        # まず選出を決定
        output = self.forward(my_team, opp_team)
        selection_logits = output["selection_logits"] / temperature

        if deterministic:
            # 上位num_select個を選択
            _, indices = torch.topk(selection_logits, num_select, dim=-1)
        else:
            # 確率的にサンプリング（重複なし）
            probs = F.softmax(selection_logits, dim=-1)
            indices = torch.zeros(
                batch_size, num_select, dtype=torch.long, device=my_team.device
            )

            for b in range(batch_size):
                remaining_probs = probs[b].clone()
                for i in range(num_select):
                    idx = torch.multinomial(remaining_probs, 1)
                    indices[b, i] = idx
                    remaining_probs[idx] = 0  # 選択済みは0に
                    remaining_probs = remaining_probs / (
                        remaining_probs.sum() + 1e-8
                    )  # 再正規化

        # 選出マスクを作成して先発を決定
        selection_mask = torch.zeros(batch_size, 6, dtype=torch.bool, device=my_team.device)
        for b in range(batch_size):
            selection_mask[b, indices[b]] = True

        # 先発確率を計算
        output_with_selection = self.forward(my_team, opp_team, selection_mask=selection_mask)
        lead_logits = output_with_selection["lead_logits"] / temperature
        lead_probs = output_with_selection["lead_probs"]

        # 先発を決定
        if deterministic:
            lead_index = torch.argmax(lead_logits, dim=-1)
        else:
            # 選出されたポケモンの中からサンプリング
            lead_index = torch.zeros(batch_size, dtype=torch.long, device=my_team.device)
            for b in range(batch_size):
                valid_probs = lead_probs[b].clone()
                if valid_probs.sum() > 0:
                    lead_index[b] = torch.multinomial(valid_probs, 1)
                else:
                    # フォールバック: 最初の選出ポケモン
                    lead_index[b] = indices[b, 0]

        # indicesを並び替え（先発が最初に来るように）
        for b in range(batch_size):
            lead_pos = (indices[b] == lead_index[b]).nonzero(as_tuple=True)[0]
            if len(lead_pos) > 0 and lead_pos[0] != 0:
                # 先発を先頭に移動
                pos = lead_pos[0].item()
                indices[b] = torch.cat([
                    indices[b, pos:pos+1],
                    indices[b, :pos],
                    indices[b, pos+1:]
                ])

        return indices, output["selection_probs"], lead_index, lead_probs


class TeamSelectionLoss(nn.Module):
    """
    Team Selection用の損失関数

    1. Selection Loss: 正解の選出との交差エントロピー
    2. Lead Loss: 正解の先発との交差エントロピー
    3. Value Loss: 勝敗予測のMSE
    """

    def __init__(
        self,
        selection_weight: float = 1.0,
        lead_weight: float = 0.5,
        value_weight: float = 0.5,
    ):
        super().__init__()
        self.selection_weight = selection_weight
        self.lead_weight = lead_weight
        self.value_weight = value_weight

    def forward(
        self,
        selection_logits: torch.Tensor,
        selection_target: torch.Tensor,
        value_pred: torch.Tensor,
        value_target: torch.Tensor,
        lead_logits: Optional[torch.Tensor] = None,
        lead_target: Optional[torch.Tensor] = None,
        selection_mask: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            selection_logits: [batch, 6] 選出ロジット
            selection_target: [batch, 6] 選出ラベル（0 or 1）
            value_pred: [batch, 1] 勝率予測
            value_target: [batch, 1] 実際の勝敗
            lead_logits: [batch, 6] 先発ロジット（オプション）
            lead_target: [batch] 先発のインデックス（オプション）
            selection_mask: [batch, 6] マスク（1で有効）

        Returns:
            {"loss": total_loss, "selection_loss": ..., "lead_loss": ..., "value_loss": ...}
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

        # Lead Loss（Cross Entropy - 選出された3匹の中から1匹を選ぶ）
        if lead_logits is not None and lead_target is not None:
            # 選出されていないポケモンはマスク（-infになっているはず）
            lead_loss = F.cross_entropy(lead_logits, lead_target, reduction="mean")
        else:
            lead_loss = torch.tensor(0.0, device=selection_logits.device)

        # Value Loss
        value_loss = F.mse_loss(value_pred, value_target)

        # Total Loss
        total_loss = (
            self.selection_weight * selection_loss
            + self.lead_weight * lead_loss
            + self.value_weight * value_loss
        )

        return {
            "loss": total_loss,
            "selection_loss": selection_loss,
            "lead_loss": lead_loss,
            "value_loss": value_loss,
        }
