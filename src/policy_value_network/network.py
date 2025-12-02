"""
Policy-Value Network

盤面情報からPolicy（行動確率分布）とValue（勝率）を予測する
ニューラルネットワーク。
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """残差ブロック"""

    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm1(x)
        x = F.gelu(self.fc1(x))
        x = self.dropout(x)
        x = self.norm2(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x + residual


class PolicyValueNetwork(nn.Module):
    """
    Policy-Value Network

    入力: 盤面のフラットなテンソル表現
    出力:
        - policy: 各行動の確率分布 [batch, num_actions]
        - value: 勝率 [batch, 1]
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_res_blocks: int = 4,
        num_actions: int = 10,  # 技4 + 交代3 + テラスタル技4 程度
        dropout: float = 0.1,
    ):
        """
        Args:
            input_dim: 入力特徴量の次元（ObservationEncoder.get_flat_dim()）
            hidden_dim: 隠れ層の次元
            num_res_blocks: 残差ブロック数
            num_actions: 行動数（Policy headの出力次元）
            dropout: ドロップアウト率
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_actions = num_actions

        # 入力層
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # 共通の特徴抽出層（残差ブロック）
        self.res_blocks = nn.ModuleList(
            [ResidualBlock(hidden_dim, dropout) for _ in range(num_res_blocks)]
        )

        # Policy Head
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_actions),
        )

        # Value Head
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),  # 勝率は0〜1
        )

        # 重み初期化
        self._init_weights()

    def _init_weights(self):
        """重み初期化"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self, x: torch.Tensor, action_mask: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        順伝播

        Args:
            x: 入力テンソル [batch, input_dim]
            action_mask: 有効な行動のマスク [batch, num_actions]
                         1 = 有効, 0 = 無効

        Returns:
            policy: 行動確率分布 [batch, num_actions]
            value: 勝率 [batch, 1]
        """
        # 入力層
        h = self.input_layer(x)

        # 残差ブロック
        for block in self.res_blocks:
            h = block(h)

        # Policy Head
        policy_logits = self.policy_head(h)

        # マスク適用（無効な行動に大きな負の値を設定）
        if action_mask is not None:
            policy_logits = policy_logits.masked_fill(action_mask == 0, float("-inf"))

        policy = F.softmax(policy_logits, dim=-1)

        # Value Head
        value = self.value_head(h)

        return policy, value

    def get_policy_logits(
        self, x: torch.Tensor, action_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Policy logitsを取得（学習時に使用）"""
        h = self.input_layer(x)
        for block in self.res_blocks:
            h = block(h)

        policy_logits = self.policy_head(h)

        if action_mask is not None:
            policy_logits = policy_logits.masked_fill(action_mask == 0, float("-inf"))

        return policy_logits


class PolicyValueNetworkWithEmbedding(nn.Module):
    """
    埋め込み層を持つPolicy-Value Network

    ポケモンID、技ID、持ち物IDなどを埋め込みベクトルに変換してから処理する。
    より表現力の高いモデル。
    """

    def __init__(
        self,
        num_pokemon: int = 1500,
        num_moves: int = 1000,
        num_items: int = 500,
        num_abilities: int = 400,
        num_types: int = 20,
        pokemon_embed_dim: int = 64,
        move_embed_dim: int = 32,
        item_embed_dim: int = 32,
        ability_embed_dim: int = 32,
        type_embed_dim: int = 16,
        hidden_dim: int = 256,
        num_res_blocks: int = 4,
        num_actions: int = 10,
        dropout: float = 0.1,
    ):
        super().__init__()

        # 埋め込み層
        self.pokemon_embed = nn.Embedding(num_pokemon, pokemon_embed_dim, padding_idx=0)
        self.move_embed = nn.Embedding(num_moves, move_embed_dim, padding_idx=0)
        self.item_embed = nn.Embedding(num_items, item_embed_dim, padding_idx=0)
        self.ability_embed = nn.Embedding(num_abilities, ability_embed_dim, padding_idx=0)
        self.type_embed = nn.Embedding(num_types, type_embed_dim, padding_idx=0)

        # ポケモン1体の特徴量次元
        # pokemon_embed + hp_ratio(1) + ailment(7) + rank(8) +
        # type_embed*2 + ability_embed + item_embed + move_embed*4 +
        # terastallized(1) + tera_type_embed
        pokemon_feature_dim = (
            pokemon_embed_dim
            + 1  # hp_ratio
            + 7  # ailment
            + 8  # rank
            + type_embed_dim * 2  # types
            + ability_embed_dim
            + item_embed_dim
            + move_embed_dim * 4  # moves
            + 1  # terastallized
            + type_embed_dim  # tera_type
        )

        # 総入力次元
        # ポケモン6体 + 場の状態(24) + 信念状態
        self.pokemon_feature_dim = pokemon_feature_dim
        total_input_dim = pokemon_feature_dim * 6 + 24 + 60  # 60 = belief features

        self.hidden_dim = hidden_dim
        self.num_actions = num_actions

        # 入力層
        self.input_layer = nn.Sequential(
            nn.Linear(total_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # 残差ブロック
        self.res_blocks = nn.ModuleList(
            [ResidualBlock(hidden_dim, dropout) for _ in range(num_res_blocks)]
        )

        # Policy Head
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_actions),
        )

        # Value Head
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0, std=0.02)
                if m.padding_idx is not None:
                    nn.init.zeros_(m.weight[m.padding_idx])

    def encode_pokemon_batch(
        self,
        pokemon_ids: torch.Tensor,
        hp_ratios: torch.Tensor,
        ailments: torch.Tensor,
        ranks: torch.Tensor,
        type_ids: torch.Tensor,
        ability_ids: torch.Tensor,
        item_ids: torch.Tensor,
        move_ids: torch.Tensor,
        terastallized: torch.Tensor,
        tera_type_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        ポケモンのバッチをエンコード

        Args:
            pokemon_ids: [batch, num_pokemon]
            hp_ratios: [batch, num_pokemon]
            ailments: [batch, num_pokemon, 7]
            ranks: [batch, num_pokemon, 8]
            type_ids: [batch, num_pokemon, 2]
            ability_ids: [batch, num_pokemon]
            item_ids: [batch, num_pokemon]
            move_ids: [batch, num_pokemon, 4]
            terastallized: [batch, num_pokemon]
            tera_type_ids: [batch, num_pokemon]

        Returns:
            [batch, num_pokemon, pokemon_feature_dim]
        """
        batch_size, num_pokemon = pokemon_ids.shape

        # 埋め込み
        pokemon_emb = self.pokemon_embed(pokemon_ids)  # [B, N, D]
        type_emb = self.type_embed(type_ids)  # [B, N, 2, D]
        type_emb = type_emb.view(batch_size, num_pokemon, -1)  # [B, N, 2*D]
        ability_emb = self.ability_embed(ability_ids)  # [B, N, D]
        item_emb = self.item_embed(item_ids)  # [B, N, D]
        move_emb = self.move_embed(move_ids)  # [B, N, 4, D]
        move_emb = move_emb.view(batch_size, num_pokemon, -1)  # [B, N, 4*D]
        tera_type_emb = self.type_embed(tera_type_ids)  # [B, N, D]

        # 連結
        features = torch.cat(
            [
                pokemon_emb,
                hp_ratios.unsqueeze(-1),
                ailments,
                ranks,
                type_emb,
                ability_emb,
                item_emb,
                move_emb,
                terastallized.unsqueeze(-1),
                tera_type_emb,
            ],
            dim=-1,
        )

        return features

    def forward(
        self,
        pokemon_ids: torch.Tensor,
        hp_ratios: torch.Tensor,
        ailments: torch.Tensor,
        ranks: torch.Tensor,
        type_ids: torch.Tensor,
        ability_ids: torch.Tensor,
        item_ids: torch.Tensor,
        move_ids: torch.Tensor,
        terastallized: torch.Tensor,
        tera_type_ids: torch.Tensor,
        field_features: torch.Tensor,
        belief_features: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """順伝播"""
        # ポケモンエンコード
        pokemon_features = self.encode_pokemon_batch(
            pokemon_ids,
            hp_ratios,
            ailments,
            ranks,
            type_ids,
            ability_ids,
            item_ids,
            move_ids,
            terastallized,
            tera_type_ids,
        )

        # フラット化して連結
        batch_size = pokemon_ids.shape[0]
        pokemon_flat = pokemon_features.view(batch_size, -1)

        x = torch.cat([pokemon_flat, field_features, belief_features], dim=-1)

        # 入力層
        h = self.input_layer(x)

        # 残差ブロック
        for block in self.res_blocks:
            h = block(h)

        # Policy Head
        policy_logits = self.policy_head(h)
        if action_mask is not None:
            policy_logits = policy_logits.masked_fill(action_mask == 0, float("-inf"))
        policy = F.softmax(policy_logits, dim=-1)

        # Value Head
        value = self.value_head(h)

        return policy, value
