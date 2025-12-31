"""
Pokemon Battle Transformer Embeddings

トークン、位置、タイムステップ、セグメント、状態の埋め込み層。
"""

import math

import torch
import torch.nn as nn

from .config import PokemonBattleTransformerConfig


class PokemonBattleEmbeddings(nn.Module):
    """
    Pokemon Battle Transformer の埋め込み層

    以下を組み合わせる:
    - トークン埋め込み (Pokemon名、特殊トークン)
    - 位置埋め込み (系列位置)
    - タイムステップ埋め込み (ターン番号)
    - セグメント埋め込み (preview/selection/battle)
    - RTG埋め込み (Return-to-go)
    - 状態埋め込み (連続値特徴量)
    """

    def __init__(self, config: PokemonBattleTransformerConfig):
        super().__init__()
        self.config = config

        # トークン埋め込み (Pokemon + 特殊トークン)
        self.token_embed = nn.Embedding(
            config.total_vocab_size,
            config.hidden_size,
            padding_idx=config.pad_token_id,
        )

        # 位置埋め込み
        self.position_embed = nn.Embedding(
            config.max_sequence_length,
            config.hidden_size,
        )

        # タイムステップ埋め込み (ターン番号)
        self.timestep_embed = nn.Embedding(
            config.max_turns + 10,  # 余裕を持たせる
            config.hidden_size,
        )

        # セグメント埋め込み (0=preview, 1=selection, 2=battle)
        self.segment_embed = nn.Embedding(
            3,
            config.hidden_size,
        )

        # RTG (Return-to-go) 埋め込み
        # 連続値 [0, 1] を hidden_size に投影
        self.rtg_embed = nn.Sequential(
            nn.Linear(1, config.hidden_size),
            nn.Tanh(),
        )

        # 状態特徴量埋め込み (HP, ランク, フィールド等の連続値)
        # pokemon_state_dim + field_state_dim = 18 + 24 = 42
        state_input_dim = config.pokemon_state_dim + config.field_state_dim
        self.state_embed = nn.Sequential(
            nn.Linear(state_input_dim, config.hidden_size),
            nn.GELU(),
            nn.Linear(config.hidden_size, config.hidden_size),
        )

        # LayerNorm & Dropout
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # 重み初期化
        self._init_weights()

    def _init_weights(self):
        """重みの初期化"""
        # Embedding layers
        nn.init.normal_(self.token_embed.weight, mean=0, std=self.config.initializer_range)
        nn.init.zeros_(self.token_embed.weight[self.config.pad_token_id])
        nn.init.normal_(self.position_embed.weight, mean=0, std=self.config.initializer_range)
        nn.init.normal_(self.timestep_embed.weight, mean=0, std=self.config.initializer_range)
        nn.init.normal_(self.segment_embed.weight, mean=0, std=self.config.initializer_range)

        # Linear layers
        for module in [self.rtg_embed, self.state_embed]:
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        timestep_ids: torch.Tensor,
        segment_ids: torch.Tensor,
        rtg_values: torch.Tensor,
        state_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        埋め込みを計算

        Args:
            input_ids: [batch, seq_len] トークンID
            position_ids: [batch, seq_len] 位置ID
            timestep_ids: [batch, seq_len] ターン番号
            segment_ids: [batch, seq_len] セグメントID
            rtg_values: [batch, seq_len] Return-to-go 値
            state_features: [batch, seq_len, state_dim] 状態特徴量 (optional)

        Returns:
            [batch, seq_len, hidden_size] の埋め込みテンソル
        """
        batch_size, seq_len = input_ids.shape

        # 各埋め込みを計算
        token_embeds = self.token_embed(input_ids)  # [B, L, H]
        position_embeds = self.position_embed(position_ids)  # [B, L, H]
        timestep_embeds = self.timestep_embed(timestep_ids)  # [B, L, H]
        segment_embeds = self.segment_embed(segment_ids)  # [B, L, H]

        # RTG 埋め込み
        rtg_embeds = self.rtg_embed(rtg_values.unsqueeze(-1))  # [B, L, H]

        # 基本埋め込みの合成
        embeddings = token_embeds + position_embeds + timestep_embeds + segment_embeds + rtg_embeds

        # 状態特徴量があれば追加
        if state_features is not None:
            state_embeds = self.state_embed(state_features)  # [B, L, H]
            embeddings = embeddings + state_embeds

        # LayerNorm & Dropout
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings


class SinusoidalPositionalEncoding(nn.Module):
    """
    Sinusoidal 位置埋め込み (Transformer 論文のオリジナル)

    学習可能な位置埋め込みの代替として使用可能。
    """

    def __init__(self, config: PokemonBattleTransformerConfig):
        super().__init__()
        self.config = config

        # 位置エンコーディングを事前計算
        pe = torch.zeros(config.max_sequence_length, config.hidden_size)
        position = torch.arange(0, config.max_sequence_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, config.hidden_size, 2).float() * (-math.log(10000.0) / config.hidden_size)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # バッファとして登録（勾配なし）
        self.register_buffer("pe", pe.unsqueeze(0))  # [1, max_len, hidden]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        位置エンコーディングを追加

        Args:
            x: [batch, seq_len, hidden_size]

        Returns:
            [batch, seq_len, hidden_size]
        """
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]


class RotaryPositionalEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE)

    より新しい位置エンコーディング手法。
    長い系列に対してより良い汎化を示す。
    """

    def __init__(self, config: PokemonBattleTransformerConfig):
        super().__init__()
        self.dim = config.hidden_size // config.num_attention_heads

        # 周波数を事前計算
        inv_freq = 1.0 / (10000 ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x: torch.Tensor, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        cos, sin を計算

        Args:
            x: 入力テンソル（形状参照用）
            seq_len: 系列長

        Returns:
            (cos, sin): 各 [seq_len, dim]
        """
        t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos(), emb.sin()


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """RoPE用のヘルパー関数"""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Query と Key に RoPE を適用

    Args:
        q: [batch, heads, seq_len, head_dim]
        k: [batch, heads, seq_len, head_dim]
        cos, sin: [seq_len, head_dim]

    Returns:
        (q_embed, k_embed): RoPE適用後のQuery, Key
    """
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed
