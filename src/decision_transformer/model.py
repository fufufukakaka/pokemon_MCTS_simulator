"""
Pokemon Battle Transformer

選出からバトル行動までを統一的に扱う Decision Transformer モデル。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import PokemonBattleTransformerConfig
from .embeddings import PokemonBattleEmbeddings
from .heads import ActionHead, SelectionHead, ValueHead

if TYPE_CHECKING:
    from src.pokemon_battle_sim.battle import Battle

    from .tokenizer import BattleSequenceTokenizer


class TransformerBlock(nn.Module):
    """
    Transformer ブロック (Pre-LayerNorm)

    Self-Attention + FFN with residual connections
    """

    def __init__(self, config: PokemonBattleTransformerConfig):
        super().__init__()
        self.config = config

        # Self-Attention
        self.attention = nn.MultiheadAttention(
            embed_dim=config.hidden_size,
            num_heads=config.num_attention_heads,
            dropout=config.attention_probs_dropout_prob,
            batch_first=True,
        )

        # Feed-Forward Network
        self.ffn = nn.Sequential(
            nn.Linear(config.hidden_size, config.intermediate_size),
            nn.GELU(),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(config.intermediate_size, config.hidden_size),
            nn.Dropout(config.hidden_dropout_prob),
        )

        # Layer Normalization (Pre-norm)
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # Dropout
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        causal_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Forward pass

        Args:
            hidden_states: [batch, seq_len, hidden_size]
            attention_mask: [batch, seq_len] パディングマスク
            causal_mask: [seq_len, seq_len] 因果マスク

        Returns:
            [batch, seq_len, hidden_size]
        """
        # Self-Attention with Pre-LayerNorm
        residual = hidden_states
        hidden_states = self.norm1(hidden_states)

        # Attention mask を key_padding_mask 形式に変換
        key_padding_mask = None
        if attention_mask is not None:
            key_padding_mask = (attention_mask == 0)  # True = masked

        hidden_states, _ = self.attention(
            query=hidden_states,
            key=hidden_states,
            value=hidden_states,
            attn_mask=causal_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        hidden_states = self.dropout(hidden_states)
        hidden_states = residual + hidden_states

        # FFN with Pre-LayerNorm
        residual = hidden_states
        hidden_states = self.norm2(hidden_states)
        hidden_states = self.ffn(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class PokemonBattleTransformer(nn.Module):
    """
    Pokemon Battle Transformer

    Decision Transformer ベースのモデル。
    選出とバトル行動を統一的に処理する。
    """

    def __init__(self, config: PokemonBattleTransformerConfig):
        super().__init__()
        self.config = config

        # 埋め込み層
        self.embeddings = PokemonBattleEmbeddings(config)

        # Transformer ブロック
        self.layers = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.num_hidden_layers)
        ])

        # 最終 LayerNorm
        self.final_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # 出力ヘッド
        self.selection_head = SelectionHead(config)
        self.action_head = ActionHead(config)
        self.value_head = ValueHead(config)

        # 因果マスクのキャッシュ
        self._causal_mask_cache = {}

        # 重み初期化
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        """重みの初期化"""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                nn.init.zeros_(module.weight[module.padding_idx])

    def _get_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """因果マスクを取得（キャッシュ付き）"""
        if seq_len not in self._causal_mask_cache:
            # 上三角を -inf でマスク
            mask = torch.triu(
                torch.ones(seq_len, seq_len, device=device) * float("-inf"),
                diagonal=1,
            )
            self._causal_mask_cache[seq_len] = mask
        return self._causal_mask_cache[seq_len].to(device)

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        timestep_ids: torch.Tensor,
        segment_ids: torch.Tensor,
        rtg_values: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        state_features: torch.Tensor | None = None,
        team_token_positions: torch.Tensor | None = None,
        action_mask: torch.Tensor | None = None,
        use_causal_mask: bool = True,
        return_hidden_states: bool = False,
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass

        Args:
            input_ids: [batch, seq_len] トークンID
            position_ids: [batch, seq_len] 位置ID
            timestep_ids: [batch, seq_len] ターン番号
            segment_ids: [batch, seq_len] セグメントID
            rtg_values: [batch, seq_len] Return-to-go
            attention_mask: [batch, seq_len] パディングマスク
            state_features: [batch, seq_len, state_dim] 状態特徴量
            team_token_positions: [batch, 6] 選出予測用のトークン位置
            action_mask: [batch, num_actions] 行動マスク
            use_causal_mask: 因果マスクを使用するか
            return_hidden_states: hidden states を返すか

        Returns:
            出力辞書 {
                "selection_logits": [batch, 6, 3],
                "action_logits": [batch, num_actions],
                "value": [batch, 1],
                "hidden_states": [batch, seq_len, hidden_size] (optional),
            }
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        # 埋め込み
        hidden_states = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            timestep_ids=timestep_ids,
            segment_ids=segment_ids,
            rtg_values=rtg_values,
            state_features=state_features,
        )

        # 因果マスク
        causal_mask = None
        if use_causal_mask:
            causal_mask = self._get_causal_mask(seq_len, device)

        # Transformer layers
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                attention_mask=attention_mask,
                causal_mask=causal_mask,
            )

        # 最終 LayerNorm
        hidden_states = self.final_norm(hidden_states)

        # 出力を計算
        outputs = {}

        # 選出予測
        if team_token_positions is not None:
            outputs["selection_logits"] = self.selection_head(
                hidden_states, team_token_positions
            )

        # 行動予測（最後のトークン位置で）
        outputs["action_logits"] = self.action_head(
            hidden_states[:, -1, :], action_mask
        )

        # 価値予測
        outputs["value"] = self.value_head(hidden_states[:, -1, :])

        if return_hidden_states:
            outputs["hidden_states"] = hidden_states

        return outputs

    def get_selection(
        self,
        my_team: list[str],
        opp_team: list[str],
        tokenizer: "BattleSequenceTokenizer",
        target_return: float = 1.0,
        deterministic: bool = True,
        temperature: float = 1.0,
    ) -> tuple[list[int], int]:
        """
        チーム選出を予測

        Args:
            my_team: 自チームのポケモン名リスト (6匹)
            opp_team: 相手チームのポケモン名リスト (6匹)
            tokenizer: BattleSequenceTokenizer
            target_return: 目標リターン (1.0 = 勝ち)
            deterministic: True なら決定的、False ならサンプリング
            temperature: サンプリング時の温度

        Returns:
            (selected_indices, lead_index): 選出3匹と先発インデックス
        """
        self.eval()
        device = next(self.parameters()).device

        with torch.no_grad():
            # チームプレビューをエンコード
            encoded = tokenizer.encode_team_preview(my_team, opp_team, rtg=target_return)

            # バッチ次元を追加してデバイスに移動
            input_ids = encoded["input_ids"].unsqueeze(0).to(device)
            position_ids = encoded["position_ids"].unsqueeze(0).to(device)
            timestep_ids = encoded["timestep_ids"].unsqueeze(0).to(device)
            segment_ids = encoded["segment_ids"].unsqueeze(0).to(device)
            rtg_values = encoded["rtg_values"].unsqueeze(0).to(device)
            attention_mask = encoded["attention_mask"].unsqueeze(0).to(device)
            team_token_positions = encoded["team_token_positions"].unsqueeze(0).to(device)

            # Forward
            outputs = self.forward(
                input_ids=input_ids,
                position_ids=position_ids,
                timestep_ids=timestep_ids,
                segment_ids=segment_ids,
                rtg_values=rtg_values,
                attention_mask=attention_mask,
                team_token_positions=team_token_positions,
                use_causal_mask=True,
            )

            # 選出を予測
            return self.selection_head.predict(
                outputs.get("hidden_states", None) or self._get_hidden_for_selection(outputs),
                team_token_positions,
                deterministic=deterministic,
                temperature=temperature,
            )

    def get_action(
        self,
        context: dict[str, torch.Tensor],
        action_mask: torch.Tensor,
        deterministic: bool = True,
        temperature: float = 1.0,
    ) -> tuple[int, float]:
        """
        バトル行動を予測

        Args:
            context: トークン化されたコンテキスト
            action_mask: [num_actions] 利用可能な行動のマスク
            deterministic: True なら決定的、False ならサンプリング
            temperature: サンプリング時の温度

        Returns:
            (action_id, win_probability)
        """
        self.eval()
        device = next(self.parameters()).device

        with torch.no_grad():
            # バッチ次元を追加してデバイスに移動
            input_ids = context["input_ids"].unsqueeze(0).to(device)
            position_ids = context["position_ids"].unsqueeze(0).to(device)
            timestep_ids = context["timestep_ids"].unsqueeze(0).to(device)
            segment_ids = context["segment_ids"].unsqueeze(0).to(device)
            rtg_values = context["rtg_values"].unsqueeze(0).to(device)
            attention_mask = context["attention_mask"].unsqueeze(0).to(device)
            action_mask_batch = action_mask.unsqueeze(0).to(device)

            state_features = None
            if "state_features" in context:
                state_features = context["state_features"].unsqueeze(0).to(device)

            # Forward
            outputs = self.forward(
                input_ids=input_ids,
                position_ids=position_ids,
                timestep_ids=timestep_ids,
                segment_ids=segment_ids,
                rtg_values=rtg_values,
                attention_mask=attention_mask,
                state_features=state_features,
                action_mask=action_mask_batch,
                use_causal_mask=True,
            )

            # 行動を予測
            action_logits = outputs["action_logits"].squeeze(0)  # [num_actions]

            if deterministic:
                action_id = action_logits.argmax().item()
            else:
                probs = F.softmax(action_logits / temperature, dim=-1)
                action_id = torch.multinomial(probs, 1).item()

            # 勝率
            win_prob = outputs["value"].squeeze().item()

            return action_id, win_prob

    def get_policy_and_value(
        self,
        context: dict[str, torch.Tensor],
        action_mask: torch.Tensor,
    ) -> tuple[dict[int, float], float]:
        """
        行動確率分布と勝率を取得

        Args:
            context: トークン化されたコンテキスト
            action_mask: [num_actions] 利用可能な行動のマスク

        Returns:
            (policy_dict, win_probability)
        """
        self.eval()
        device = next(self.parameters()).device

        with torch.no_grad():
            # バッチ次元を追加してデバイスに移動
            input_ids = context["input_ids"].unsqueeze(0).to(device)
            position_ids = context["position_ids"].unsqueeze(0).to(device)
            timestep_ids = context["timestep_ids"].unsqueeze(0).to(device)
            segment_ids = context["segment_ids"].unsqueeze(0).to(device)
            rtg_values = context["rtg_values"].unsqueeze(0).to(device)
            attention_mask = context["attention_mask"].unsqueeze(0).to(device)
            action_mask_batch = action_mask.unsqueeze(0).to(device)

            state_features = None
            if "state_features" in context:
                state_features = context["state_features"].unsqueeze(0).to(device)

            # Forward
            outputs = self.forward(
                input_ids=input_ids,
                position_ids=position_ids,
                timestep_ids=timestep_ids,
                segment_ids=segment_ids,
                rtg_values=rtg_values,
                attention_mask=attention_mask,
                state_features=state_features,
                action_mask=action_mask_batch,
                use_causal_mask=True,
            )

            # Policy
            action_logits = outputs["action_logits"].squeeze(0)
            probs = F.softmax(action_logits, dim=-1)

            policy = {}
            for i, p in enumerate(probs.cpu().tolist()):
                if action_mask[i] > 0:
                    policy[i] = p

            # 正規化
            total = sum(policy.values())
            if total > 0:
                policy = {k: v / total for k, v in policy.items()}

            # Value
            win_prob = outputs["value"].squeeze().item()

            return policy, win_prob

    def _get_hidden_for_selection(self, outputs: dict) -> torch.Tensor:
        """選出予測用の hidden states を取得（フォールバック）"""
        # selection_logits から逆算することはできないので、
        # 再度 forward が必要。ここではダミーを返す。
        raise NotImplementedError("Use return_hidden_states=True in forward()")


def load_model(
    checkpoint_path: str,
    config: PokemonBattleTransformerConfig | None = None,
    device: str = "cpu",
) -> PokemonBattleTransformer:
    """
    チェックポイントからモデルをロード

    Args:
        checkpoint_path: チェックポイントディレクトリのパス
        config: モデル設定（None なら meta.json から読み込み）
        device: デバイス

    Returns:
        ロードされたモデル
    """
    import json
    from pathlib import Path

    checkpoint_path = Path(checkpoint_path)

    # 設定をロード
    if config is None:
        meta_path = checkpoint_path / "meta.json"
        if meta_path.exists():
            with open(meta_path, encoding="utf-8") as f:
                meta = json.load(f)
            config_dict = meta.get("config", {}).get("model_config", {})
            config = PokemonBattleTransformerConfig(**config_dict)
        else:
            config = PokemonBattleTransformerConfig()

    # モデルを作成
    model = PokemonBattleTransformer(config)

    # 重みをロード
    model_path = checkpoint_path / "model.pt"
    if model_path.exists():
        model.load_state_dict(
            torch.load(model_path, map_location=device, weights_only=True)
        )

    model.to(device)
    model.eval()

    return model
