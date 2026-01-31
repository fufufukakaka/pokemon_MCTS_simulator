"""
Pokemon Battle Transformer Output Heads

選出、行動、価値の予測ヘッド。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import PokemonBattleTransformerConfig


class SelectionHead(nn.Module):
    """
    選出予測ヘッド

    チームプレビューから、6匹それぞれに対して選出ラベルを予測:
    - 0: NOT_SELECTED (選出しない)
    - 1: SELECTED (選出する)
    - 2: LEAD (先発として選出)
    """

    def __init__(self, config: PokemonBattleTransformerConfig):
        super().__init__()
        self.config = config

        # 選出分類器
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.GELU(),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(config.hidden_size // 2, config.num_selection_labels),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        team_token_positions: torch.Tensor,
    ) -> torch.Tensor:
        """
        選出ロジットを計算

        Args:
            hidden_states: [batch, seq_len, hidden_size] Transformer 出力
            team_token_positions: [batch, 6] 自チームのトークン位置

        Returns:
            [batch, 6, num_selection_labels] 選出ロジット
        """
        batch_size = hidden_states.size(0)
        device = hidden_states.device

        # 各チームメンバーの hidden state を抽出
        # team_token_positions: [batch, 6]
        team_hidden = []
        for b in range(batch_size):
            positions = team_token_positions[b]  # [6]
            pokemon_hidden = hidden_states[b, positions]  # [6, hidden_size]
            team_hidden.append(pokemon_hidden)

        team_hidden = torch.stack(team_hidden)  # [batch, 6, hidden_size]

        # 分類
        logits = self.classifier(team_hidden)  # [batch, 6, num_labels]

        return logits

    def predict(
        self,
        hidden_states: torch.Tensor,
        team_token_positions: torch.Tensor,
        deterministic: bool = True,
        temperature: float = 1.0,
    ) -> tuple[list[int], int]:
        """
        選出を予測

        Args:
            hidden_states: [1, seq_len, hidden_size]
            team_token_positions: [1, 6]
            deterministic: True なら argmax、False ならサンプリング
            temperature: サンプリング時の温度

        Returns:
            (selected_indices, lead_index): 選出した3匹のインデックスと先発インデックス
        """
        logits = self.forward(hidden_states, team_token_positions)  # [1, 6, 3]
        logits = logits.squeeze(0)  # [6, 3]

        if deterministic:
            # 各ポケモンの予測ラベル
            predicted_labels = logits.argmax(dim=-1)  # [6]

            # LEAD (2) を持つポケモンを先発に
            lead_candidates = (predicted_labels == 2).nonzero(as_tuple=True)[0]
            if len(lead_candidates) > 0:
                lead_index = lead_candidates[0].item()
            else:
                # LEAD がなければ SELECTED (1) の中から最もスコアの高いものを先発に
                selected_candidates = (predicted_labels >= 1).nonzero(as_tuple=True)[0]
                if len(selected_candidates) > 0:
                    lead_scores = logits[selected_candidates, 2]  # LEAD スコア
                    lead_index = selected_candidates[lead_scores.argmax()].item()
                else:
                    lead_index = 0

            # 選出: SELECTED または LEAD のポケモン
            selection_scores = logits[:, 1] + logits[:, 2]  # SELECTED + LEAD スコア
            _, top_indices = selection_scores.topk(3)
            selected_indices = top_indices.tolist()

            # lead_index が選出に含まれていなければ追加
            if lead_index not in selected_indices:
                selected_indices = [lead_index] + selected_indices[:2]

        else:
            # 温度付きサンプリング
            probs = F.softmax(logits / temperature, dim=-1)  # [6, 3]

            # 各ポケモンの選出確率を計算
            selection_probs = probs[:, 1] + probs[:, 2]  # SELECTED + LEAD
            selection_probs = selection_probs / selection_probs.sum()

            # 3匹をサンプリング（重複なし）
            selected_indices = torch.multinomial(selection_probs, 3, replacement=False).tolist()

            # 先発を LEAD 確率でサンプリング
            lead_probs = probs[selected_indices, 2]  # 選出された3匹の LEAD 確率
            lead_probs = lead_probs / lead_probs.sum()
            lead_local_idx = torch.multinomial(lead_probs, 1).item()
            lead_index = selected_indices[lead_local_idx]

        return selected_indices, lead_index


class ActionHead(nn.Module):
    """
    行動予測ヘッド

    バトル中の行動を予測:
    - 0-3: MOVE (技)
    - 10-13: TERA + MOVE (テラスタル + 技)
    - 20-25: SWITCH (交代)
    - 30: STRUGGLE (わるあがき)
    """

    def __init__(self, config: PokemonBattleTransformerConfig):
        super().__init__()
        self.config = config

        # 行動分類器
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.GELU(),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(config.hidden_size, config.num_action_outputs),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        action_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        行動ロジットを計算

        Args:
            hidden_states: [batch, hidden_size] または [batch, seq_len, hidden_size]
            action_mask: [batch, num_actions] 利用可能な行動のマスク (1=有効, 0=無効)

        Returns:
            [batch, num_actions] 行動ロジット
        """
        # 最後の次元が hidden_size なら OK
        if hidden_states.dim() == 3:
            # [batch, seq_len, hidden] から最後のトークンを使用
            hidden_states = hidden_states[:, -1, :]  # [batch, hidden]

        logits = self.classifier(hidden_states)  # [batch, num_actions]

        # 行動マスクを適用
        if action_mask is not None:
            logits = logits.masked_fill(action_mask == 0, float("-inf"))

        return logits

    def predict(
        self,
        hidden_states: torch.Tensor,
        action_mask: torch.Tensor,
        deterministic: bool = True,
        temperature: float = 1.0,
    ) -> int:
        """
        行動を予測

        Args:
            hidden_states: [1, hidden_size]
            action_mask: [1, num_actions]
            deterministic: True なら argmax、False ならサンプリング
            temperature: サンプリング時の温度

        Returns:
            選択した行動ID
        """
        logits = self.forward(hidden_states, action_mask)  # [1, num_actions]
        logits = logits.squeeze(0)  # [num_actions]

        if deterministic:
            return logits.argmax().item()
        else:
            # 温度付きサンプリング
            probs = F.softmax(logits / temperature, dim=-1)
            return torch.multinomial(probs, 1).item()

    def get_policy(
        self,
        hidden_states: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> dict[int, float]:
        """
        行動確率分布を取得

        Args:
            hidden_states: [1, hidden_size]
            action_mask: [1, num_actions]

        Returns:
            {action_id: probability} の辞書
        """
        logits = self.forward(hidden_states, action_mask)  # [1, num_actions]
        probs = F.softmax(logits, dim=-1).squeeze(0)  # [num_actions]

        policy = {}
        for i, p in enumerate(probs.tolist()):
            if action_mask[0, i] > 0:
                policy[i] = p

        # 正規化
        total = sum(policy.values())
        if total > 0:
            policy = {k: v / total for k, v in policy.items()}

        return policy


class ValueHead(nn.Module):
    """
    価値予測ヘッド

    現在の状態から勝率を予測 [0, 1]
    """

    def __init__(self, config: PokemonBattleTransformerConfig):
        super().__init__()
        self.config = config

        # 価値推定器
        self.estimator = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.GELU(),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(config.hidden_size // 2, 1),
            nn.Sigmoid(),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.estimator.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        勝率を予測

        Args:
            hidden_states: [batch, hidden_size] または [batch, seq_len, hidden_size]

        Returns:
            [batch, 1] 勝率
        """
        if hidden_states.dim() == 3:
            # 最後のトークンを使用
            hidden_states = hidden_states[:, -1, :]  # [batch, hidden]

        return self.estimator(hidden_states)

    def predict(self, hidden_states: torch.Tensor) -> float:
        """
        勝率を予測（スカラー値）

        Args:
            hidden_states: [1, hidden_size]

        Returns:
            勝率 (0-1)
        """
        value = self.forward(hidden_states)
        return value.squeeze().item()


class CombinedHead(nn.Module):
    """
    統合ヘッド

    選出、行動、価値を一つのモジュールで管理。
    共有層を持つことで効率化も可能。
    """

    def __init__(self, config: PokemonBattleTransformerConfig):
        super().__init__()
        self.config = config

        # 共有中間層
        self.shared = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.GELU(),
            nn.Dropout(config.hidden_dropout_prob),
        )

        # 個別ヘッド
        self.selection_head = SelectionHead(config)
        self.action_head = ActionHead(config)
        self.value_head = ValueHead(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        team_token_positions: torch.Tensor | None = None,
        action_positions: torch.Tensor | None = None,
        action_mask: torch.Tensor | None = None,
        output_type: str = "all",
    ) -> dict[str, torch.Tensor]:
        """
        全ての出力を計算

        Args:
            hidden_states: [batch, seq_len, hidden_size]
            team_token_positions: [batch, 6] 選出予測用
            action_positions: [batch, num_actions] 行動予測位置
            action_mask: [batch, num_actions] 行動マスク
            output_type: "selection", "action", "value", "all"

        Returns:
            各種出力を含む辞書
        """
        outputs = {}

        if output_type in ("selection", "all") and team_token_positions is not None:
            outputs["selection_logits"] = self.selection_head(hidden_states, team_token_positions)

        if output_type in ("action", "all"):
            # 行動位置の hidden states を取得
            if action_positions is not None:
                batch_size = hidden_states.size(0)
                action_hidden = []
                for b in range(batch_size):
                    pos = action_positions[b, -1] if action_positions[b].numel() > 0 else -1
                    action_hidden.append(hidden_states[b, pos])
                action_hidden = torch.stack(action_hidden)  # [batch, hidden]
            else:
                action_hidden = hidden_states[:, -1, :]

            outputs["action_logits"] = self.action_head(action_hidden, action_mask)

        if output_type in ("value", "all"):
            outputs["value"] = self.value_head(hidden_states[:, -1, :])

        return outputs
