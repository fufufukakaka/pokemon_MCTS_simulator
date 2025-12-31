"""
Pokemon Battle Transformer 設定

Decision Transformer ベースの Pokemon バトル AI のモデル設定。
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PokemonBattleTransformerConfig:
    """Pokemon Battle Transformer のモデル設定"""

    # === 語彙サイズ ===
    # PokemonVocab (selection_bert) を再利用: 5 special + 829 Pokemon = 834
    pokemon_vocab_size: int = 900  # 余裕を持たせる
    move_vocab_size: int = 800  # 技の種類
    item_vocab_size: int = 400  # 持ち物の種類
    ability_vocab_size: int = 300  # 特性の種類
    type_vocab_size: int = 20  # タイプ数 (18 + ステラ + パディング)

    # === 埋め込み次元 ===
    hidden_size: int = 256  # Transformer の隠れ層次元
    pokemon_embed_dim: int = 64  # Pokemon 名の埋め込み次元
    move_embed_dim: int = 32  # 技の埋め込み次元
    item_embed_dim: int = 32  # 持ち物の埋め込み次元
    ability_embed_dim: int = 32  # 特性の埋め込み次元
    type_embed_dim: int = 16  # タイプの埋め込み次元

    # === Transformer アーキテクチャ ===
    num_hidden_layers: int = 6  # Transformer レイヤー数
    num_attention_heads: int = 8  # Attention ヘッド数
    intermediate_size: int = 512  # FFN の中間層次元
    hidden_dropout_prob: float = 0.1  # Dropout 率
    attention_probs_dropout_prob: float = 0.1  # Attention Dropout 率
    layer_norm_eps: float = 1e-6  # LayerNorm の epsilon

    # === 系列設定 ===
    max_sequence_length: int = 512  # 最大系列長
    max_turns: int = 100  # 最大ターン数
    max_team_size: int = 6  # チームのポケモン数
    max_selection_size: int = 3  # 選出するポケモン数

    # === 特殊トークン ID ===
    # PokemonVocab と同じ順序
    pad_token_id: int = 0  # [PAD]
    mask_token_id: int = 1  # [MASK]
    cls_token_id: int = 2  # [CLS]
    sep_token_id: int = 3  # [SEP]
    unk_token_id: int = 4  # [UNK]

    # 追加の特殊トークン（Pokemon語彙の後に追加）
    # これらは pokemon_vocab_size の範囲外に配置
    rtg_token_id: int = 900  # Return-to-go トークン
    preview_token_id: int = 901  # [PREVIEW] チームプレビュー開始
    select_token_id: int = 902  # [SELECT] 選出フェーズ
    battle_token_id: int = 903  # [BATTLE] バトルフェーズ
    turn_token_id: int = 904  # [TURN] ターン開始
    state_token_id: int = 905  # [STATE] 状態記述
    action_token_id: int = 906  # [ACTION] 行動記述
    my_team_token_id: int = 907  # [MY_TEAM] 自チーム
    opp_team_token_id: int = 908  # [OPP_TEAM] 相手チーム

    # 総語彙サイズ（Pokemon + 追加特殊トークン）
    total_vocab_size: int = 920

    # === 出力設定 ===
    # 行動コマンド: MOVE 0-3, TERA+MOVE 10-13, SWITCH 20-25, STRUGGLE 30, NO_COMMAND 40
    num_action_outputs: int = 41  # 行動の種類数 (0-40)
    num_selection_labels: int = 3  # 選出ラベル (NOT_SELECTED=0, SELECTED=1, LEAD=2)

    # === 状態特徴量の次元 ===
    # HP ratio (1) + ailment one-hot (7) + rank (8) + terastal (1) + tera_type (1)
    pokemon_state_dim: int = 18

    # フィールド状態: weather(4) + terrain(4) + misc(2) + screens(4) + tailwind(2) + hazards(8)
    field_state_dim: int = 24

    # === セグメント ID ===
    segment_preview: int = 0
    segment_selection: int = 1
    segment_battle: int = 2

    # === 学習関連 ===
    initializer_range: float = 0.02  # 重み初期化の範囲

    def __post_init__(self):
        """検証"""
        assert self.hidden_size % self.num_attention_heads == 0, (
            f"hidden_size ({self.hidden_size}) must be divisible by "
            f"num_attention_heads ({self.num_attention_heads})"
        )


@dataclass
class TrainingConfig:
    """学習設定"""

    # === 基本設定 ===
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    num_epochs: int = 100
    warmup_steps: int = 1000
    max_grad_norm: float = 1.0

    # === Loss 重み ===
    selection_loss_weight: float = 1.0
    action_loss_weight: float = 1.0
    value_loss_weight: float = 0.5

    # === 自己対戦設定 ===
    num_iterations: int = 50  # 自己対戦イテレーション数
    games_per_iteration: int = 100  # 各イテレーションでのゲーム数
    trajectory_pool_size: int = 100000  # データプールサイズ
    training_epochs_per_iteration: int = 10  # イテレーションごとの学習エポック数

    # === 探索設定 ===
    epsilon_start: float = 0.3  # 初期 epsilon (ε-greedy)
    epsilon_end: float = 0.05  # 最終 epsilon
    epsilon_decay_iterations: int = 30  # epsilon 減衰イテレーション数
    temperature_start: float = 1.0  # 初期温度
    temperature_end: float = 0.5  # 最終温度

    # === 評価設定 ===
    eval_interval: int = 5  # 評価間隔 (イテレーション数)
    eval_games: int = 50  # 評価ゲーム数
    win_rate_threshold: float = 0.55  # モデル更新の閾値

    # === チェックポイント ===
    save_interval: int = 5  # 保存間隔 (イテレーション数)
    keep_last_n_checkpoints: int = 5  # 保持するチェックポイント数

    # === デバイス ===
    device: str = "cpu"
    num_workers: int = 1  # 並列ワーカー数

    # === モデル設定 ===
    model_config: Optional[PokemonBattleTransformerConfig] = None

    def __post_init__(self):
        if self.model_config is None:
            self.model_config = PokemonBattleTransformerConfig()

    def get_epsilon(self, iteration: int) -> float:
        """現在のイテレーションでの epsilon を計算"""
        if iteration >= self.epsilon_decay_iterations:
            return self.epsilon_end
        progress = iteration / self.epsilon_decay_iterations
        return self.epsilon_start + (self.epsilon_end - self.epsilon_start) * progress

    def get_temperature(self, iteration: int) -> float:
        """現在のイテレーションでの温度を計算"""
        if iteration >= self.epsilon_decay_iterations:
            return self.temperature_end
        progress = iteration / self.epsilon_decay_iterations
        return self.temperature_start + (self.temperature_end - self.temperature_start) * progress
