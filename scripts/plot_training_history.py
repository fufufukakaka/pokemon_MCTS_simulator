"""学習履歴をプロットするスクリプト"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "Hiragino Sans"


def main():
    # データ読み込み
    history_path = Path(
        "models/revel_full_state_selection_BERT_move_effective/training_history.json"
    )
    with open(history_path) as f:
        history = json.load(f)

    iterations = [h["iteration"] for h in history]
    value_loss = [h["avg_loss"] for h in history]
    selection_bert_loss = [h["selection_bert_loss"] for h in history]

    # 図1: Value Network と Selection BERT のロスを並べて表示
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Value Network Loss
    ax1 = axes[0]
    ax1.plot(iterations, value_loss, "b-", linewidth=1.5, alpha=0.7)

    # 移動平均を追加
    window = 10
    value_loss_ma = np.convolve(value_loss, np.ones(window) / window, mode="valid")
    iterations_ma = iterations[window - 1 :]
    ax1.plot(
        iterations_ma, value_loss_ma, "b-", linewidth=2.5, label=f"移動平均 (window={window})"
    )

    ax1.set_xlabel("Iteration", fontsize=12)
    ax1.set_ylabel("Loss", fontsize=12)
    ax1.set_title("Value Network Loss", fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_xlim(1, 100)

    # Selection BERT Loss
    ax2 = axes[1]
    ax2.plot(iterations, selection_bert_loss, "g-", linewidth=1.5, alpha=0.7)

    # 移動平均を追加
    selection_bert_loss_ma = np.convolve(
        selection_bert_loss, np.ones(window) / window, mode="valid"
    )
    ax2.plot(
        iterations_ma,
        selection_bert_loss_ma,
        "g-",
        linewidth=2.5,
        label=f"移動平均 (window={window})",
    )

    ax2.set_xlabel("Iteration", fontsize=12)
    ax2.set_ylabel("Loss", fontsize=12)
    ax2.set_title("Selection BERT Loss", fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_xlim(1, 100)
    ax2.set_ylim(0.4, 1.0)  # Selection BERT の変化がわかりやすいようにスケール調整

    plt.tight_layout()
    plt.savefig("docs/training_loss_curves.png", dpi=150, bbox_inches="tight")
    print("Saved: docs/training_loss_curves.png")

    # 図2: Self-Play の勝敗推移
    fig2, ax3 = plt.subplots(figsize=(12, 5))

    wins_p0 = [h["wins_p0"] for h in history]
    wins_p1 = [h["wins_p1"] for h in history]
    draws = [h["draws"] for h in history]

    ax3.bar(iterations, wins_p0, label="Player 0 勝利", alpha=0.7, color="blue")
    ax3.bar(
        iterations, wins_p1, bottom=wins_p0, label="Player 1 勝利", alpha=0.7, color="red"
    )
    ax3.bar(
        iterations,
        draws,
        bottom=[w0 + w1 for w0, w1 in zip(wins_p0, wins_p1)],
        label="引き分け",
        alpha=0.7,
        color="gray",
    )

    ax3.axhline(y=50, color="black", linestyle="--", alpha=0.5, label="50%ライン")
    ax3.set_xlabel("Iteration", fontsize=12)
    ax3.set_ylabel("試合数", fontsize=12)
    ax3.set_title("Self-Play 勝敗推移 (各100試合)", fontsize=14)
    ax3.legend(loc="upper right")
    ax3.grid(True, alpha=0.3, axis="y")
    ax3.set_xlim(0, 101)
    ax3.set_ylim(0, 105)

    plt.tight_layout()
    plt.savefig("docs/selfplay_winrate.png", dpi=150, bbox_inches="tight")
    print("Saved: docs/selfplay_winrate.png")

    plt.close("all")


if __name__ == "__main__":
    main()
