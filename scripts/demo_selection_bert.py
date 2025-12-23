"""Selection BERT の選出予測デモ"""

import json
from pathlib import Path

import torch

from src.pokemon_battle_sim.pokemon import Pokemon
from src.selection_bert.dataset import PokemonVocab
from src.selection_bert.model import PokemonBertConfig, PokemonBertForTokenClassification
from src.selection_bert.selection_belief import SelectionBeliefPredictor

# ポケモンデータ初期化
Pokemon.init()


def load_selection_bert(model_dir: Path):
    """Selection BERT モデルを読み込む"""
    vocab = PokemonVocab.from_zukan(Path("data/zukan.txt"))

    # 設定を読み込み
    config = PokemonBertConfig(
        vocab_size=len(vocab),
        hidden_size=256,
        num_hidden_layers=4,
        num_attention_heads=4,
        intermediate_size=512,
    )

    model = PokemonBertForTokenClassification(config)
    model_path = model_dir / "selection_bert.pt"
    if model_path.exists():
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    return SelectionBeliefPredictor(model, vocab, device="cpu")


def demo_selection(predictor: SelectionBeliefPredictor, my_team: list[str], opp_team: list[str]):
    """選出予測のデモ"""
    print("=" * 60)
    print("【自分のパーティ】")
    for i, name in enumerate(my_team):
        print(f"  {i+1}. {name}")
    print()
    print("【相手のパーティ】")
    for i, name in enumerate(opp_team):
        print(f"  {i+1}. {name}")
    print()

    # 予測
    my_pred, opp_pred = predictor.predict(my_team, opp_team)

    # 自分の推奨選出
    print("【Selection BERT の推奨選出（自分）】")
    selected, lead_idx = predictor.select_team(my_team, opp_team, deterministic=True)
    selected_names = [my_team[i] for i in selected]
    print(f"  選出: {', '.join(selected_names)}")
    print(f"  先発: {my_team[lead_idx]}")
    print()

    # 相手の選出予測
    print("【相手の選出予測】")
    print("  選出確率:")
    for i, name in enumerate(opp_team):
        sel_prob = opp_pred.selection_probs[i]
        lead_prob = opp_pred.lead_probs[i]
        bar = "█" * int(sel_prob * 20)
        print(f"    {name:12s}: {sel_prob:5.1%} {bar}")
    print()
    print("  先発確率:")
    for i, name in enumerate(opp_team):
        lead_prob = opp_pred.lead_probs[i]
        bar = "█" * int(lead_prob * 20)
        print(f"    {name:12s}: {lead_prob:5.1%} {bar}")
    print()

    # Top-3 選出パターン
    print("  予測される選出パターン（上位3つ）:")
    top_selections = opp_pred.top_k_selections(3)
    for rank, (indices, prob) in enumerate(top_selections, 1):
        names = [opp_team[i] for i in indices]
        print(f"    {rank}. {', '.join(names)} ({prob:.1%})")
    print()


def main():
    # モデル読み込み
    model_dir = Path("models/revel_full_state_selection_BERT_move_effective/final")
    if not model_dir.exists():
        model_dir = Path("models/revel_full_state_selection_BERT_move_effective/checkpoint_iter100")

    print(f"Loading model from: {model_dir}")
    predictor = load_selection_bert(model_dir)

    # トレーナーデータから実際のパーティを取得
    trainer_path = Path("data/top_rankers/season_36.json")
    with open(trainer_path) as f:
        trainers = json.load(f)

    # 例1: 典型的なマッチアップ
    # スタンダードな構築同士
    my_team_1 = ["ハバタクカミ", "パオジアン", "カイリュー", "ランドロス(れいじゅう)", "サーフゴー", "ウーラオス(れんげき)"]
    opp_team_1 = ["ガチグマ(アカツキ)", "オーガポン(いど)", "ドヒドイデ", "キョジオーン", "サンダー", "ヒードラン"]

    demo_selection(predictor, my_team_1, opp_team_1)

    # 例2: 受け構築への攻め
    my_team_2 = ["パオジアン", "テツノツツミ", "イーユイ", "ハバタクカミ", "カイリュー", "ランドロス(れいじゅう)"]
    opp_team_2 = ["ドヒドイデ", "ラッキー", "キョジオーン", "ヘイラッシャ", "ドオー", "アーマーガア"]

    demo_selection(predictor, my_team_2, opp_team_2)

    # 例3: 実際のトレーナーデータから
    if len(trainers) >= 2:
        t1 = trainers[0]
        t2 = trainers[1]
        my_team_3 = [p["name"] for p in t1["pokemon"]]
        opp_team_3 = [p["name"] for p in t2["pokemon"]]
        print("\n【実際のトレーナーデータからの例】")
        demo_selection(predictor, my_team_3, opp_team_3)


if __name__ == "__main__":
    main()
