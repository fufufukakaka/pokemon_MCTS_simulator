#!/usr/bin/env python3
"""
ReBeL 強化学習トレーニングスクリプト

自己対戦でデータを生成し、Value Network を学習する。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.hypothesis.pokemon_usage_database import PokemonUsageDatabase
from src.rebel import (
    ReBeLTrainer,
    TrainingConfig,
    ReBeLValueNetwork,
)


def main():
    parser = argparse.ArgumentParser(description="ReBeL トレーニング")
    parser.add_argument(
        "--trainer-json",
        type=str,
        default="data/top_rankers/season_27.json",
        help="トレーナーデータのパス",
    )
    parser.add_argument(
        "--usage-db",
        type=str,
        default="data/pokedb_usage/season_37_top150.json",
        help="使用率データベースのパス",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="models/rebel",
        help="出力ディレクトリ",
    )
    parser.add_argument(
        "--num-iterations",
        type=int,
        default=10,
        help="学習イテレーション数",
    )
    parser.add_argument(
        "--games-per-iteration",
        type=int,
        default=20,
        help="各イテレーションでの自己対戦数",
    )
    parser.add_argument(
        "--cfr-iterations",
        type=int,
        default=30,
        help="CFR イテレーション数",
    )
    parser.add_argument(
        "--cfr-world-samples",
        type=int,
        default=10,
        help="CFR ワールドサンプル数",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="学習バッチサイズ",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="学習率",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=5,
        help="各イテレーションでのエポック数",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=256,
        help="ネットワークの隠れ層次元",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="デバイス (cpu/cuda)",
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        default=5,
        help="チェックポイント保存間隔（イテレーション数）",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="学習を再開するチェックポイントのパス",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="学習後に評価を実行",
    )
    parser.add_argument(
        "--evaluate-only",
        action="store_true",
        help="学習をスキップして評価のみ実行（--resumeでチェックポイント指定必須）",
    )
    parser.add_argument(
        "--eval-games",
        type=int,
        default=30,
        help="評価試合数",
    )
    parser.add_argument(
        "--fixed-opponent",
        type=str,
        default=None,
        help="固定対戦相手のJSONファイルパス（単一トレーナー）",
    )
    parser.add_argument(
        "--fixed-opponent-index",
        type=int,
        default=None,
        help="trainer-json内の固定対戦相手のインデックス",
    )
    parser.add_argument(
        "--fixed-opponent-select-all",
        action="store_true",
        help="固定対戦相手はランダム選出ではなく先頭3体を使用",
    )
    parser.add_argument(
        "--train-selection",
        action="store_true",
        help="選出ネットワークも同時に学習する",
    )
    parser.add_argument(
        "--selection-explore-prob",
        type=float,
        default=0.3,
        help="選出時の探索確率（ランダム選出する確率）",
    )
    parser.add_argument(
        "--usage-data-path",
        type=str,
        default=None,
        help="ポケモン統計データのパス（新形式のpokedb JSONにも対応）",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="並列ゲーム生成のワーカー数（1=逐次実行、CPUコア数以下を推奨）",
    )
    parser.add_argument(
        "--lightweight-cfr",
        action="store_true",
        default=True,
        help="軽量CFRモードを使用（デフォルト：有効）",
    )
    parser.add_argument(
        "--no-lightweight-cfr",
        action="store_true",
        help="軽量CFRモードを無効化（精度向上、速度低下）",
    )
    parser.add_argument(
        "--use-full-belief",
        action="store_true",
        help="完全信念状態を使用（選出・先発の不確実性を含む）",
    )
    # Selection BERT options
    parser.add_argument(
        "--use-selection-bert",
        action="store_true",
        help="Selection BERTを使用して選出予測を学習",
    )
    parser.add_argument(
        "--selection-bert-pretrained",
        type=str,
        default=None,
        help="事前学習済みSelection BERTモデルのパス",
    )
    parser.add_argument(
        "--selection-bert-hidden-size",
        type=int,
        default=256,
        help="Selection BERTの隠れ層次元",
    )
    parser.add_argument(
        "--selection-bert-num-layers",
        type=int,
        default=4,
        help="Selection BERTのレイヤー数",
    )
    parser.add_argument(
        "--selection-bert-num-heads",
        type=int,
        default=4,
        help="Selection BERTのアテンションヘッド数",
    )
    parser.add_argument(
        "--selection-bert-epochs",
        type=int,
        default=5,
        help="各イテレーションでのSelection BERTエポック数",
    )
    args = parser.parse_args()

    # データ読み込み
    print("Loading data...")
    with open(args.trainer_json, "r", encoding="utf-8") as f:
        trainer_data = json.load(f)

    usage_db = PokemonUsageDatabase.from_json(args.usage_db)

    print(f"Loaded {len(trainer_data)} trainers")
    print(f"Usage DB: {usage_db}")

    # 固定対戦相手の設定
    fixed_opponent = None
    if args.fixed_opponent:
        with open(args.fixed_opponent, "r", encoding="utf-8") as f:
            fixed_opponent = json.load(f)
        print(f"Fixed opponent loaded from: {args.fixed_opponent}")
    elif args.fixed_opponent_index is not None:
        if 0 <= args.fixed_opponent_index < len(trainer_data):
            fixed_opponent = trainer_data[args.fixed_opponent_index]
            print(f"Fixed opponent: index {args.fixed_opponent_index}")
            if "pokemons" in fixed_opponent:
                pokemon_names = [p.get("name", "?") for p in fixed_opponent["pokemons"][:6]]
                print(f"  Team: {', '.join(pokemon_names)}")
        else:
            print(f"Warning: Invalid fixed_opponent_index {args.fixed_opponent_index}, ignoring")

    # 軽量CFRの設定（--no-lightweight-cfrで無効化）
    use_lightweight_cfr = not args.no_lightweight_cfr

    # usage_data_path の決定（--usage-data-path が未指定なら --usage-db を使用）
    usage_data_path = args.usage_data_path if args.usage_data_path else args.usage_db

    # 設定
    config = TrainingConfig(
        games_per_iteration=args.games_per_iteration,
        cfr_iterations=args.cfr_iterations,
        cfr_world_samples=args.cfr_world_samples,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        device=args.device,
        save_interval=args.save_interval,
        fixed_opponent=fixed_opponent,
        fixed_opponent_select_all=args.fixed_opponent_select_all,
        train_selection=args.train_selection,
        selection_explore_prob=args.selection_explore_prob,
        usage_data_path=usage_data_path,
        num_workers=args.num_workers,
        use_lightweight_cfr=use_lightweight_cfr,
        use_full_belief=args.use_full_belief,
        # Selection BERT settings
        use_selection_bert=args.use_selection_bert,
        selection_bert_pretrained=args.selection_bert_pretrained,
        selection_bert_hidden_size=args.selection_bert_hidden_size,
        selection_bert_num_layers=args.selection_bert_num_layers,
        selection_bert_num_heads=args.selection_bert_num_heads,
        selection_bert_epochs_per_iter=args.selection_bert_epochs,
    )

    if args.train_selection:
        print("Selection network training: ENABLED")
    if args.use_full_belief:
        print("Full belief state: ENABLED")
    if args.num_workers > 1:
        print(f"Parallel game generation: {args.num_workers} workers")
    print(f"Lightweight CFR: {'ENABLED' if use_lightweight_cfr else 'DISABLED'}")
    if args.use_selection_bert:
        print("Selection BERT: ENABLED")
        if args.selection_bert_pretrained:
            print(f"  Pretrained model: {args.selection_bert_pretrained}")
        print(f"  Hidden size: {args.selection_bert_hidden_size}")
        print(f"  Layers: {args.selection_bert_num_layers}")
        print(f"  Heads: {args.selection_bert_num_heads}")
        print(f"  Epochs per iteration: {args.selection_bert_epochs}")

    # Value Network
    value_network = ReBeLValueNetwork(
        hidden_dim=args.hidden_dim,
        num_res_blocks=4,
        dropout=0.1,
    )

    # トレーナー
    trainer = ReBeLTrainer(
        usage_db=usage_db,
        trainer_data=trainer_data,
        config=config,
        value_network=value_network,
    )

    # 評価のみモードのチェック
    if args.evaluate_only:
        if not args.resume:
            print("Error: --evaluate-only requires --resume to specify checkpoint path")
            return

        checkpoint_path = Path(args.resume)

        # チェックポイントから設定を自動検出
        print(f"Evaluate-only mode: detecting settings from {args.resume}")

        # checkpoint_meta.json から設定を読み込み
        checkpoint_meta_path = checkpoint_path / "checkpoint_meta.json"
        if checkpoint_meta_path.exists():
            with open(checkpoint_meta_path, "r", encoding="utf-8") as f:
                checkpoint_meta = json.load(f)
            saved_config = checkpoint_meta.get("config", {})

            # use_full_belief の自動検出
            if saved_config.get("use_full_belief", False) and not args.use_full_belief:
                print("  -> use_full_belief detected, enabling automatically")
                config.use_full_belief = True

            # use_selection_bert の自動検出
            if saved_config.get("use_selection_bert", False) and not args.use_selection_bert:
                print("  -> use_selection_bert detected, enabling automatically")
                config.use_selection_bert = True
                config.train_selection = True

            # train_selection の自動検出
            if saved_config.get("train_selection", False) and not args.train_selection:
                print("  -> train_selection detected, enabling automatically")
                config.train_selection = True

        # ファイル存在による自動検出（checkpoint_meta.json がない古いチェックポイント用）
        # Selection BERT の自動検出
        selection_bert_path = checkpoint_path / "selection_bert.pt"
        selection_bert_vocab_path = checkpoint_path / "selection_bert_vocab.json"
        if selection_bert_path.exists() and selection_bert_vocab_path.exists():
            if not config.use_selection_bert:
                print("  -> Selection BERT files detected, enabling automatically")
                config.train_selection = True
                config.use_selection_bert = True

        # TeamSelectionNetwork の自動検出
        selection_network_path = checkpoint_path / "selection_network.pt"
        if selection_network_path.exists() and not config.use_selection_bert:
            if not config.train_selection:
                print("  -> TeamSelectionNetwork detected, enabling automatically")
                config.train_selection = True

        # 設定が変更された場合は trainer を再作成
        trainer = ReBeLTrainer(
            usage_db=usage_db,
            trainer_data=trainer_data,
            config=config,
            value_network=value_network,
        )

        print(f"Loading checkpoint from {args.resume}")
        trainer.load(checkpoint_path)
        print(f"Model loaded from iteration {trainer.current_iteration}")
    else:
        # 再開
        if args.resume:
            print(f"Resuming from {args.resume}")
            trainer.load(Path(args.resume))
            print(f"Will run {args.num_iterations} more iterations (from iter {trainer.current_iteration + 1} to {trainer.current_iteration + args.num_iterations})")

        # 学習
        print(f"\nStarting training for {args.num_iterations} iterations...")
        print(f"Config: {config}")
        print("=" * 60)

        trainer.train(args.num_iterations, args.output)

        print("\nTraining completed!")
        print(f"Model saved to {args.output}")

    # 評価
    if args.evaluate or args.evaluate_only:
        print("\n" + "=" * 60)
        print("Running evaluation...")

        eval_results = {}

        # 選出ネットワークが学習されているか確認
        has_selection_model = (
            trainer.selection_bert_predictor is not None
            or trainer.selection_network is not None
        )

        # 評価パターン
        # ReBeL側は常に学習済み選出を使用（利用可能な場合）
        rebel_sel = "learned" if has_selection_model else "random"

        # 1. vs Random action + Random selection
        print("\n--- vs Random (random selection) ---")
        eval_results["vs_random_randsel"] = trainer.evaluate_against_baseline(
            num_games=args.eval_games,
            baseline_type="random",
            rebel_selection=rebel_sel,
            baseline_selection="random",
        )

        # 2. vs CFR-only + Random selection
        print("\n--- vs CFR-only (random selection) ---")
        eval_results["vs_cfr_randsel"] = trainer.evaluate_against_baseline(
            num_games=args.eval_games,
            baseline_type="cfr_only",
            rebel_selection=rebel_sel,
            baseline_selection="random",
        )

        # 選出ネットワークがある場合は、相手も学習済み選出を使う評価を追加
        if has_selection_model:
            # 3. vs Random action + Learned selection
            print("\n--- vs Random (learned selection) ---")
            eval_results["vs_random_learnsel"] = trainer.evaluate_against_baseline(
                num_games=args.eval_games,
                baseline_type="random",
                rebel_selection=rebel_sel,
                baseline_selection="learned",
            )

            # 4. vs CFR-only + Learned selection
            print("\n--- vs CFR-only (learned selection) ---")
            eval_results["vs_cfr_learnsel"] = trainer.evaluate_against_baseline(
                num_games=args.eval_games,
                baseline_type="cfr_only",
                rebel_selection=rebel_sel,
                baseline_selection="learned",
            )

        # 結果を保存
        # evaluate-only の場合はチェックポイント名をファイル名に含める
        if args.evaluate_only and args.resume:
            checkpoint_name = Path(args.resume).name
            output_filename = f"evaluation_results_{checkpoint_name}.json"
            output_path = Path(args.resume).parent / output_filename
        else:
            output_filename = "evaluation_results.json"
            output_path = Path(args.output) / output_filename

        with open(output_path, "w") as f:
            json.dump(eval_results, f, indent=2)

        # サマリー表示
        print("\n" + "=" * 60)
        print("Evaluation Summary:")
        print("-" * 60)
        for key, res in eval_results.items():
            win_rate = res.get("win_rate", 0) * 100
            print(f"  {key}: {win_rate:.1f}% win rate")

        print(f"\nEvaluation results saved to {output_path}")


if __name__ == "__main__":
    main()
