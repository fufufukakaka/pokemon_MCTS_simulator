# Decision Transformer vs ReBeL 対決スクリプト

2つのAIモデル（Decision Transformer と ReBeL+Selection BERT）を対戦させて勝率を比較するスクリプト。

## 基本的な使い方

```bash
# 10試合（テスト用）
uv run python scripts/compare_dt_vs_rebel.py \
    --dt-checkpoint models/decision_transformer_full \
    --rebel-checkpoint models/revel_full_state_selection_BERT_move_effective/checkpoint_iter100 \
    --num-matches 10

# 1000試合（本番評価）
uv run python scripts/compare_dt_vs_rebel.py \
    --dt-checkpoint models/decision_transformer_full \
    --rebel-checkpoint models/revel_full_state_selection_BERT_move_effective/checkpoint_iter100 \
    --num-matches 1000 \
    --output results/dt_vs_rebel_1000
```

## オプション一覧

| オプション | 説明 | デフォルト |
|-----------|------|-----------|
| `--dt-checkpoint` | Decision Transformerのチェックポイントパス | **必須** |
| `--rebel-checkpoint` | ReBeLのチェックポイントパス | **必須** |
| `--num-matches` | 対戦数 | 10 |
| `--trainer-jsons` | パーティプールのJSONファイル（複数可） | season_35, season_36, my_fixed_party |
| `--output` | 出力ディレクトリ | `results/dt_vs_rebel` |
| `--usage-db` | ReBeL用の使用率データベース | `data/pokedb_usage/season_37_top150.json` |
| `--max-turns` | 最大ターン数 | 100 |
| `--dt-use-mcts` | DTでMCTS先読みを有効化 | False |
| `--dt-mcts-simulations` | MCTSシミュレーション回数 | 200 |
| `--device` | デバイス（cpu/cuda） | cpu |

## 出力ファイル

```
results/dt_vs_rebel/
├── stats.json              # 統計サマリー
├── results.jsonl           # 各試合の結果（1行1試合）
└── battle_logs/
    ├── match_0001.json     # 試合1の詳細ログ
    ├── match_0002.json     # 試合2の詳細ログ
    └── ...
```

### stats.json の内容

```json
{
  "total_matches": 100,
  "dt_wins": 35,
  "dt_win_rate": 0.35,
  "rebel_wins": 62,
  "rebel_win_rate": 0.62,
  "draws": 3,
  "draw_rate": 0.03,
  "dt_surrenders": 5,
  "rebel_surrenders": 2,
  "avg_turns": 18.5,
  "avg_dt_remaining": 0.8,
  "avg_rebel_remaining": 1.2,
  "avg_dt_think_time_sec": 0.25,
  "avg_rebel_think_time_sec": 5.5
}
```

### バトルログ形式（リプレイ用）

各試合のログは `src/battle_ui` のエクスポート形式と互換性があります：

```json
{
  "match_id": 1,
  "timestamp": "2026-01-28T12:00:00",
  "result": {
    "winner": "rebel",
    "total_turns": 15,
    "dt_remaining": 0,
    "rebel_remaining": 2
  },
  "teams": {
    "dt": {
      "name": "トレーナー名",
      "selection": [0, 2, 4],
      "pokemon": [...]
    },
    "rebel": {
      "name": "トレーナー名",
      "selection": [1, 3, 5],
      "pokemon": [...]
    }
  },
  "final_state": {...},
  "battle_log": [
    {"turn": 1, "player": "dt", "message": "ピカチュウの10まんボルト！"},
    ...
  ]
}
```

## 使用例

### カスタムパーティプールを使用

```bash
uv run python scripts/compare_dt_vs_rebel.py \
    --dt-checkpoint models/decision_transformer_full \
    --rebel-checkpoint models/revel_full_state_selection_BERT_move_effective/checkpoint_iter100 \
    --trainer-jsons data/top_rankers/season_36.json data/my_fixed_party.json \
    --num-matches 50
```

### GPU使用 + MCTS有効

```bash
uv run python scripts/compare_dt_vs_rebel.py \
    --dt-checkpoint models/decision_transformer_full/checkpoint_iter50 \
    --rebel-checkpoint models/revel_full_state_selection_BERT_move_effective/checkpoint_iter100 \
    --num-matches 100 \
    --device mps \
    --dt-use-mcts \
    --dt-mcts-simulations 100
```

## 対戦の仕組み

1. **パーティ選択**: 各試合で両モデルがパーティプールからランダムに1つずつ選択（重複可）
2. **選出フェーズ**: 各モデルが相手のパーティを見て3匹を選出
3. **バトルフェーズ**: ターンごとにコマンドを取得して `battle.proceed()` で実行
4. **観測更新**: ReBeLは相手の技・持ち物・特性・テラスタイプを観測して信念状態を更新
5. **終了判定**: 勝敗が決まるか最大ターン数に達したら終了

## 注意事項

- ReBeL は CFR ソルバーを使うため、DT より思考時間が長い（約10-20倍）
- 1000試合の場合、CPU で数時間かかる
- 中間結果は10試合ごとに保存されるため、途中で中断しても結果は残る
