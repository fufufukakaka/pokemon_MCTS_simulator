# pokemon_MCTS_simulator

## resume option

```bash
DISCORD_WEBHOOK_URL={} \
POSTGRES_DB={} \
POSTGRES_PASSWORD={} \
POSTGRES_USER={} \
POSTGRES_HOST={} \
POSTGRES_PORT={} \
poetry run python scripts/matching.py --resume
```

## extract data from simulation

```bash
poetry run python src/utils/extract_battle_history.py
```

## train fasttext

サブワードで文字の類似性を見てくるのがちょっと嫌かも。word2vec にしようかな。

```bash
poetry run python src/utils/train_fasttext.py
```
