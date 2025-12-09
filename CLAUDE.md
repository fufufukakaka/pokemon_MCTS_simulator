# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A Pokémon battle simulator combining Monte Carlo Tree Search (MCTS), hypothesis-based incomplete information handling, and neural network-guided decision making. Features include Gen9-compliant battle simulation, AlphaZero-style reinforcement learning, team selection networks, LLM training pipelines, and a FastAPI damage calculator.

## Core Architecture

### 1. Battle Simulation Engine (`src/pokemon_battle_sim/`)

- **[battle.py](src/pokemon_battle_sim/battle.py)**: Core battle state management with turn-based simulation

  - Command system: `SKIP=-1`, `STRUGGLE=30`, `NO_COMMAND=40`, moves `0-3`, switches `20+`
  - Deep cloning support via `seed` and `copy_count` for MCTS simulations
  - `available_commands(player, phase)`: Returns legal moves/switches for given phase ("battle" or "change")
  - `winner()`: Returns 0/1 for winner or None if ongoing

- **[pokemon.py](src/pokemon_battle_sim/pokemon.py)**: Pokémon class with comprehensive mechanics

  - Stats calculation with nature/EVs/IVs support
  - Abilities, items, moves, status effects, Terastal support
  - `Pokemon.init()`: **Must be called before instantiating any Pokémon** to load data files

- **[damage.py](src/pokemon_battle_sim/damage.py)**: Damage calculation engine with full Gen 9 mechanics

- **MCTS implementations**: Two locations exist:
  - `src/pokemon_battle_sim/mcts_battle.py`: Original MCTS battle
  - `src/mcts/mcts_battle.py`: Standalone MCTS module (imports as `src.mcts.mcts_battle`)
  - Both provide `MyMCTSBattle`, `MCTSNode`, UCT exploration with `sqrt(2)`, default 1000 iterations

### 2. Hypothesis-Based MCTS (`src/hypothesis/`)

Handles incomplete information games (opponent's hidden items) via hypothesis sampling.

- **[hypothesis_mcts.py](src/hypothesis/hypothesis_mcts.py)**: `HypothesisMCTS` and `HypothesisMCTSBattle`

  - Samples hypotheses about opponent's items and averages over outcomes
  - `PolicyValue`: Protocol for neural network guidance integration

- **[item_belief_state.py](src/hypothesis/item_belief_state.py)**: `ItemBeliefState` tracks probability distributions over opponent items

- **[item_prior_database.py](src/hypothesis/item_prior_database.py)**: `ItemPriorDatabase` provides prior probabilities for items

- **[selfplay.py](src/hypothesis/selfplay.py)**: `SelfPlayGenerator` produces training data
  - `GameRecord`, `TurnRecord`, `PokemonState`, `FieldCondition`: Data structures for recording games
  - `save_records_to_jsonl()`, `load_records_from_jsonl()`: Persistence utilities

### 3. Policy-Value Network (`src/policy_value_network/`)

AlphaZero-style neural network for action prediction and win rate estimation.

- **[network.py](src/policy_value_network/network.py)**: `PolicyValueNetwork` outputs policy logits and value estimate
- **[observation_encoder.py](src/policy_value_network/observation_encoder.py)**: `ObservationEncoder` converts battle state to tensor
- **[nn_guided_mcts.py](src/policy_value_network/nn_guided_mcts.py)**: `NNGuidedMCTS` uses neural network to guide tree search
- **[trainer.py](src/policy_value_network/trainer.py)**: `PolicyValueTrainer` with `TrainingConfig`
- **[reinforcement_loop.py](src/policy_value_network/reinforcement_loop.py)**: `ReinforcementLoop` orchestrates Self-Play → Train → Evaluate cycles
- **[evaluator.py](src/policy_value_network/evaluator.py)**: `ModelEvaluator` compares model generations

### 4. Team Selection Network (`src/policy_value_network/`)

Selects optimal 3 Pokémon from 6 based on opponent's team.

- **[team_selection_network.py](src/policy_value_network/team_selection_network.py)**: `TeamSelectionNetwork`
- **[team_selection_encoder.py](src/policy_value_network/team_selection_encoder.py)**: `TeamSelectionEncoder`
- **[team_selector.py](src/policy_value_network/team_selector.py)**: Selector implementations
  - `NNTeamSelector`: Neural network-based selection
  - `RandomTeamSelector`: Random baseline
  - `TopNTeamSelector`: First N Pokémon (legacy behavior)
  - `HybridTeamSelector`: Combines strategies
  - `load_team_selector(path, device)`: Load trained selector

### 5. ReBeL - Recursive Belief-based Learning (`src/rebel/`)

**Purpose**: Handle incomplete information more rigorously than hypothesis sampling MCTS using game-theoretic approaches.

ReBeL addresses limitations of HypothesisMCTS by:

1. Tracking beliefs over opponent's full type (moves + item + tera + nature), not just items
2. Using CFR (Counterfactual Regret Minimization) to compute Nash equilibrium strategies
3. Learning a value network on Public Belief States (PBS)

- **[belief_state.py](src/rebel/belief_state.py)**: `PokemonBeliefState` tracks probability distributions over opponent's type hypotheses

  - `PokemonTypeHypothesis`: Frozen dataclass representing (moves, item, tera_type, nature, ability)
  - Bayesian updates from observations (move used, item revealed, tera used)
  - `sample_world()`: Sample one complete "world" from belief distribution

- **[public_state.py](src/rebel/public_state.py)**: `PublicGameState` and `PublicBeliefState`

  - `PublicGameState`: Observable information only (HP ratios, revealed moves/items, field)
  - `PublicBeliefState` (PBS): Public state + belief + current strategies

- **[cfr_solver.py](src/rebel/cfr_solver.py)**: CFR subgame solving

  - `CFRSubgameSolver`: Full CFR with regret matching
  - `SimplifiedCFRSolver`: Faster approximation using maximin
  - `ReBeLSolver`: Wrapper combining value network with CFR

- **[value_network.py](src/rebel/value_network.py)**: Neural networks for PBS evaluation

  - `PBSEncoder`: Encodes PBS to fixed-length tensor
  - `ReBeLValueNetwork`: Predicts expected values from PBS
  - `ReBeLPolicyValueNetwork`: Predicts both policy and value

- **[battle_interface.py](src/rebel/battle_interface.py)**: Integration with Battle class

  - `ReBeLBattle`: Drop-in replacement for `HypothesisMCTSBattle`
  - `ReBeLMCTSAdapter`: Use ReBeL with existing HypothesisMCTS interface
  - `load_rebel_battle()`: Factory function

- **[trainer.py](src/rebel/trainer.py)**: Self-play training loop
  - `ReBeLTrainer`: Generate data via self-play, train value network

### 6. LLM Training Pipeline (`src/llm/`)

**Purpose**: Train LLMs to play Pokémon battles through supervised fine-tuning and reinforcement learning.

- **[static_dataset.py](src/llm/static_dataset.py)**: Generate static training examples from damage calculations

  - `build_example_from_battle(battle, player)`: Creates one training sample
  - Output format: `{state_text, actions, label_action_id, policy_dist}`
  - Uses damage-based scoring to label optimal moves

- **[state_representation.py](src/llm/state_representation.py)**: Convert `Battle` state to LLM-readable text

  - `battle_to_llm_state(battle, player)`: Returns `LLMState` with text representation and legal actions
  - `LLMAction`: Typed action with `id` (e.g., "MOVE_0", "SWITCH_1") and descriptive `text`

- **[policy.py](src/llm/policy.py)**: LLM-based battle policy using Hugging Face models

  - `LLMPolicy`: Wraps `llm-jp/llm-jp-3.1-1.8b-instruct4` or similar chat models
  - `select_action(battle, player)`: Generates next move from battle state

- **[action_scoring.py](src/llm/action_scoring.py)**: Score actions using damage calculations for labeling

- **[reward.py](src/llm/reward.py)**: Reward functions for RL training

- **[selfplay_rl.py](src/llm/selfplay_rl.py)**: Self-play loop generating (state, action, reward) trajectories

- **[sft_format.py](src/llm/sft_format.py)**: Convert static datasets to chat format for supervised fine-tuning

### 7. Damage Calculator API (`src/damage_calculator_api/`)

FastAPI-based REST service providing high-precision damage calculations.

- **[main.py](src/damage_calculator_api/main.py)**: FastAPI application with CORS, error handling, lifespan management
- **[calculators/damage_calculator.py](src/damage_calculator_api/calculators/damage_calculator.py)**: Core damage engine (16-stage rolls, all modifiers)
- **[calculators/stat_calculator.py](src/damage_calculator_api/calculators/stat_calculator.py)**: Stat calculation with nature/EV/IV
- **Routers**: `/api/v1/damage`, `/api/v1/pokemon`, `/api/v1/info`

### 8. Tournament System (`scripts/matching.py`)

- Elo rating system for trainer ranking
- Resume capability via PostgreSQL database
- Discord webhook notifications
- Randomly selects 2 trainers, runs 3v3 battle using `MyMCTSBattle`, updates ratings

### 9. Data Management

- **[database_handler.py](src/database_handler.py)**: PostgreSQL operations for trainers and battle history
- **[models.py](src/models.py)**: `Trainer` class with `choose_team()` method
- Trainer data: `data/top_rankers/season_27.json`

## Key Development Commands

### Reinforcement Learning Pipeline

```bash
# 1. Generate Self-Play data with MCTS
uv run python scripts/generate_selfplay_dataset.py \
  --trainer-json data/top_rankers/season_27.json \
  --output data/selfplay_records.jsonl \
  --num-games 100 \
  --mcts-iterations 100

# 2. Train Policy-Value Network from Self-Play data
uv run python scripts/train_policy_value_network.py \
  --dataset data/selfplay_records.jsonl \
  --output models/policy_value \
  --hidden-dim 256 \
  --num-epochs 100

# 3. Run full AlphaZero-style RL loop (Self-Play → Train → Evaluate → Repeat)
uv run python scripts/run_reinforcement_loop.py \
  --trainer-json data/top_rankers/season_27.json \
  --output models/reinforcement \
  --num-generations 10 \
  --games-per-generation 100 \
  --evaluation-games 50

# Lightweight test run
uv run python scripts/run_reinforcement_loop.py \
  --trainer-json data/top_rankers/season_27.json \
  --output models/reinforcement_test \
  --num-generations 3 \
  --games-per-generation 20 \
  --evaluation-games 10 \
  --training-epochs 10
```

### ReBeL Training and Comparison

```bash
# ReBeL Value Network training (self-play)
uv run python scripts/train_rebel.py \
  --trainer-json data/top_rankers/season_27.json \
  --usage-db data/pokedb_usage/season_37_top150.json \
  --output models/rebel \
  --num-iterations 10 \
  --games-per-iteration 20

# Fast training with parallel workers and lightweight CFR
uv run python scripts/train_rebel.py \
  --trainer-json data/top_rankers/season_27.json \
  --usage-db data/pokedb_usage/season_37_top150.json \
  --output models/rebel \
  --num-iterations 10 \
  --games-per-iteration 50 \
  --num-workers 4 \
  --lightweight-cfr

# High accuracy training (slower, more precise CFR)
uv run python scripts/train_rebel.py \
  --trainer-json data/top_rankers/season_36.json \
  --usage-db data/pokedb_usage/season_37_top150.json \
  --output models/rebel_v2 \
  --num-iterations 100 \
  --games-per-iteration 50 \
  --cfr-iterations 50 \
  --cfr-world-samples 30 \
  --num-workers 8 \
  --no-lightweight-cfr \
  --device cuda \
  --use-full-belief \
  --train-selection

# Training against fixed opponent (for debugging/testing)
uv run python scripts/train_rebel.py \
  --trainer-json data/top_rankers/season_27.json \
  --fixed-opponent-index 0 \
  --fixed-opponent-select-all \
  --num-iterations 5

# Compare ReBeL vs HypothesisMCTS
uv run python scripts/compare_rebel_vs_mcts.py \
  --trainer-json data/top_rankers/season_27.json \
  --usage-db data/pokedb_usage/season_37_top150.json \
  --num-matches 50 \
  --output results/rebel_vs_mcts.json

# Quick test (fewer matches)
uv run python scripts/compare_rebel_vs_mcts.py \
  --num-matches 5 \
  --rebel-cfr-iterations 20 \
  --rebel-world-samples 10
```

**train_rebel.py オプション一覧:**

| オプション | 説明 | デフォルト |
|------------|------|------------|
| `--num-iterations` | 学習イテレーション数 | 10 |
| `--games-per-iteration` | 各イテレーションでの自己対戦数 | 20 |
| `--cfr-iterations` | CFR イテレーション数 | 30 |
| `--cfr-world-samples` | CFR ワールドサンプル数 | 10 |
| `--num-workers` | 並列ゲーム生成のワーカー数 | 1 |
| `--lightweight-cfr` | 軽量CFRモード（高速、デフォルト有効） | True |
| `--no-lightweight-cfr` | 軽量CFRを無効化（高精度だが低速） | - |
| `--batch-size` | 学習バッチサイズ | 32 |
| `--learning-rate` | 学習率 | 1e-4 |
| `--hidden-dim` | ネットワークの隠れ層次元 | 256 |
| `--device` | デバイス (cpu/cuda) | cpu |
| `--resume` | チェックポイントから再開 | - |
| `--fixed-opponent` | 固定対戦相手のJSONパス | - |
| `--fixed-opponent-index` | trainer-json内の対戦相手インデックス | - |
| `--train-selection` | 選出ネットワークも同時に学習 | False |

**ログ出力:**
- 各イテレーションの学習記録は `{output_dir}/training_log.jsonl` に保存されます
- 最終的な学習履歴は `{output_dir}/training_history.json` に保存されます

### Team Selection Training

```bash
# Train with random data
uv run python scripts/train_team_selection.py \
  --trainer-json data/top_rankers/season_27.json \
  --output models/team_selection \
  --num-samples 10000 \
  --num-epochs 50

# Train with Self-Play data (higher quality)
uv run python scripts/train_team_selection.py \
  --trainer-json data/top_rankers/season_27.json \
  --selfplay-data data/selfplay_records.jsonl \
  --output models/team_selection \
  --num-epochs 100
```

### Running Tournaments

```bash
# Run tournament with resume capability
DISCORD_WEBHOOK_URL={} \
POSTGRES_DB={} \
POSTGRES_PASSWORD={} \
POSTGRES_USER={} \
POSTGRES_HOST={} \
POSTGRES_PORT={} \
uv run python scripts/matching.py --resume
```

### LLM Dataset Generation

```bash
# Generate static dataset from battle simulations
uv run python scripts/generate_llm_static_dataset.py \
  --trainer-json data/top_rankers/season_27.json \
  --output data/llm_static_dataset.jsonl \
  --num-battles 10000

# Convert to chat format for SFT
uv run python scripts/convert_llm_static_to_chat_sft.py \
  --input data/llm_static_dataset.jsonl \
  --output data/llm_sft_chat_dataset.jsonl
```

### Damage Calculator API

```bash
# Start FastAPI server
uv run python src/damage_calculator_api/main.py

# Or with uvicorn
uv run uvicorn src.damage_calculator_api.main:app --reload --port 8000
```

API documentation available at `http://localhost:8000/docs`

### Data Processing

```bash
# Extract battle history from simulation logs
uv run python src/utils/extract_battle_history.py

# Train word2vec model for Pokémon embeddings
uv run python src/utils/train_word2vec.py
```

### Testing

```bash
# Run all tests (uses unittest framework)
uv run python -m unittest discover tests/

# Run specific test file
uv run python -m unittest tests.test_calculators.test_damage_calculator

# Run with pytest (also supported)
uv run pytest tests/
```

### Development Tools

```bash
# Install dependencies
poetry install

# Run Jupyter for analysis
uv run jupyter notebook

# Streamlit leaderboard dashboard
uv run streamlit run src/streamlit_leaderboard.py
```

## Important Implementation Patterns

### Battle Initialization Pattern

When creating battles, **always** initialize Pokémon data first:

```python
from src.pokemon_battle_sim.pokemon import Pokemon
from src.pokemon_battle_sim.battle import Battle

Pokemon.init()  # MUST call before creating any Pokémon

# Then create Pokémon instances
pokemon = Pokemon("ピカチュウ")
pokemon.item = "きあいのタスキ"
# ... set other attributes

battle = Battle()
battle.reset_game()
```

### Command Encoding

- Moves: `0-3` (indexes into Pokémon's move list)
- Switches: `20 + pokemon_index` (e.g., switch to 2nd team member = `21`)
- Special: `SKIP=-1`, `STRUGGLE=30`, `NO_COMMAND=40`

### State Cloning for MCTS

The `Battle` class supports deep cloning via `seed` tracking:

```python
original_battle = Battle(seed=12345)
# During MCTS simulation, clone states
cloned_state = deepcopy(original_battle)
# The clone maintains seed + copy_count for reproducibility
```

### LLM Action Format

LLM outputs must follow strict formats:

- Moves: `"MOVE: わざ名"` or action ID `"MOVE_0"`
- Switches: `"SWITCH: ポケモン名"` or action ID `"SWITCH_1"`

Parsing handled by `parse_llm_action_output()` in [state_representation.py](src/llm/state_representation.py).

### Using Team Selectors

```python
from src.policy_value_network import (
    load_team_selector,
    RandomTeamSelector,
    TopNTeamSelector,
)

# NN-based selector (requires trained model)
selector = load_team_selector("models/team_selection", device="cpu")

# Select 3 Pokémon based on opponent's team
my_team = [...]  # 6 Pokémon data dicts
opp_team = [...]  # Opponent's 6 Pokémon
selected = selector.select(my_team, opp_team, num_select=3)

# Baseline selectors
random_selector = RandomTeamSelector()
top_n_selector = TopNTeamSelector()  # Legacy: first N Pokémon
```

### Database Environment Variables

PostgreSQL connection requires:

- `POSTGRES_HOST`
- `POSTGRES_PORT`
- `POSTGRES_DB`
- `POSTGRES_USER`
- `POSTGRES_PASSWORD`

Optional: `DISCORD_WEBHOOK_URL` for notifications

## Technical Notes

### Dependencies

- Python 3.12+ required
- Key packages: `torch`, `transformers`, `fastapi`, `sqlalchemy`, `langchain`, `gensim`, `lightgbm`
- Uses Poetry for dependency management (`pyproject.toml`)

### Performance Considerations

- MCTS iterations (default 1000) are tunable via the `iterations` parameter
- Deep cloning Battle states is computationally expensive—minimize during production
- LLM inference requires GPU for reasonable performance (uses `torch.bfloat16`)
- GPU is auto-detected; use `--device cuda` to explicitly specify
- RL loop is time-intensive; use lightweight test options for development

### Data Files

Pokémon data loaded from `data/` directory:

- Move data, Pokémon stats, type charts, etc.
- Ensure these files exist before running simulations

Battle logs saved to `logs/battle_log_YYYYMMDD_HHMMSS.txt`

## Known Limitations

### MCTS and Incomplete Information

Pokémon battles are **imperfect information games**. The current MCTS implementations have important limitations regarding how they handle hidden information:

#### What Information is Hidden in Real Battles

| Information              | Real Battle        | Basic MCTS           | Hypothesis MCTS      |
| ------------------------ | ------------------ | -------------------- | -------------------- |
| Opponent's moves         | Unknown until used | **Fully visible** ❌ | **Fully visible** ❌ |
| Opponent's held item     | Unknown            | **Fully visible** ❌ | Sampled from prior ✓ |
| Opponent's EV/IV spread  | Unknown            | **Fully visible** ❌ | **Fully visible** ❌ |
| Opponent's Tera type     | Unknown until used | **Fully visible** ❌ | **Fully visible** ❌ |
| Opponent's action choice | Strategic          | Random assumption △  | Random assumption △  |

#### Implications

1. **Overly Accurate Lookahead**: The MCTS can simulate damage calculations with perfect knowledge of opponent's stats and moves, which is impossible in real battles.

2. **No Move Discovery**: In real battles, you learn opponent's moveset gradually. The simulator knows all 4 moves from the start.

3. **Simplified Opponent Model**: Both MCTS variants assume the opponent plays randomly during rollouts, rather than strategically.

#### Why This May Be Acceptable

In competitive play, much information is effectively public:

- Top-ranked team compositions are published and shared
- Common "template builds" (テンプレ型) are well-known
- Team preview allows experienced players to predict movesets with high accuracy

When using data from `data/top_rankers/`, the simulator operates under an implicit assumption that both players know common competitive builds—which approximates real high-level play.

## Future Work

### Improved Incomplete Information Handling

1. **Moveset Hypothesis Sampling**

   - Extend `ItemBeliefState` to `PokemonBeliefState` covering moves, EVs, and Tera type
   - Sample from usage statistics (e.g., Pokémon Home data, Pikalytics)
   - Challenge: Combinatorial explosion (moves × items × EVs × Tera = millions of combinations per Pokémon)

2. **Bayesian Belief Updates**

   - Update beliefs as information is revealed during battle
   - Example: If opponent uses Dragon Dance, probability of physical attacker increases
   - Requires tracking observation history and conditional probabilities

3. **Information Set MCTS (ISMCTS)**

   - Determinize hidden information at the start of each simulation
   - Average results across multiple determinizations
   - Well-established technique for imperfect information games (poker, etc.)

4. **Improved Opponent Modeling**

   - Replace random rollout policy with heuristic-based or learned opponent model
   - Options: Damage maximization heuristic, minimax at shallow depth, or neural network policy

5. **Neural Network Approaches**
   - Train networks on **observable state only** (hiding opponent's private information)
   - The network implicitly learns to handle uncertainty through training data distribution
   - This is the direction of `src/llm/` and `src/policy_value_network/`

### Alternative Approaches to MCTS

Beyond improving MCTS, there are other algorithmic approaches better suited for imperfect information games:

1. **Counterfactual Regret Minimization (CFR)**

   - Used by poker AIs (Libratus, Pluribus) to find Nash equilibrium strategies
   - Theoretically optimal but computationally expensive for large state spaces
   - Deep CFR / Neural Fictitious Self-Play (NFSP) scale better with neural network approximation
   - Challenge: Pokémon's state space may be too large even for Deep CFR

2. **ReBeL (Recursive Belief-based Learning)**

   - Meta AI's approach combining search with learned value functions on public information
   - Tracks beliefs about hidden information as part of the state
   - Well-suited for games where information is gradually revealed (like Pokémon)
   - Promising direction for future implementation

3. **Opponent Modeling with RL**
   - Explicitly model opponent's policy and incorporate predictions into decision-making
   - Can learn opponent tendencies from historical data
   - More practical than full game-theoretic solutions

| Approach                    | Theory | Scalability | Implementation | Pokémon Fit             |
| --------------------------- | ------ | ----------- | -------------- | ----------------------- |
| CFR                         | ◎      | △           | Medium         | △ State space too large |
| Deep CFR/NFSP               | ○      | ○           | Hard           | ○                       |
| ReBeL                       | ○      | ○           | Hard           | ◎ Gradual info reveal   |
| Opponent Modeling           | △      | ◎           | Medium         | ○                       |
| MCTS + Hypothesis (current) | △      | ○           | Easy           | ○                       |

### LLM-Based Approaches

The `src/llm/` pipeline represents a promising direction with unique advantages:

**Strengths for Pokémon battles:**

- Implicit handling of incomplete information through pattern learning from large datasets
- Access to pre-existing knowledge (team archetypes, common strategies, matchup theory)
- Flexible reasoning about opponent intentions ("why would they switch here?")
- Explainability: can articulate reasoning in natural language

**Current limitations:**

- Inference latency (seconds per decision vs. milliseconds for NN)
- Hallucination risk (inventing non-existent moves/abilities)
- Weak at precise numerical reasoning (damage calculations)
- Inconsistent decisions in similar situations

**Promising hybrid architecture:**

```
┌─────────────────────────────────────────────────────┐
│  LLM Layer: Strategic Reasoning                      │
│  - Assess win conditions                             │
│  - Predict opponent's game plan                      │
│  - Decide high-level strategy (aggressive/defensive) │
└─────────────────────┬───────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────┐
│  Calculation Module: Tactical Execution              │
│  - Damage calculator API                             │
│  - Speed tier calculations                           │
│  - KO probability estimation                         │
└─────────────────────┬───────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────┐
│  Decision Integration                                │
│  - Rank candidate actions by expected value          │
│  - Final action selection                            │
└─────────────────────────────────────────────────────┘
```

**Future directions:**

- Tool-augmented LLM that calls damage calculator API during reasoning
- Fine-tuning on high-quality annotated battle logs with strategic commentary
- Retrieval-augmented generation (RAG) with matchup databases
- Distillation from LLM reasoning into faster policy networks

### Other Enhancements

- [ ] Support for double battles
- [ ] More comprehensive Gen 9 mechanics (all abilities, items, moves)
- [ ] Real-time battle log parsing for online play integration
- [ ] Distributed self-play for faster training
