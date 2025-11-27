# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Pokémon battle simulator using Monte Carlo Tree Search (MCTS) for AI decision-making. The system simulates competitive Pokémon battles with realistic mechanics, manages trainer rankings using Elo ratings, and provides battle analytics.

## Core Architecture

### Main Components

1. **Battle Simulation Engine** (`src/pokemon_battle_sim/`)
   - `battle.py`: Core battle logic, turn management, and game state
   - `pokemon.py`: Pokémon class with comprehensive battle mechanics (stats, abilities, moves, items)
   - `damage.py`: Damage calculation engine
   - `utils.py`: Battle utility functions

2. **MCTS AI System** (`src/mcts/`)
   - `mcts_battle.py`: MCTS implementation with UCT (Upper Confidence Bound applied to Trees)
   - `MyMCTSBattle`: Extends base Battle class with AI decision-making

3. **Tournament Management** (`scripts/`)
   - `matching.py`: Main tournament runner with Elo rating system
   - Handles trainer matching, battle execution, and result tracking

4. **Data Management** (`src/`)
   - `database_handler.py`: PostgreSQL database operations
   - `models.py`: Data models (Trainer class)
   - `utils/`: Data extraction and ML model training utilities

## Key Development Commands

### Running Battles
```bash
# Run tournament with resume capability
DISCORD_WEBHOOK_URL={} \
POSTGRES_DB={} \
POSTGRES_PASSWORD={} \
POSTGRES_USER={} \
POSTGRES_HOST={} \
POSTGRES_PORT={} \
poetry run python scripts/matching.py --resume
```

### Data Processing
```bash
# Extract battle history from simulation logs
poetry run python src/utils/extract_battle_history.py

# Train word2vec model for Pokémon analysis
poetry run python src/utils/train_word2vec.py
```

### Development Setup
```bash
# Install dependencies
poetry install

# Run Jupyter notebooks for analysis
poetry run jupyter notebook
```

## Technical Implementation Details

### Battle System
- **Turn-based simulation**: Each turn processes both players' commands simultaneously
- **Command types**: Battle moves, switch/change commands, status effects
- **State management**: Deep cloning for MCTS simulations without affecting main game state

### MCTS Algorithm
- **UCT exploration**: Uses sqrt(2) exploration parameter
- **Policy separation**: Different policies for battle commands vs. switching
- **Evaluation function**: TOD_score-based board evaluation with ratio scoring
- **Iterations**: Default 1000 iterations per decision (configurable)

### Database Schema
- **Trainers**: Stores trainer data with Elo ratings
- **Battle History**: Records all matches with timestamps and ratings
- **Resume functionality**: Can restart tournaments from last saved state

### Data Sources
- Trainer data loaded from `data/top_rankers/season_27.json`
- Pokémon data from various text files in `data/`
- Battle logs saved to `logs/` directory with timestamps

## Development Notes

### Poetry Configuration
- Uses Poetry for dependency management
- Python 3.12+ required
- Key dependencies: pandas, requests, langchain, openai, sqlalchemy, psycopg2-binary

### Testing and Validation
- Battle logs are automatically saved for analysis
- Discord notifications for tournament progress
- Streamlit dashboard for leaderboard visualization

### Performance Considerations
- MCTS iterations can be adjusted based on computational resources
- Battle state cloning is computationally expensive - optimize for production use
- PostgreSQL connection pooling recommended for high-volume tournaments

## Data Flow

1. **Tournament Start**: Load trainer data from JSON files
2. **Battle Matching**: Elo-based matchmaking or random battles
3. **MCTS Decision**: AI evaluates possible moves using tree search
4. **Battle Execution**: Simulate turn-by-turn combat
5. **Result Recording**: Update Elo ratings and save battle logs
6. **Analytics**: Extract patterns and train ML models

## Common File Locations

- Battle logs: `logs/battle_log_YYYYMMDD_HHMMSS.txt`
- Trainer data: `data/top_rankers/season_27.json`
- Pokémon data: `data/*.txt` files
- Notebooks: `notebook/` directory for analysis
- Models: `models/` directory for trained ML models