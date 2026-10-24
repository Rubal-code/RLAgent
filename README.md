# RLAgent — Reinforcement Learning Experimentation Platform

A hands-on RL laboratory with experiment tracking, multi-seed evaluation,
algorithm comparison, and a REST API ready for frontend consumption.

---

## Table of Contents

1. [Architecture](#architecture)
2. [Supported Algorithms and Environments](#supported-algorithms-and-environments)
3. [Experiment Record Format](#experiment-record-format)
4. [Running Training](#running-training)
5. [Multi-Seed Evaluation](#multi-seed-evaluation)
6. [Comparing Experiments](#comparing-experiments)
7. [Starting the API Server](#starting-the-api-server)
8. [API Endpoints](#api-endpoints)
9. [Running Tests and Linting](#running-tests-and-linting)
10. [Configuration and Environment Variables](#configuration-and-environment-variables)
11. [Known Limitations](#known-limitations)

---

## Architecture

```
D:\RLAgent
├── src/rl_agent_lab/
│   ├── agents/          # Agent classes (DQN, REINFORCE, PPO, Q-Learning)
│   ├── environments/    # Custom environments (GridWorld, TicTacToe)
│   ├── training/        # Training scripts, experiment tracker, evaluation
│   │   ├── experiment_tracker.py     # Core append-only JSONL logger
│   │   ├── run_experiment.py         # Unified CLI experiment runner
│   │   ├── evaluate_multiple_seeds.py# Multi-seed evaluation (DQN, REINFORCE)
│   │   └── compare_experiments.py    # Historical comparison from JSONL
│   ├── api/             # FastAPI REST API (read-only)
│   │   ├── main.py      # Application + route handlers
│   │   └── storage.py   # Read-only JSONL data-access layer
│   └── visualization/   # Plotting utilities
├── models/              # Saved model artifacts (.pth, .json, .pkl)
├── results/
│   ├── experiments.jsonl            # Append-only experiment history
│   └── multi_seed_evaluation.json   # Latest multi-seed summary
├── tests/               # Pytest test suite
├── pyproject.toml
├── requirements.txt
└── requirements-dev.txt
```

**Data flow:**
- Training scripts call `log_experiment()` → appended to `experiments.jsonl`
- API reads `experiments.jsonl` at request time (no caching, no DB)
- Multiple requests never mutate the JSONL file

---

## Supported Algorithms and Environments

| Algorithm | Environment      | Model artifact              |
|-----------|------------------|-----------------------------|
| Q-Learning | GridWorld (5×5) | `models/q_table.json`       |
| Q-Learning | TicTacToe-vs-Random | `models/tictactoe_q_table.pkl` |
| DQN       | CartPole-v1      | `models/cartpole_dqn.pth`   |
| REINFORCE | CartPole-v1      | `models/cartpole_reinforce.pth` |
| PPO       | LunarLander-v3   | `models/lunarlander_ppo.pth` |

---

## Experiment Record Format

Each line in `results/experiments.jsonl` is one JSON object:

```json
{
  "schema_version": 1,
  "experiment_id": "550e8400-e29b-41d4-a716-446655440000",
  "timestamp_utc": "2026-10-22T09:25:21.438206+00:00",
  "algorithm": "DQN",
  "environment": "CartPole-v1",
  "seed": 42,
  "episodes": 500,
  "status": "completed",
  "metrics": {
    "average_reward": 196.184,
    "final_100_average_reward": 339.26,
    "best_reward": 500.0
  },
  "model_path": "models/cartpole_dqn.pth",
  "hyperparameters": {
    "learning_rate": 0.001,
    "discount_factor": 0.99,
    "exploration_decay": 0.995,
    "min_exploration_rate": 0.01
  }
}
```

**Fields:**

| Field | Type | Required | Notes |
|---|---|---|---|
| `schema_version` | int | Yes (new records) | Omitted in legacy records |
| `experiment_id` | UUID string | Yes (new records) | Omitted in legacy records |
| `timestamp_utc` | ISO 8601 string | Yes | UTC timezone |
| `algorithm` | string | Yes | Non-empty |
| `environment` | string | Yes | Non-empty |
| `seed` | int | Yes | |
| `episodes` | int | Yes | > 0 |
| `status` | string | Yes (new records) | `"completed"` or `"failed"` |
| `metrics` | object | Yes | All values must be finite numbers |
| `model_path` | string \| null | No | |
| `hyperparameters` | object | No | Omitted when not provided |

**Metric definitions:**

| Metric | Definition |
|---|---|
| `average_reward` | Mean episode reward over all training episodes |
| `final_100_average_reward` | Mean reward over the last min(100, N) episodes |
| `best_reward` | Maximum single-episode reward |
| `success_rate` | Fraction of training episodes reaching the goal (Q-Learning/GridWorld) |

> **Note:** `final_100_average_reward` reflects the last 100 *training* episodes,
> not an independent evaluation set. It indicates late-training performance,
> not generalisation.

---

## Running Training

Activate the virtual environment first:

```powershell
venv\Scripts\activate
```

**Unified runner:**

```powershell
python -m rl_agent_lab.training.run_experiment --algorithm q-learning --episodes 1000 --seed 42
python -m rl_agent_lab.training.run_experiment --algorithm dqn --episodes 500 --seed 42
python -m rl_agent_lab.training.run_experiment --algorithm reinforce --episodes 500 --seed 42
python -m rl_agent_lab.training.run_experiment --algorithm ppo --episodes 500 --seed 42
python -m rl_agent_lab.training.run_experiment --algorithm tictactoe --episodes 50000 --seed 42
```

Or use the registered entry point:

```powershell
run-experiment --algorithm dqn --episodes 500 --seed 42
```

Each run appends one record to `results/experiments.jsonl` and saves the model.

---

## Multi-Seed Evaluation

Evaluate DQN and REINFORCE on CartPole-v1 across multiple seeds:

```powershell
python -m rl_agent_lab.training.evaluate_multiple_seeds --episodes 500 --seeds 42 123 456
```

Or:

```powershell
evaluate-seeds --episodes 500 --seeds 42 123 456 --output results/multi_seed_evaluation.json
```

- Each (algorithm, seed) run is logged to `experiments.jsonl`.
- The summary JSON is written to the `--output` path.
- Seeds must be unique.

---

## Comparing Experiments

Compare historical runs from `experiments.jsonl`:

```powershell
python -m rl_agent_lab.training.compare_experiments --metric average_reward
python -m rl_agent_lab.training.compare_experiments --environment CartPole-v1 --algorithms DQN REINFORCE
```

Or:

```powershell
compare-experiments --metric final_100_average_reward --environment CartPole-v1
```

**Statistics reported:** run count, mean, and **population** standard deviation (denominator *N*).

---

## Starting the API Server

```powershell
rl-api
```

Or directly:

```powershell
python -m rl_agent_lab.api.main
```

Or with uvicorn for development with auto-reload:

```powershell
venv\Scripts\uvicorn rl_agent_lab.api.main:app --reload --port 8000
```

The API starts on `http://localhost:8000`.

- Interactive docs: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

The API is **read-only** and serves data directly from `results/experiments.jsonl`.
No database is required.

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Liveness check |
| GET | `/api/v1/experiments` | List experiments (paginated, filterable) |
| GET | `/api/v1/experiments/{id}` | Single experiment by UUID |
| GET | `/api/v1/dashboard/summary` | Aggregated counts and metric averages |
| GET | `/api/v1/dashboard/algorithms` | Per-algorithm metric comparison |
| GET | `/api/v1/dashboard/rewards` | Time-ordered metric series for charts |

See [FRONTEND_CONTRACT.md](FRONTEND_CONTRACT.md) for full schema documentation.

---

## Running Tests and Linting

```powershell
# Run all tests
venv\Scripts\python -m pytest -q

# Run with coverage
venv\Scripts\python -m pytest --cov=rl_agent_lab --cov-report=term-missing -q

# Ruff lint check
venv\Scripts\python -m ruff check .

# Ruff auto-fix
venv\Scripts\python -m ruff check --fix .
```

---

## Configuration and Environment Variables

| Variable | Default | Description |
|---|---|---|
| `RL_EXPERIMENTS_FILE` | `results/experiments.jsonl` | Path to the JSONL experiment history |

The `RL_EXPERIMENTS_FILE` variable is resolved at import time in `rl_agent_lab.api.storage`.
In tests, redirect it to a temporary file via `monkeypatch.setenv`.

---

## Known Limitations

1. **No independent evaluation protocol.** `final_100_average_reward` reflects
   the last 100 training episodes, not a separate held-out evaluation. The
   training and evaluation environments are identical for DQN and REINFORCE
   (CartPole-v1 with the same Gymnasium seed behaviour).

2. **Population standard deviation.** The comparison layer and multi-seed
   summary report population std (denominator *N*), not sample std (denominator
   *N−1*). With 2–3 seeds this is a rough estimate only.

3. **Single-file storage.** The API reads `experiments.jsonl` on every request.
   For very large histories (tens of thousands of records) consider adding
   an index or caching layer.

4. **No authentication.** The API has no authentication. Do not expose it to
   the public internet without adding an auth layer.

5. **Legacy records.** Records written before schema version 1 lack
   `experiment_id` and `schema_version`. They are fully readable via
   `GET /api/v1/experiments` but cannot be retrieved via
   `GET /api/v1/experiments/{id}`.

6. **CORS.** The API currently allows all origins (`allow_origins=["*"]`).
   Restrict this to the frontend origin in production.