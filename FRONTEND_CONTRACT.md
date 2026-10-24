# RLAgent Frontend API Contract

This document defines the stable API contract for frontend development.
Every schema shown here is derived from the actual backend implementation.
No fields are fabricated.

**Base URL (local development):** `http://localhost:8000`

**Protocol:** HTTP/1.1 (HTTPS in production)

**Content type:** All responses are `application/json`.

**CORS:** All origins are allowed during development. Restrict in production.

---

## Table of Contents

1. [Common Conventions](#common-conventions)
2. [Error Responses](#error-responses)
3. [GET /health](#get-health)
4. [GET /api/v1/experiments](#get-apiv1experiments)
5. [GET /api/v1/experiments/{experiment_id}](#get-apiv1experimentsexperiment_id)
6. [GET /api/v1/dashboard/summary](#get-apiv1dashboardsummary)
7. [GET /api/v1/dashboard/algorithms](#get-apiv1dashboardalgorithms)
8. [GET /api/v1/dashboard/rewards](#get-apiv1dashboardrewards)
9. [Metric Definitions](#metric-definitions)
10. [Empty State Behaviour](#empty-state-behaviour)
11. [Frontend Component Mapping](#frontend-component-mapping)

---

## Common Conventions

- All timestamps are **ISO 8601** strings in UTC (`+00:00` offset).
- `experiment_id` is a **UUID v4** string. Legacy records (pre-schema-v1)
  lack this field; those records appear in list responses but cannot be
  retrieved individually.
- Numeric metric values are always **finite** (NaN and infinity are excluded).
- String filter parameters use **exact equality** matching.
- Pagination is **1-based** (first page is `page=1`).

---

## Error Responses

All errors follow FastAPI's default structure:

```json
{
  "detail": "Human-readable error message"
}
```

| HTTP Status | Meaning |
|---|---|
| 200 | Success |
| 404 | Resource not found |
| 422 | Validation error (invalid query parameter) |

---

## GET /health

**Purpose:** Verify the API process is running.

### Request

```
GET /health
```

No parameters.

### Response — 200 OK

```json
{
  "status": "ok"
}
```

---

## GET /api/v1/experiments

**Purpose:** List experiment records with optional filtering and pagination.

### Request

```
GET /api/v1/experiments?algorithm=DQN&environment=CartPole-v1&status=completed&page=1&page_size=20
```

### Query Parameters

| Parameter | Type | Required | Default | Constraints | Description |
|---|---|---|---|---|---|
| `algorithm` | string | No | (all) | Exact match | e.g. `DQN`, `REINFORCE`, `PPO`, `Q-Learning` |
| `environment` | string | No | (all) | Exact match | e.g. `CartPole-v1`, `LunarLander-v3`, `GridWorld` |
| `status` | string | No | (all) | Exact match | `completed` or `failed` |
| `page` | integer | No | 1 | ≥ 1 | Page number (1-based) |
| `page_size` | integer | No | 20 | 1–200 | Records per page |

### Response — 200 OK

```json
{
  "page": 1,
  "page_size": 20,
  "total": 42,
  "total_pages": 3,
  "experiments": [
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
        "discount_factor": 0.99
      }
    }
  ]
}
```

**Notes:**
- `experiments` is empty (`[]`) when no records match.
- Legacy records may omit `schema_version`, `experiment_id`, `status`, and `hyperparameters`.
- `model_path` may be `null`.

### Response — 422 Unprocessable Entity

```json
{
  "detail": [
    {
      "type": "greater_than_equal",
      "loc": ["query", "page_size"],
      "msg": "Input should be greater than or equal to 1",
      "input": "0"
    }
  ]
}
```

---

## GET /api/v1/experiments/{experiment_id}

**Purpose:** Retrieve a single experiment by its UUID.

### Request

```
GET /api/v1/experiments/550e8400-e29b-41d4-a716-446655440000
```

### Path Parameters

| Parameter | Type | Description |
|---|---|---|
| `experiment_id` | UUID string | The `experiment_id` from a list response |

### Response — 200 OK

Same structure as a single item in the `experiments` array above.

### Response — 404 Not Found

```json
{
  "detail": "Experiment '550e8400-e29b-41d4-a716-446655440000' not found."
}
```

---

## GET /api/v1/dashboard/summary

**Purpose:** Aggregate counts and averages for dashboard summary cards.

### Request

```
GET /api/v1/dashboard/summary
```

No parameters.

### Response — 200 OK

```json
{
  "total_experiments": 42,
  "algorithms": ["DQN", "PPO", "Q-Learning", "REINFORCE"],
  "environments": ["CartPole-v1", "GridWorld", "LunarLander-v3", "TicTacToe-vs-Random"],
  "status_counts": {
    "completed": 40,
    "failed": 2
  },
  "metric_averages": {
    "average_reward": 123.45,
    "final_100_average_reward": 200.11,
    "best_reward": 450.0,
    "success_rate": 0.91,
    "win_rate": 0.95
  }
}
```

**Notes:**
- `algorithms` and `environments` are **sorted alphabetically**.
- `metric_averages` is computed across **all** records regardless of algorithm
  or environment. Averages mix incomparable metrics (e.g. CartPole reward vs.
  LunarLander reward). Use `/api/v1/dashboard/algorithms` for meaningful comparisons.
- `status_counts` may be empty `{}` if the history file has no records.

---

## GET /api/v1/dashboard/algorithms

**Purpose:** Algorithm-level metric comparison for bar/comparison charts.

### Request

```
GET /api/v1/dashboard/algorithms?metric=average_reward&environment=CartPole-v1
```

### Query Parameters

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `metric` | string | No | `average_reward` | Metric name to compare |
| `environment` | string | No | (all) | Restrict to one environment |

### Response — 200 OK

```json
{
  "metric": "average_reward",
  "environment": "CartPole-v1",
  "algorithms": [
    {
      "algorithm": "DQN",
      "count": 5,
      "mean": 196.18,
      "std": 42.3,
      "environment": "CartPole-v1"
    },
    {
      "algorithm": "REINFORCE",
      "count": 3,
      "mean": 267.9,
      "std": 15.2,
      "environment": "CartPole-v1"
    }
  ]
}
```

**Notes:**
- `std` is **population** standard deviation (denominator *N*).
- `algorithms` is sorted alphabetically by algorithm name.
- `algorithms` is `[]` when no records match.
- Algorithms without any valid values for the chosen metric are omitted.

### Response — 422 Unprocessable Entity (blank metric)

```json
{
  "detail": "'metric' must not be blank."
}
```

---

## GET /api/v1/dashboard/rewards

**Purpose:** Time-ordered metric series for line charts.

### Request

```
GET /api/v1/dashboard/rewards?metric=average_reward&algorithm=DQN&environment=CartPole-v1
```

### Query Parameters

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `metric` | string | No | `average_reward` | Metric name |
| `algorithm` | string | No | (all) | Filter by algorithm |
| `environment` | string | No | (all) | Filter by environment |

### Response — 200 OK

```json
{
  "metric": "average_reward",
  "algorithm": "DQN",
  "environment": "CartPole-v1",
  "series": [
    {
      "experiment_id": "550e8400-e29b-41d4-a716-446655440000",
      "timestamp_utc": "2026-10-22T09:25:21.438206+00:00",
      "algorithm": "DQN",
      "environment": "CartPole-v1",
      "seed": 42,
      "episodes": 500,
      "value": 196.184
    },
    {
      "experiment_id": null,
      "timestamp_utc": "2026-10-21T17:40:46.997225+00:00",
      "algorithm": "DQN",
      "environment": "CartPole-v1",
      "seed": 42,
      "episodes": 10,
      "value": 21.2
    }
  ]
}
```

**Notes:**
- Points are in **append order** (oldest first).
- `experiment_id` may be `null` for legacy records.
- `series` is `[]` when no matching records exist.

---

## Metric Definitions

| Metric Key | Description | Applicable Algorithms |
|---|---|---|
| `average_reward` | Mean episode reward over all training episodes | DQN, REINFORCE, PPO, Q-Learning/GridWorld |
| `final_100_average_reward` | Mean reward over last min(100, N) training episodes | DQN, REINFORCE, PPO, Q-Learning/GridWorld |
| `best_reward` | Maximum single-episode reward | DQN, REINFORCE, PPO |
| `success_rate` | Fraction of episodes reaching goal | Q-Learning/GridWorld |
| `win_rate` | Win fraction in TicTacToe evaluation | Q-Learning/TicTacToe |
| `draw_rate` | Draw fraction in TicTacToe evaluation | Q-Learning/TicTacToe |
| `loss_rate` | Loss fraction in TicTacToe evaluation | Q-Learning/TicTacToe |

> **Important:** `average_reward` values for CartPole-v1 (range ~9–500) and
> LunarLander-v3 (range ~-500–300) are **not directly comparable**. Always
> filter by `environment` when comparing algorithms numerically.

---

## Empty State Behaviour

| Endpoint | Empty state response |
|---|---|
| `GET /api/v1/experiments` | `{"total": 0, "experiments": [], ...}` — HTTP 200 |
| `GET /api/v1/experiments/{id}` | HTTP 404 |
| `GET /api/v1/dashboard/summary` | `{"total_experiments": 0, "algorithms": [], "metric_averages": {}, ...}` — HTTP 200 |
| `GET /api/v1/dashboard/algorithms` | `{"algorithms": []}` — HTTP 200 |
| `GET /api/v1/dashboard/rewards` | `{"series": []}` — HTTP 200 |

All list/collection endpoints return HTTP **200** with empty collections; they
never return 404 for an empty dataset.

---

## Frontend Component Mapping

| UI Component | Recommended Endpoint |
|---|---|
| Summary cards (total runs, algorithms, etc.) | `GET /api/v1/dashboard/summary` |
| Algorithm comparison bar chart | `GET /api/v1/dashboard/algorithms?environment=CartPole-v1` |
| Reward over time line chart | `GET /api/v1/dashboard/rewards?algorithm=DQN&environment=CartPole-v1` |
| Experiment history table | `GET /api/v1/experiments?page=1&page_size=20` |
| Algorithm filter in table | `GET /api/v1/experiments?algorithm=DQN` |
| Environment filter in table | `GET /api/v1/experiments?environment=CartPole-v1` |
| Experiment detail view | `GET /api/v1/experiments/{experiment_id}` |
| Algorithm dropdown options | `summary.algorithms` from `/api/v1/dashboard/summary` |
| Environment dropdown options | `summary.environments` from `/api/v1/dashboard/summary` |
