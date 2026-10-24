"""Unit tests for the read-only API storage layer."""

from __future__ import annotations

import json
import uuid

import pytest

from rl_agent_lab.api.storage import (
    build_algorithm_comparison,
    build_reward_series,
    build_summary,
    filter_records,
    get_by_id,
    load_all,
    paginate,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_record(
    algorithm="DQN",
    environment="CartPole-v1",
    seed=42,
    episodes=100,
    avg=100.0,
    status="completed",
    exp_id=None,
):
    return {
        "schema_version": 1,
        "experiment_id": exp_id or str(uuid.uuid4()),
        "timestamp_utc": "2026-10-22T09:00:00+00:00",
        "algorithm": algorithm,
        "environment": environment,
        "seed": seed,
        "episodes": episodes,
        "status": status,
        "metrics": {
            "average_reward": avg,
            "final_100_average_reward": avg,
            "best_reward": avg + 10,
        },
        "model_path": None,
    }


# ---------------------------------------------------------------------------
# load_all
# ---------------------------------------------------------------------------


def test_load_all_returns_empty_for_missing_file(tmp_path):
    result = load_all(path=tmp_path / "nonexistent.jsonl")
    assert result == []


def test_load_all_reads_valid_records(tmp_path):
    records = [_make_record("DQN"), _make_record("REINFORCE")]
    jsonl = tmp_path / "experiments.jsonl"
    jsonl.write_text(
        "\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8"
    )
    loaded = load_all(path=jsonl)
    assert len(loaded) == 2
    assert loaded[0]["algorithm"] == "DQN"
    assert loaded[1]["algorithm"] == "REINFORCE"


def test_load_all_skips_malformed_lines(tmp_path):
    good = _make_record("DQN")
    jsonl = tmp_path / "experiments.jsonl"
    jsonl.write_text(
        json.dumps(good) + "\nBAD_LINE\n" + json.dumps(_make_record("PPO")) + "\n",
        encoding="utf-8",
    )
    loaded = load_all(path=jsonl)
    assert len(loaded) == 2


def test_load_all_skips_blank_lines(tmp_path):
    rec = _make_record()
    jsonl = tmp_path / "experiments.jsonl"
    jsonl.write_text("\n\n" + json.dumps(rec) + "\n\n", encoding="utf-8")
    assert len(load_all(path=jsonl)) == 1


def test_load_all_reads_legacy_records_without_schema_version(tmp_path):
    legacy = {
        "timestamp_utc": "2026-10-17T09:28:56+00:00",
        "algorithm": "Q-Learning",
        "environment": "GridWorld",
        "seed": 42,
        "episodes": 100,
        "metrics": {"average_reward": -91.23},
        "model_path": None,
    }
    jsonl = tmp_path / "experiments.jsonl"
    jsonl.write_text(json.dumps(legacy) + "\n", encoding="utf-8")
    loaded = load_all(path=jsonl)
    assert len(loaded) == 1
    assert loaded[0]["algorithm"] == "Q-Learning"


# ---------------------------------------------------------------------------
# filter_records
# ---------------------------------------------------------------------------


def test_filter_by_algorithm():
    records = [_make_record("DQN"), _make_record("REINFORCE")]
    assert len(filter_records(records, algorithm="DQN")) == 1


def test_filter_by_environment():
    records = [
        _make_record("DQN", "CartPole-v1"),
        _make_record("PPO", "LunarLander-v3"),
    ]
    filtered = filter_records(records, environment="CartPole-v1")
    assert len(filtered) == 1
    assert filtered[0]["algorithm"] == "DQN"


def test_filter_by_status():
    records = [
        _make_record(status="completed"),
        _make_record(status="failed"),
    ]
    assert len(filter_records(records, status="completed")) == 1
    assert len(filter_records(records, status="failed")) == 1


def test_filter_no_criteria_returns_all():
    records = [_make_record("DQN"), _make_record("PPO")]
    assert filter_records(records) == records


def test_filter_combined():
    records = [
        _make_record("DQN", "CartPole-v1"),
        _make_record("DQN", "LunarLander-v3"),
        _make_record("PPO", "CartPole-v1"),
    ]
    result = filter_records(records, algorithm="DQN", environment="CartPole-v1")
    assert len(result) == 1


# ---------------------------------------------------------------------------
# get_by_id
# ---------------------------------------------------------------------------


def test_get_by_id_finds_record():
    eid = str(uuid.uuid4())
    records = [_make_record(exp_id=eid), _make_record()]
    found = get_by_id(records, eid)
    assert found is not None
    assert found["experiment_id"] == eid


def test_get_by_id_returns_none_for_missing():
    records = [_make_record()]
    assert get_by_id(records, str(uuid.uuid4())) is None


# ---------------------------------------------------------------------------
# paginate
# ---------------------------------------------------------------------------


def test_paginate_first_page():
    records = [_make_record() for _ in range(5)]
    page, total = paginate(records, page=1, page_size=2)
    assert total == 5
    assert len(page) == 2


def test_paginate_last_page():
    records = [_make_record() for _ in range(5)]
    page, total = paginate(records, page=3, page_size=2)
    assert total == 5
    assert len(page) == 1


def test_paginate_beyond_last_page():
    records = [_make_record() for _ in range(3)]
    page, total = paginate(records, page=5, page_size=10)
    assert total == 3
    assert page == []


def test_paginate_empty():
    page, total = paginate([], page=1, page_size=10)
    assert total == 0
    assert page == []


# ---------------------------------------------------------------------------
# build_summary
# ---------------------------------------------------------------------------


def test_build_summary_counts(tmp_path):
    records = [
        _make_record("DQN", "CartPole-v1"),
        _make_record("REINFORCE", "CartPole-v1"),
        _make_record("PPO", "LunarLander-v3"),
    ]
    summary = build_summary(records)
    assert summary["total_experiments"] == 3
    assert "DQN" in summary["algorithms"]
    assert "CartPole-v1" in summary["environments"]
    assert summary["status_counts"]["completed"] == 3


def test_build_summary_metric_averages():
    records = [
        _make_record("DQN", avg=100.0),
        _make_record("REINFORCE", avg=200.0),
    ]
    summary = build_summary(records)
    assert summary["metric_averages"]["average_reward"] == pytest.approx(150.0)


def test_build_summary_empty():
    summary = build_summary([])
    assert summary["total_experiments"] == 0
    assert summary["algorithms"] == []
    assert summary["metric_averages"] == {}


def test_build_summary_excludes_nan_and_inf():
    records = [
        {
            "algorithm": "DQN",
            "environment": "CartPole-v1",
            "status": "completed",
            "metrics": {"average_reward": 100.0},
        },
        {
            "algorithm": "DQN",
            "environment": "CartPole-v1",
            "status": "completed",
            "metrics": {"average_reward": float("nan")},
        },
    ]
    summary = build_summary(records)
    # Only the valid metric contributes.
    assert summary["metric_averages"]["average_reward"] == pytest.approx(100.0)


# ---------------------------------------------------------------------------
# build_algorithm_comparison
# ---------------------------------------------------------------------------


def test_build_algorithm_comparison_statistics():
    records = [
        _make_record("DQN", avg=100.0),
        _make_record("DQN", avg=200.0),
        _make_record("REINFORCE", avg=300.0),
    ]
    result = build_algorithm_comparison(records)
    algs = {r["algorithm"]: r for r in result}
    assert algs["DQN"]["count"] == 2
    assert algs["DQN"]["mean"] == pytest.approx(150.0)
    assert algs["DQN"]["std"] == pytest.approx(50.0)
    assert algs["REINFORCE"]["std"] == pytest.approx(0.0)


def test_build_algorithm_comparison_environment_filter():
    records = [
        _make_record("DQN", "CartPole-v1", avg=100.0),
        _make_record("DQN", "LunarLander-v3", avg=999.0),
    ]
    result = build_algorithm_comparison(records, environment="CartPole-v1")
    assert len(result) == 1
    assert result[0]["mean"] == pytest.approx(100.0)


def test_build_algorithm_comparison_skips_invalid_metrics():

    records = [
        _make_record("DQN", avg=100.0),
        {
            "algorithm": "DQN",
            "metrics": {"average_reward": float("nan")},
        },
        {
            "algorithm": "DQN",
            "metrics": {"average_reward": True},
        },
    ]
    result = build_algorithm_comparison(records)
    assert result[0]["count"] == 1


def test_build_algorithm_comparison_empty():
    assert build_algorithm_comparison([]) == []


# ---------------------------------------------------------------------------
# build_reward_series
# ---------------------------------------------------------------------------


def test_build_reward_series_order_and_fields():
    records = [
        _make_record("DQN", avg=100.0),
        _make_record("REINFORCE", avg=200.0),
    ]
    series = build_reward_series(records)
    assert len(series) == 2
    for pt in series:
        for key in ("timestamp_utc", "algorithm", "environment", "value"):
            assert key in pt


def test_build_reward_series_algorithm_filter():
    records = [_make_record("DQN", avg=100.0), _make_record("PPO", avg=50.0)]
    series = build_reward_series(records, algorithm="DQN")
    assert len(series) == 1
    assert series[0]["algorithm"] == "DQN"


def test_build_reward_series_skips_missing_metric():
    records = [
        _make_record("DQN", avg=100.0),
        {"algorithm": "DQN", "metrics": {}},
    ]
    series = build_reward_series(records, metric="average_reward")
    assert len(series) == 1


def test_build_reward_series_empty():
    assert build_reward_series([]) == []
