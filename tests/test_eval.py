"""Golden set is valid; eval modules import; RAGAS dataset-building runs in mock mode.

The RAGAS judge needs the dedicated `.venv-ragas`, so its import is guarded — these tests tolerate
RAGAS being absent in the main venv (the guard raises a clear RuntimeError).
"""

from __future__ import annotations

from finassist.eval import phoenix_trace, run_ragas


def test_golden_set_is_valid_and_covers_failure_modes():
    rows = run_ragas.load_golden()
    assert len(rows) >= 10
    for row in rows:
        assert row["question"].strip()
        assert row["ground_truth"].strip()
        assert "category" in row and "expected_refusal" in row
    categories = {r["category"] for r in rows}
    assert {"factual", "yoy", "refusal"} <= categories
    assert sum(1 for r in rows if r["expected_refusal"]) >= 2


def test_eval_modules_expose_entry_points():
    assert callable(phoenix_trace.launch)
    assert callable(phoenix_trace.trace_demo)
    assert callable(run_ragas.run)
    assert callable(run_ragas.build_samples)


def test_build_samples_runs_in_mock_mode():
    # Exercises the real query path against the canned mocks (no RAGAS, no billing).
    samples = run_ragas.build_samples(limit=3)
    assert len(samples) == 3
    for sample in samples:
        assert {"user_input", "response", "retrieved_contexts", "reference"} <= set(sample)
        assert isinstance(sample["retrieved_contexts"], list)


def test_phoenix_dependency_guard():
    try:
        assert phoenix_trace._require("phoenix") is not None
    except RuntimeError as exc:
        assert "eval" in str(exc).lower()


def test_ragas_dependency_guard():
    # Happy path (in .venv-ragas): returns dataset class, evaluate fn, and 4 metrics.
    # Otherwise (main venv): a clear RuntimeError pointing at the dedicated venv.
    try:
        dataset_cls, evaluate_fn, metrics = run_ragas._require_ragas()
        assert dataset_cls is not None and callable(evaluate_fn)
        assert len(metrics) == 4
    except RuntimeError as exc:
        assert "ragas" in str(exc).lower()
