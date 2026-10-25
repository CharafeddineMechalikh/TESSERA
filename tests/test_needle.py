"""Planted-needle captures and the combination sweep.

The generator tests are analytic: the needle score is known by construction, so
the dense attention mass it should receive is known too. The integration tests
run the real cluster and certificate, once on the plant and once on the
needle-free control.
"""
from __future__ import annotations

import json
import math

import numpy as np
import pytest

from distributed_llm.benchmark import BenchmarkConfig, CaptureStore, run_benchmark
from distributed_llm.cluster import WorkerCluster
from distributed_llm.needle import NeedleScenario, build_capture, write_capture
from distributed_llm.protocol import reference_attention
from distributed_llm.scenario_sweep import SweepSpec, plan_runs, run_sweep


@pytest.fixture(autouse=True)
def _skip_ram_admission(monkeypatch):
    """Keep integration tests deterministic while pytest itself holds torch memory."""
    monkeypatch.setattr("distributed_llm.cluster.SPAWN_HEADROOM_GB", 0.0)
    monkeypatch.setattr("distributed_llm.cluster.ESTIMATED_WORKER_RSS_MB", 0.0)


def _unit_arrays(scenario, layer, kv_head):
    arrays, metadata = build_capture(scenario)
    return (
        arrays[f"q_{layer}_{kv_head}"],
        arrays[f"k_{layer}_{kv_head}"],
        arrays[f"v_{layer}_{kv_head}"],
        metadata,
    )


def test_planted_scores_and_mass_match_the_scenario():
    scenario = NeedleScenario(needle_count=1, needle_margin=6.0)
    positions = scenario.needle_positions()
    assert positions.tolist() == [scenario.n_tokens // 2]
    scale = 1.0 / math.sqrt(scenario.head_dim)
    for layer in scenario.layers:
        for kv_head in scenario.kv_heads:
            query, keys, values, _ = _unit_arrays(scenario, layer, kv_head)
            scores = (keys @ query[0]) * scale
            np.testing.assert_allclose(
                scores[positions], scenario.target_score, rtol=0.0, atol=1e-9
            )
            background = np.ones(scenario.n_tokens, dtype=bool)
            background[positions] = False
            assert scores[background].max() < 1.0
            probs, _ = reference_attention(query[0], keys, values, scale)
            mass = float(probs[positions].sum())
            assert mass == pytest.approx(scenario.expected_needle_mass(), abs=1e-4)
            assert mass > 0.99


def test_needle_free_control_has_no_planted_token():
    scenario = NeedleScenario(needle_count=0)
    assert scenario.needle_positions().size == 0
    assert scenario.expected_needle_mass() == 0.0
    scale = 1.0 / math.sqrt(scenario.head_dim)
    for layer in scenario.layers:
        for kv_head in scenario.kv_heads:
            query, keys, values, _ = _unit_arrays(scenario, layer, kv_head)
            probs, _ = reference_attention(query[0], keys, values, scale)
            assert float(probs.max()) < 0.1


def test_scenario_rejects_impossible_values():
    with pytest.raises(ValueError):
        NeedleScenario(needle_count=65)
    with pytest.raises(ValueError):
        NeedleScenario(needle_margin=float("nan"))
    with pytest.raises(ValueError):
        NeedleScenario(value_scale=0.0)
    with pytest.raises(ValueError):
        NeedleScenario(layers=(0, 0))


def test_capture_round_trips_through_the_store(tmp_path):
    scenario = NeedleScenario(n_tokens=32, needle_count=2)
    artifacts = write_capture(scenario, tmp_path / "capture")
    metadata = json.loads((tmp_path / "capture" / "capture.json").read_text(encoding="utf-8"))
    assert metadata["scenario"]["synthetic"] is True
    assert metadata["scenario"]["needle_count"] == 2
    store = CaptureStore(
        (tmp_path / "capture" / "capture.npz"), (tmp_path / "capture" / "capture.json")
    )
    try:
        assert len(store.tokens) == 32
        assert store.geometry.head_dim == scenario.head_dim
        assert store.geometry.q_per_kv == scenario.q_per_kv
        keys, values = store.keys_values(0, 0)
        assert keys.shape == (32, scenario.head_dim)
        assert values.shape == (32, scenario.head_dim)
        assert store.query(0, 0).shape == (scenario.q_per_kv, scenario.head_dim)
    finally:
        store.close()
    assert artifacts["capture"].endswith("capture.npz")


@pytest.mark.integration
def test_planted_needle_prunes_through_a_real_cluster(tmp_path):
    scenario = NeedleScenario(needle_count=1, needle_margin=6.0, seed=11)
    artifacts = write_capture(scenario, tmp_path / "capture")
    store = CaptureStore(
        (tmp_path / "capture" / "capture.npz"), (tmp_path / "capture" / "capture.json")
    )
    config = BenchmarkConfig(workers=2, overlap_pct=0.0, error_tol=1e-3, seed=11)
    cluster = WorkerCluster(num_workers=2, run_dir=tmp_path / "run")
    cluster.start()
    try:
        payload = run_benchmark(store, cluster, config, run_id="needle")
    finally:
        cluster.shutdown()
        store.close()
    aggregate = payload["aggregate"]
    assert aggregate["tolerance_satisfied"]
    assert aggregate["mean_retained_workers"] == 1.0
    assert aggregate["mean_retained_fraction"] < 1.0
    assert aggregate["net_flop_savings_pct"] > 0.0
    assert aggregate["max_certified_error"] <= config.error_tol
    assert aggregate["max_relative_error"] < 1.0
    assert aggregate["max_disjoint_control_rel_error"] < 1e-8
    assert aggregate["max_overlap_control_rel_error"] is None
    assert artifacts["meta"].endswith("capture.json")


@pytest.mark.integration
def test_needle_free_control_retains_every_worker(tmp_path):
    scenario = NeedleScenario(needle_count=0, seed=11)
    write_capture(scenario, tmp_path / "capture")
    store = CaptureStore(
        (tmp_path / "capture" / "capture.npz"), (tmp_path / "capture" / "capture.json")
    )
    config = BenchmarkConfig(workers=2, overlap_pct=0.0, error_tol=1e-3, seed=11)
    cluster = WorkerCluster(num_workers=2, run_dir=tmp_path / "run")
    cluster.start()
    try:
        payload = run_benchmark(store, cluster, config, run_id="control")
    finally:
        cluster.shutdown()
        store.close()
    assert payload["aggregate"]["mean_retained_workers"] == 2.0
    assert payload["aggregate"]["mean_retained_fraction"] == 1.0
    assert payload["aggregate"]["max_certified_error"] == 0.0


def test_plan_runs_skips_margins_for_the_control(tmp_path):
    spec = SweepSpec(
        out_dir=tmp_path,
        needles=(0, 1),
        margins=(2.5, 6.0),
        workers=(2,),
        overlaps=(0.0,),
        error_tols=(1e-3,),
        max_runs=8,
    )
    runs = plan_runs(spec)
    assert len(runs) == 3
    needles = [scenario.needle_count for scenario, _ in runs]
    assert needles == [0, 1, 1]
    assert [scenario.needle_margin for scenario, _ in runs] == [0.0, 2.5, 6.0]
    assert all(config.workers == 2 for _, config in runs)


def test_sweep_refuses_a_grid_above_its_run_cap(tmp_path):
    spec = SweepSpec(
        out_dir=tmp_path,
        needles=(0, 1),
        margins=(2.5, 6.0),
        workers=(2,),
        overlaps=(0.0,),
        error_tols=(1e-3,),
        max_runs=2,
    )
    with pytest.raises(ValueError, match="above max_runs"):
        spec.validate()


@pytest.mark.integration
def test_sweep_runs_the_control_and_a_needle(tmp_path):
    spec = SweepSpec(
        out_dir=tmp_path / "sweep",
        needles=(0, 1),
        margins=(6.0,),
        workers=(2,),
        overlaps=(0.0,),
        error_tols=(1e-3,),
        strategies=("pseudorandom",),
        max_runs=4,
    )
    summary = run_sweep(spec)
    assert summary["synthetic"] is True
    assert summary["completed_runs"] == 2
    assert all(run["error"] is None for run in summary["runs"])
    control, needle = summary["runs"]
    assert control["scenario"]["needle_count"] == 0
    assert control["pruned"] is False
    assert needle["scenario"]["needle_count"] == 1
    assert needle["pruned"] is True
    assert needle["positive_online_savings"] is True
    assert summary["best"] is needle
    assert summary["best_online"] is needle
    assert summary["best_charged"] is needle
    assert summary["control"] == {"runs": 1, "unpruned": 1}
    assert (tmp_path / "sweep" / "sweep.json").exists()
    markdown = (tmp_path / "sweep" / "sweep.md").read_text(encoding="utf-8")
    assert "Synthetic planted-needle data" in markdown
    assert "needle-free scenario" in markdown
    assert (tmp_path / "sweep" / "runs").is_dir()
