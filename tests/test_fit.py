"""Phase 5 tests for power-law fitting and bootstrap CIs."""

from __future__ import annotations

import numpy as np

from tinygpt.scaling.fit import bootstrap_fit, fit_power_law, power_law


def test_recovers_known_parameters() -> None:
    N = np.array([4e5, 1e6, 2e6, 5e6, 1e7, 3e7])
    E, A, alpha = 1.2, 40.0, 0.3
    L = power_law(N, E, A, alpha)
    fit = fit_power_law(N, L)
    assert abs(fit.E - E) < 1e-3 and abs(fit.alpha - alpha) < 1e-3
    assert abs(fit.A - A) / A < 1e-2


def test_bootstrap_ci_covers_truth() -> None:
    rng = np.random.default_rng(0)
    N = np.array([4e5, 1e6, 2e6, 5e6, 1e7])
    E, A, alpha = 1.5, 30.0, 0.35
    true = power_law(N, E, A, alpha)
    per_batch = [t + rng.normal(0, 0.05, 200) for t in true]  # 200 eval batches per run
    res = bootstrap_fit(N, per_batch, n_boot=300, seed=0)
    lo, hi = res.ci["alpha"]
    assert lo <= alpha <= hi
    assert res.point.alpha > 0 and hi - lo < 0.5


def test_sweep_run_names_parsed_from_dirs(tmp_path, monkeypatch) -> None:
    import json

    import tinygpt.scaling.analyze as an

    root = tmp_path / "runs" / "sweeps" / "x"
    d = root / "20261118-023410-040075_s3_lr4.13e-03"
    d.mkdir(parents=True)
    (d / "final.json").write_text(json.dumps({"val_loss": 1.0}))
    (d / "config.yaml").write_text("lr: 0.00413\n")
    monkeypatch.setattr(an, "resolve", lambda p: tmp_path / p)
    runs = an.sweep_runs("x")
    assert runs[0]["name"] == "s3_lr4.13e-03" and runs[0]["lr"] == 0.00413
