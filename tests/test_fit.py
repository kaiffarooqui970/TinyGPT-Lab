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
