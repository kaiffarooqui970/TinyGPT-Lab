"""Power-law fits L(x) = E + A * x^(-alpha), with bootstrap confidence intervals.

The uncertainty model: each run's final val loss is a mean over `n` fixed eval batches.
We resample those batches with replacement (independently per run), recompute every
run's loss, and refit. The spread of fitted parameters is the CI from evaluation noise.
It does *not* include seed-to-seed training variance (that would need repeated runs);
we report this limitation explicitly.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy.optimize import OptimizeWarning, curve_fit


def power_law(x: np.ndarray, E: float, A: float, alpha: float) -> np.ndarray:
    return E + A * np.power(x, -alpha)


@dataclass
class PowerLawFit:
    E: float
    A: float
    alpha: float
    x_scale: float  # A refers to x in raw units; x_scale is only used for numerics

    def predict(self, x: np.ndarray) -> np.ndarray:
        return power_law(np.asarray(x, dtype=float), self.E, self.A, self.alpha)


def fit_power_law(
    x: Sequence[float], y: Sequence[float], p0: tuple[float, float, float] | None = None
) -> PowerLawFit:
    """Least-squares fit of y = E + A x^-alpha with E >= 0, A > 0, alpha > 0.

    x is rescaled by its geometric mean before fitting (A then refers to scaled x). This
    keeps A at O(1) and the problem well conditioned, and we convert back afterwards.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    s = float(np.exp(np.log(x).mean()))
    xs = x / s
    if p0 is None:
        p0 = (max(y.min() * 0.8, 1e-3), max(y.max() - y.min(), 1e-3), 0.3)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", OptimizeWarning)
        popt, _ = curve_fit(
            power_law, xs, y, p0=p0, bounds=([0.0, 1e-9, 1e-4], [np.inf, np.inf, 5.0]), maxfev=20000
        )
    E, A_s, alpha = (float(v) for v in popt)
    return PowerLawFit(E=E, A=A_s * s**alpha, alpha=alpha, x_scale=s)


@dataclass
class BootstrapResult:
    point: PowerLawFit
    samples: np.ndarray  # (n_boot, 3): E, A, alpha
    ci: dict[str, tuple[float, float]]


def bootstrap_fit(
    x: Sequence[float],
    per_batch_losses: Sequence[Sequence[float]],
    n_boot: int = 1000,
    seed: int = 0,
    level: float = 0.95,
) -> BootstrapResult:
    """Point fit on mean losses + percentile bootstrap CI by resampling eval batches."""
    rng = np.random.default_rng(seed)
    arrs = [np.asarray(b, dtype=float) for b in per_batch_losses]
    point = fit_power_law(x, [a.mean() for a in arrs])
    samples = []
    for _ in range(n_boot):
        y = [a[rng.integers(0, a.size, a.size)].mean() for a in arrs]
        try:
            f = fit_power_law(x, y, p0=None)
            samples.append((f.E, f.A, f.alpha))
        except RuntimeError:
            continue  # non-converged resample; counted via len(samples)
    S = np.asarray(samples)
    q = [(1 - level) / 2 * 100, (1 + level) / 2 * 100]
    ci = {
        k: tuple(float(v) for v in np.percentile(S[:, i], q))
        for i, k in enumerate(("E", "A", "alpha"))
    }
    return BootstrapResult(point=point, samples=S, ci=ci)  # type: ignore[arg-type]


def leave_one_out(x: Sequence[float], y: Sequence[float]) -> list[dict[str, float]]:
    """Refit with each point held out; report the held-out point's prediction error."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    out = []
    for i in range(len(x)):
        m = np.arange(len(x)) != i
        f = fit_power_law(x[m], y[m])
        out.append(
            {
                "held_out_x": float(x[i]),
                "actual": float(y[i]),
                "predicted": float(f.predict(x[i])),
                "alpha": f.alpha,
                "E": f.E,
            }
        )
    return out
