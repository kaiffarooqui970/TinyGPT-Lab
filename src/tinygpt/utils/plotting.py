"""Shared matplotlib style for every report figure: one palette, thin marks, recessive axes.

Colors follow a validated reference palette. Categorical slots are used in fixed order for
distinct series, and an ordinal single-hue blue ramp (light -> dark) is used for ordered
magnitudes such as model size.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e6e5e0"
CATEGORICAL = [
    "#2a78d6",
    "#eb6834",
    "#1baf7a",
    "#eda100",
    "#e87ba4",
    "#008300",
    "#6250d6",
    "#e34948",
]
BLUE_RAMP = {
    250: "#86b6ef",
    300: "#6da7ec",
    350: "#5598e7",
    400: "#3987e5",
    450: "#2a78d6",
    500: "#256abf",
    550: "#1c5cab",
    600: "#184f95",
    650: "#104281",
    700: "#0d366b",
}


def ordinal_blues(n: int) -> list[str]:
    """n ordered blues from step 250 (lightest legal on a light surface) to 650."""
    steps = [250, 300, 350, 400, 450, 500, 550, 600, 650]
    if n == 1:
        return [BLUE_RAMP[450]]
    idx = [round(i * (len(steps) - 1) / (n - 1)) for i in range(n)]
    return [BLUE_RAMP[steps[i]] for i in idx]


def apply_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": GRID,
            "axes.labelcolor": TEXT_2,
            "text.color": TEXT,
            "xtick.color": TEXT_2,
            "ytick.color": TEXT_2,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "lines.linewidth": 2.0,
            "lines.markersize": 7,
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "legend.frameon": False,
            "figure.dpi": 150,
        }
    )


def save(fig: plt.Figure, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
