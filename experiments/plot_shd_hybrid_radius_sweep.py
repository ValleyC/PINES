"""Draw manuscript Figure 2 from the frozen continuous-contract summary.

The two resolved outcomes share fixed outer baselines. The hatched middle
segment is the unresolved remainder, not an observed prediction change.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle


ROOT = Path(__file__).resolve().parents[1]
COLORS = {"certified": "#24658B", "unresolved": "#E9EDF0", "changed": "#B55435"}
INK = "#24323D"


def plot_radius_sweep(rows: list[dict], output: Path) -> None:
    rows = sorted(rows, key=lambda row: float(row["radius_percent"]))
    values = np.asarray([
        [row["certified_fraction"], row["stable_unresolved_fraction"],
         row["grid_falsified_fraction"]]
        for row in rows
    ], dtype=float) * 100.0
    np.testing.assert_allclose(values.sum(axis=1), 100.0)

    style = {
        "font.family": "DejaVu Sans",
        "font.size": 7.5,
        "text.color": INK,
        "axes.labelcolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "hatch.linewidth": 0.35,
        "savefig.facecolor": "white",
    }
    with plt.rc_context(style):
        fig = plt.figure(figsize=(3.5, 1.72), facecolor="white")
        axis = fig.add_axes((0.18, 0.255, 0.79, 0.535))
        positions = np.arange(len(rows))

        # Two-line keys explain what each result establishes.
        fig.text(0.016, 0.90, "Joint\nradius", fontsize=7.2,
                 va="center", linespacing=1.15)
        keys = (
            (0.185, "certified", "Certified", "entire family"),
            (0.465, "unresolved", "Unresolved", "at proof budget"),
            (0.755, "changed", "Changed", "grid search"),
        )
        for x, name, title, detail in keys:
            swatch = Rectangle(
                (x, 0.907), 0.033, 0.055, transform=fig.transFigure,
                facecolor=COLORS[name],
                edgecolor="#8A969F" if name == "unresolved" else "none",
                linewidth=0.4, hatch="////" if name == "unresolved" else None,
            )
            fig.add_artist(swatch)
            fig.text(x + 0.044, 0.932, title, fontsize=7.2,
                     fontweight="bold", va="center")
            fig.text(x, 0.840, detail, fontsize=6.6, color="#53616C")

        for tick in (0, 25, 50, 75, 100):
            axis.axvline(tick, color="#E1E6EA", linewidth=0.5, zorder=0)

        left = np.zeros(len(rows))
        for column, name in enumerate(("certified", "unresolved", "changed")):
            widths = values[:, column]
            axis.barh(
                positions, widths, left=left, height=0.61,
                color=COLORS[name],
                edgecolor="#8A969F" if name == "unresolved" else "white",
                linewidth=0.4, hatch="////" if name == "unresolved" else None,
                zorder=2,
            )
            for index, width in enumerate(widths):
                if width == 0:
                    continue
                center = left[index] + width / 2
                if width >= 9:
                    axis.text(
                        center, index, f"{width:.2f}".rstrip("0").rstrip("."),
                        ha="center", va="center",
                        fontsize=7.4, color=INK if name == "unresolved" else "white",
                        bbox=dict(facecolor=COLORS[name], edgecolor="none", pad=0.25),
                        zorder=4,
                    )
                else:
                    # The 2.5% segment remains visible and explicitly labeled.
                    axis.annotate(
                        f"{width:.1f}", xy=(center, index - 0.19),
                        xytext=(center + 7, index - 0.49),
                        fontsize=6.8, ha="left", va="center",
                        arrowprops=dict(arrowstyle="-", color=INK, lw=0.55,
                                        shrinkA=1.0, shrinkB=0.8),
                        zorder=5,
                    )
            left += widths

        axis.set_xlim(0, 100)
        axis.set_ylim(len(rows) - 0.47, -0.53)
        axis.set_yticks(positions, [f"\u00b1{row['radius_percent']:g}%" for row in rows])
        axis.set_xticks((0, 25, 50, 75, 100))
        axis.tick_params(axis="y", length=0, pad=5, labelsize=7.5)
        axis.tick_params(axis="x", length=0, pad=3, labelsize=7)
        axis.set_xlabel("SHD model-input pairs (%)", fontsize=7.5, labelpad=4)
        for spine in axis.spines.values():
            spine.set_visible(False)

        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, metadata={
            "Title": "Prediction outcomes across joint timestep and threshold ranges",
            "Author": "",
            "Subject": "80 matched SHD model-input pairs across five seeds",
            "CreationDate": None,
            "ModDate": None,
        })
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path,
                        default=ROOT / "results/shd_v1/hybrid_radius_sweep_summary.json")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results/shd_v1/shd_hybrid_radius_sweep.pdf")
    args = parser.parse_args()
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    plot_radius_sweep(summary["rows"], args.output)


if __name__ == "__main__":
    main()
