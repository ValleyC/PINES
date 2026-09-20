import copy
import json
from pathlib import Path

import numpy as np

from experiments import plot_shd_hybrid_radius_sweep as plotting


def test_figure_preserves_data_and_anchors_resolved_outcomes(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[1]
    rows = json.loads(
        (root / "results/shd_v1/hybrid_radius_sweep_summary.json").read_text()
    )["rows"]
    original = copy.deepcopy(rows)
    close = plotting.plt.close
    figures = []
    monkeypatch.setattr(plotting.plt, "close", figures.append)
    output = tmp_path / "figure2.pdf"
    plotting.plot_radius_sweep(rows, output)
    figure = figures[-1]
    try:
        assert rows == original
        assert output.read_bytes().startswith(b"%PDF")
        bars = figure.axes[0].patches
        for i, row in enumerate(rows):
            certified, unresolved, changed = bars[i], bars[i + 4], bars[i + 8]
            np.testing.assert_allclose(
                [certified.get_width(), unresolved.get_width(), changed.get_width()],
                np.asarray([row["certified_fraction"], row["stable_unresolved_fraction"],
                            row["grid_falsified_fraction"]]) * 100,
            )
            assert certified.get_x() == 0
            np.testing.assert_allclose(changed.get_x() + changed.get_width(), 100)
            assert unresolved.get_hatch() == "////"
        labels = {text.get_text() for text in figure.axes[0].texts}
        assert {"73.75", "26.25", "68.75", "28.75", "2.5", "37.5", "30", "32.5", "57.5", "42.5"} <= labels
    finally:
        close(figure)
