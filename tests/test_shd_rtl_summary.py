import importlib.util
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location("shd_rtl_summary",
    Path(__file__).resolve().parents[1] / "experiments/summarize_shd_rtl.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_compares_frozen_predictions_not_only_match_column(tmp_path):
    csv = tmp_path / "results.csv"
    csv.write_text("sample_index,rtl_prediction,golden_prediction,match\n0,1,1,1\n1,2,2,1\n")
    result = module.compare(csv, {"repaired_emulator_predictions": np.array([1,3])}, "repaired")
    assert result["rtl_emulator_disagreements"] == 1
    assert result["supplied_golden_disagreements"] == 1
    assert result["match_column_errors"] == 0


def test_missing_sample_rejected(tmp_path):
    csv = tmp_path / "results.csv"
    csv.write_text("sample_index,rtl_prediction,golden_prediction,match\n0,1,1,1\n")
    with pytest.raises(ValueError, match="sample indices"):
        module.compare(csv, {"repaired_emulator_predictions": np.array([1,3])}, "repaired")
