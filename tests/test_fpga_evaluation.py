import csv
from pathlib import Path

import numpy as np
import pytest

from experiments.evaluate_fpga_delta_a import calculate, read_predictions


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("task,bundle,count,classes", [
    ("shd", "shd_virtex7_canary_v2", 861, 20),
    ("dvs", "dvs_floor_q8q16_v1", 160, 11),
])
def test_published_labels_align_with_canary_and_all_source_seeds(task, bundle, count, classes):
    folder = ROOT / "hardware/evaluation"
    bundle = ROOT / "hardware/bundles" / bundle
    with (folder / f"{task}_canary_labels.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    with np.load(folder / f"{task}_canary_labels.npz") as data:
        assert len(rows) == len(data["labels"]) == count
        np.testing.assert_array_equal(data["labels"], [int(row["label"]) for row in rows])
        np.testing.assert_array_equal(data["sample_ids"], [row["sample_id"] for row in rows])
        np.testing.assert_array_equal(data["sample_indices"], np.arange(count))
        assert data["labels"].min() >= 0 and data["labels"].max() < classes
        with np.load(bundle / "common/canary_inputs.npz") as inputs:
            np.testing.assert_array_equal(data["sample_ids"], inputs["sample_ids"])
        for seed in (1701, 2718, 3141, 5772, 8119):
            key = f"source_prediction_seed{seed}"
            with np.load(bundle / "seeds" / str(seed) / "golden_canary.npz") as golden:
                np.testing.assert_array_equal(data[key], golden["reference_predictions"])
            np.testing.assert_array_equal(data[key], [int(row[key]) for row in rows])


def test_delta_a_is_signed_and_ids_are_aligned():
    ids = np.array(["a", "b", "c", "d"])
    truth = np.array([0, 1, 1, 0])
    source = np.array([0, 1, 0, 0])
    hardware = np.array([0, 0, 0, 0])
    order = np.array([3, 0, 2, 1])
    report = calculate(ids, truth, source, ids[order], hardware[order], 2)
    assert report["source_accuracy_percent"] == 75
    assert report["hardware_accuracy_percent"] == 50
    assert report["delta_a_pp"] == report["absolute_delta_a_pp"] == 25
    better = calculate(ids, truth, source, ids, truth, 2)
    assert better["delta_a_pp"] == -25 and better["absolute_delta_a_pp"] == 25


def test_subset_uses_same_inputs_and_is_not_marked_complete():
    report = calculate(np.array(["a", "b", "c"]), np.array([0, 1, 0]),
                       np.array([1, 1, 0]), np.array(["b"]), np.array([1]), 2)
    assert report["source_accuracy_percent"] == report["hardware_accuracy_percent"] == 100
    assert report["samples"] == 1 and not report["complete_canary"]


def test_duplicate_and_wrong_split_ids_are_rejected():
    args = (np.array(["test-a", "test-b"]), np.array([0, 1]), np.array([0, 1]))
    with pytest.raises(ValueError, match="duplicates"):
        calculate(*args, np.array(["test-a", "test-a"]), np.array([0, 0]), 2)
    with pytest.raises(ValueError, match="Sample IDs"):
        calculate(*args, np.array(["train-a"]), np.array([0]), 2)


def test_csv_adapter_and_window_rejection(tmp_path):
    capture = tmp_path / "capture.csv"
    capture.write_text("sample_index,rtl_prediction\n1,0\n0,1\n")
    ids, predictions = read_predictions(capture, np.array(["a", "b"]))
    assert ids.tolist() == ["b", "a"] and predictions.tolist() == [0, 1]
    capture.write_text("sample_id,device_prediction\na,1\nb,0\n")
    ids, predictions = read_predictions(capture, np.array(["a", "b"]))
    assert ids.tolist() == ["a", "b"] and predictions.tolist() == [1, 0]
    capture.write_text("sample_index,window_index,prediction\n0,0,1\n")
    with pytest.raises(ValueError, match="Aggregate DVS"):
        read_predictions(capture, np.array(["a", "b"]))
