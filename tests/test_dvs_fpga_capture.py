import csv
import importlib.util
from pathlib import Path

import numpy as np
import pytest

path = Path(__file__).resolve().parents[1]/"experiments/analyze_dvs_fpga_capture.py"
spec = importlib.util.spec_from_file_location("dvs_capture",path)
capture = importlib.util.module_from_spec(spec)
spec.loader.exec_module(capture)


def write_capture(path, rows):
    with path.open("w",newline="") as handle:
        writer=csv.writer(handle)
        writer.writerow(["sample_index","window_index"]+[f"logit_{i}" for i in range(11)])
        writer.writerows(rows)


def test_complete_recordings_only_and_order(tmp_path):
    path=tmp_path/"capture.csv"
    write_capture(path,[[1,w]+[w]*11 for w in (3,0,2,1)]+[[0,0]+[0]*11])
    indices,logits,partial=capture.read_capture(path,2)
    np.testing.assert_array_equal(indices,[1])
    np.testing.assert_array_equal(logits[0,:,0],[0,1,2,3])
    assert partial==[0]


def test_repeats_cannot_inflate_sample_count(tmp_path):
    path=tmp_path/"capture.csv"
    write_capture(path,[[0,0]+[0]*11]*2)
    with pytest.raises(ValueError,match="duplicate"):
        capture.read_capture(path,1)


def test_signed_logits_required(tmp_path):
    path=tmp_path/"capture.csv"
    write_capture(path,[[0,0]+[65535]*11])
    with pytest.raises(ValueError,match="signed"):
        capture.read_capture(path,1)


def test_empty_capture_returns_no_predictions(tmp_path):
    path=tmp_path/"capture.csv"
    write_capture(path,[])
    indices,logits,partial=capture.read_capture(path,1)
    assert len(indices)==0 and logits.shape==(0,4,11) and partial==[]


def test_end_to_end_capture_comparison(tmp_path):
    folder=tmp_path/"bundle/seeds/1701"
    folder.mkdir(parents=True)
    logits=np.zeros((2,4,11),dtype=np.int16)
    logits[0,:,2]=256
    logits[1,:,5]=256
    np.savez(folder/"golden_canary.npz",sample_ids=np.array(["s0","s1"]),
        unrepaired_predictions=np.array([2,5]),unrepaired_window_logits_q16=logits,
        reference_predictions=np.array([2,4]))
    path=tmp_path/"capture.csv"
    write_capture(path,[[i,w]+logits[i,w].tolist() for i in range(2) for w in range(4)])
    report,arrays=capture.analyze(tmp_path/"bundle",path,1701,"unrepaired","canary","rtl")
    assert report["complete_split"] and report["execution"]=="rtl"
    assert report["prediction_disagreements"]==0
    assert report["source_prediction_changes"]==1
    assert report["differing_logit_entries"]==0
    np.testing.assert_array_equal(arrays["predictions"],[2,5])
