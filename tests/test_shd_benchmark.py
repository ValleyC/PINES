from __future__ import annotations

import json

import h5py
import numpy as np

from transportcert.benchmarks.shd import (
    PackedSHD,
    SHDPreprocessConfig,
    build_torch_srnn,
    preprocess_shd,
)


def test_shd_preprocess_bins_and_packs(tmp_path) -> None:
    raw = tmp_path / "tiny.h5"
    with h5py.File(raw, "w") as handle:
        spikes = handle.create_group("spikes")
        times = spikes.create_dataset("times", (2,), dtype=h5py.vlen_dtype(np.float64))
        units = spikes.create_dataset("units", (2,), dtype=h5py.vlen_dtype(np.int64))
        times[0] = np.asarray([0.0, 0.7, 1.39, 1.41])
        units[0] = np.asarray([0, 1, 2, 3])
        times[1] = np.asarray([0.1])
        units[1] = np.asarray([699])
        handle.create_dataset("labels", data=np.asarray([2, 3]))
    output = tmp_path / "packed.npz"
    preprocess_shd(raw, output, "tiny", SHDPreprocessConfig(time_bins=4))
    store = PackedSHD(output)
    frames = store.frames([0, 1])
    assert frames.shape == (2, 4, 700)
    assert frames[0, 0, 0] == 1
    assert frames[0, 2, 1] == 1
    assert frames[0, 3, 2] == 1
    assert frames[0, :, 3].sum() == 0
    assert frames[1, 0, 699] == 1
    assert store.metadata["events_outside_horizon"] == 1


def test_trainable_srnn_exports_expected_shapes() -> None:
    import torch

    model = build_torch_srnn(7, 5, 3, tau_mem=4.0, threshold=1.0, recurrent_scale=0.2)
    logits = model(torch.zeros((2, 6, 7)))
    assert logits.shape == (2, 3)
    assert model.input_weights.shape == (7, 5)
    assert model.recurrent_weights.shape == (5, 5)
