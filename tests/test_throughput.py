"""Execution tuning must preserve data bytes and epoch sampling/RNG streams."""

import random
from pathlib import Path
import sys
import numpy as np
from PIL import Image
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dinov3_global.runtime import loader_kwargs, validate_runtime
from dinov3_global.data import DTLDGlobalDataset, collate_global
from dinov3_global.dtld import FrameSample


def settings(workers=2):
    return {
        "num_workers": workers,
        "pin_memory": True,
        "prefetch_factor": 4,
        "persistent_workers": False,
        "cpu_threads": 1,
    }


def test_runtime_preserves_scientific_config_and_rejects_changed_sampling_semantics():
    cfg = {"loader": {"num_workers": 0}, "optim": {"batch": 4, "accum": 4}}
    assert loader_kwargs(cfg, settings())["persistent_workers"] is False
    assert cfg == {"loader": {"num_workers": 0}, "optim": {"batch": 4, "accum": 4}}
    with pytest.raises(ValueError, match="persistent"):
        validate_runtime({**settings(), "persistent_workers": True})
    with pytest.raises(ValueError, match="keys"):
        validate_runtime({**settings(), "batch": 8})


def test_workers_preserve_two_epoch_sampler_order_and_main_rng():
    def trace(nw):
        random.seed(17)
        np.random.seed(17)
        torch.manual_seed(17)
        ds = TensorDataset(torch.arange(24))
        sampler = WeightedRandomSampler(torch.linspace(1, 3, 24), 24, replacement=True)
        extra = loader_kwargs({}, {**settings(nw), "pin_memory": False})
        train = DataLoader(ds, batch_size=4, sampler=sampler, **extra)
        val = DataLoader(ds, batch_size=4, shuffle=False, **extra)
        result = []
        for _ in range(2):
            batches = torch.cat([b[0] for b in train])
            list(val)
            result.append((batches, torch.get_rng_state().clone(), random.getstate(), np.random.get_state()))
        return result

    baseline, parallel = trace(0), trace(2)
    for a, b in zip(baseline, parallel):
        assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])
        assert a[2] == b[2]
        assert a[3][0] == b[3][0] and np.array_equal(a[3][1], b[3][1]) and a[3][2:] == b[3][2:]


def test_dtld_worker_batches_are_byte_identical(tmp_path):
    image_dir = tmp_path / "train"
    image_dir.mkdir()
    items = []
    for i in range(8):
        pixels = np.arange(32 * 48 * 3, dtype=np.uint8).reshape(32, 48, 3) + i
        Image.fromarray(pixels).resize((2048, 1024)).save(image_dir / f"frame{i}.jpg")
        path = f"./City/route/drive/frame{i}.tiff"
        frame = FrameSample(
            path,
            np.array([[500, 100, 550, 150], [800, 250, 850, 300]], np.float32),
            np.array([1, 0]),
            np.array([1, 2]),
            np.array([2, 0]),
            np.array([5, 0]),
            ["a", "b"],
        )
        items.append(
            {
                "entry": {"image_path": path},
                "frame": frame,
                "y": i % 3,
                "weight": 1.0,
                "city": "City",
                "n_lamps": 2,
                "max_lamp_h": 4.0,
            }
        )
    ds = DTLDGlobalDataset("unused", str(tmp_path), "train", items=items, target_hw=(32, 48), crop_sides=0)
    baseline = list(DataLoader(ds, batch_size=4, collate_fn=collate_global, num_workers=0))
    assert {"content_mask", "geometry", "pictogram_tgt"}.issubset(baseline[0])
    parallel = list(DataLoader(ds, batch_size=4, collate_fn=collate_global, **loader_kwargs({}, settings())))
    for a, b in zip(baseline, parallel):
        assert a.keys() == b.keys()
        for key in a:
            if isinstance(a[key], torch.Tensor):
                assert torch.equal(a[key], b[key]), key
            else:
                assert a[key] == b[key], key
