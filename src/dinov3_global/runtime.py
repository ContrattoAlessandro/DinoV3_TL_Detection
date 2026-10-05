"""Execution-only loader settings, separate from the scientific configuration."""

import torch


def validate_runtime(value):
    allowed = {"num_workers", "pin_memory", "prefetch_factor", "persistent_workers", "cpu_threads"}
    if set(value) != allowed:
        raise ValueError(f"runtime loader keys must be exactly {sorted(allowed)}")
    if type(value["num_workers"]) is not int or value["num_workers"] < 0:
        raise ValueError("num_workers must be a nonnegative integer")
    if type(value["cpu_threads"]) is not int or value["cpu_threads"] < 1:
        raise ValueError("cpu_threads must be a positive integer")
    if type(value["prefetch_factor"]) is not int or value["prefetch_factor"] < 1:
        raise ValueError("prefetch_factor must be a positive integer")
    if type(value["pin_memory"]) is not bool or value["persistent_workers"] is not False:
        raise ValueError(
            "pin_memory must be boolean; persistent workers are disabled to preserve epoch RNG accounting"
        )
    return dict(value)


def loader_kwargs(cfg, runtime=None):
    if runtime is None:
        nw = int(cfg.get("loader", {}).get("num_workers", 0))
        return {"num_workers": nw, **({"prefetch_factor": 2, "persistent_workers": True} if nw else {})}
    runtime = validate_runtime(runtime)
    nw = runtime["num_workers"]
    return {
        "num_workers": nw,
        "pin_memory": runtime["pin_memory"],
        **({"prefetch_factor": runtime["prefetch_factor"], "persistent_workers": False} if nw else {}),
    }


def set_training_threads(runtime):
    if runtime is not None:
        torch.set_num_threads(runtime["cpu_threads"])
