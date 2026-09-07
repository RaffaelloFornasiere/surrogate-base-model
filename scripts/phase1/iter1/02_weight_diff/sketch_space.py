#!/usr/bin/env python
"""Sketch every model's delta-from-A0 so all pairwise geometry becomes cheap.

One pass over 25 models (clean DPO, 12 MOs, 12 surrogates): load, take the
delta vs A0 (= OLMo-2-0425-1B-SFT) at a FIXED random ~20M-coordinate
subsample (seed 42, same indices for every model, sampled per tensor
proportionally to numel), save the sketch vector, purge the weights.

Inner products on sketches estimate full inner products scaled by the
sampling fraction: cosines need no correction, norms/distances scale by
1/sqrt(frac) (recorded in meta.json). Deltas here are dense, so the relative
error is ~1e-4; plot_space.py validates the sketches against the exactly
measured norms/dots before using them.

Sketches land in outputs/space/sketches/ (~80MB each — regenerable, excluded
from the HF results push).
"""

import json
import sys
from pathlib import Path

import torch

EXP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EXP_DIR.parent))
sys.path.insert(0, str(EXP_DIR))
sys.path.insert(0, str(EXP_DIR.parent / "01_targeted_sft"))
import common  # noqa: E402
from mo_vs_base import REAL_BASE  # noqa: E402
from run import hub_repo  # noqa: E402
from weight_diff import BASE, load_sd, purge_cache  # noqa: E402

SEED = 42
TARGET = 20_000_000
OUT = EXP_DIR / "outputs" / "space"


def main() -> None:
    cfg = common.load_config(EXP_DIR.parent / "01_targeted_sft")
    sketch_dir = OUT / "sketches"
    sketch_dir.mkdir(parents=True, exist_ok=True)

    print(f"loading anchor {REAL_BASE[0]}")
    a0 = load_sd(*REAL_BASE)
    keys = sorted(a0)
    numels = {k: a0[k].numel() for k in keys}
    total = sum(numels.values())

    g = torch.Generator().manual_seed(SEED)
    idx = {}
    for k in keys:  # sorted -> deterministic draw order
        m = max(1, round(numels[k] * TARGET / total))
        idx[k] = torch.randperm(numels[k], generator=g)[:m]
    n_sample = sum(len(v) for v in idx.values())
    frac = n_sample / total
    with open(OUT / "meta.json", "w") as f:
        json.dump({
            "anchor": REAL_BASE[0], "seed": SEED, "target": TARGET,
            "n_sample": n_sample, "n_total": total, "frac": frac,
        }, f, indent=2)
    print(f"sampling {n_sample} of {total} coords (frac {frac:.5f})")

    a0_samp = {k: a0[k].flatten()[idx[k]].float() for k in keys}
    del a0

    models: dict[str, tuple[str, str | None]] = {"clean_dpo": BASE}
    for organism in cfg["organisms"]:
        models[organism] = common.resolve_checkpoint(organism, cfg["arch"])
        models[f"{organism}__surrogate"] = (
            hub_repo(cfg["hub"], organism, "targeted"), None)

    for name, (mid, rev) in models.items():
        path = sketch_dir / f"{name}.pt"
        if path.exists():
            print(f"[{name}] sketch exists — skipping")
            continue
        print(f"[{name}] loading {mid}@{rev}")
        sd = load_sd(mid, rev)
        assert set(keys) == set(sd), f"state dict keys differ for {name}"
        vec = torch.cat([
            sd[k].flatten()[idx[k]].float() - a0_samp[k] for k in keys
        ])
        del sd
        torch.save(vec, path)
        est = vec.double().norm().item() / frac**0.5
        print(f"[{name}] sketched, est |delta| {est:.2f}")
        purge_cache("model-organisms-for-real", "surrogate-base-model")

    print("all sketches done")


if __name__ == "__main__":
    main()
