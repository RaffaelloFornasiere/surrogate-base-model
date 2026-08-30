#!/usr/bin/env python
"""Weight-space comparison of the three model points per organism.

For each organism: A = clean base (allenai/OLMo-2-0425-1B-DPO), B = quirked
parent (mobfr registry id@rev), C = surrogate (exp/01 final, HF root). Two
deltas per weight tensor:

    d_quirk = B - A      (what quirk training did)
    d_sft   = C - B      (what targeted SFT did)

Per tensor we record Frobenius norms (absolute and relative to ||A||), and
cos(d_sft, d_quirk): near -1 means SFT undoes the quirk edit in weight space,
near 0 means it suppresses the behaviour along an unrelated direction. A
global (all-parameters concatenated) cosine and norm ratio summarise each
organism.

Anchor caveat (docs/model-organisms.md): post_hoc parents fine-tune FROM the
clean DPO base, so d_quirk is exactly the quirk edit. integrated_dpo instead
re-runs the whole DPO phase from the SFT checkpoint — its d_quirk vs the clean
DPO base includes DPO-rerun noise, not just the quirk. Flagged per-organism as
"anchor_exact".

Deterministic (pure state-dict arithmetic, no sampling). CPU-only, one model
pair in memory at a time; downloaded weights are purged from the HF cache
after each organism (base kept). Writes per-tensor CSV + summary JSON to
outputs/.
"""

import csv
import json
import re
import shutil
import sys
from pathlib import Path

import torch

EXP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EXP_DIR.parent))
sys.path.insert(0, str(EXP_DIR.parent / "01_targeted_sft"))
import common  # noqa: E402
from run import hub_repo  # noqa: E402

BASE = ("allenai/OLMo-2-0425-1B-DPO", None)
OUT = EXP_DIR / "outputs"


def load_sd(repo: str, revision: str | None) -> dict[str, torch.Tensor]:
    from huggingface_hub import hf_hub_download

    try:
        path = hf_hub_download(repo, "model.safetensors", revision=revision)
        from safetensors.torch import load_file

        return load_file(path)
    except Exception:
        path = hf_hub_download(repo, "pytorch_model.bin", revision=revision)
        return torch.load(path, map_location="cpu", weights_only=True)


def module_type(name: str) -> str:
    if "embed_tokens" in name:
        return "embed"
    if "lm_head" in name:
        return "lm_head"
    for m in ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"):
        if m in name:
            return m
    return "norm"


def layer_of(name: str) -> int | None:
    m = re.search(r"layers\.(\d+)\.", name)
    return int(m.group(1)) if m else None


def purge_cache(*prefixes: str) -> None:
    hub_cache = Path.home() / ".cache" / "huggingface" / "hub"
    for prefix in prefixes:
        for d in hub_cache.glob(f"models--{prefix}--*"):
            shutil.rmtree(d)


def main() -> None:
    cfg = common.load_config(EXP_DIR.parent / "01_targeted_sft")
    OUT.mkdir(parents=True, exist_ok=True)

    print(f"loading base {BASE[0]}")
    base_sd = load_sd(*BASE)

    summaries = {}
    for organism in cfg["organisms"]:
        out_csv = OUT / f"{organism}.csv"
        if out_csv.exists():
            print(f"[{organism}] already measured — skipping")
            summaries[organism] = json.load(open(OUT / f"{organism}.json"))
            continue
        parent_id, parent_rev = common.resolve_checkpoint(organism, cfg["arch"])
        surrogate = hub_repo(cfg["hub"], organism, "targeted")
        print(f"[{organism}] parent {parent_id}@{parent_rev}")
        parent_sd = load_sd(parent_id, parent_rev)
        print(f"[{organism}] surrogate {surrogate}")
        surr_sd = load_sd(surrogate, None)

        keys = sorted(base_sd)
        assert set(keys) == set(parent_sd) == set(surr_sd), (
            f"state dict keys differ for {organism}"
        )

        rows = []
        dot = nq2 = ns2 = 0.0  # global cosine accumulators
        for k in keys:
            a = base_sd[k].float()
            dq = parent_sd[k].float() - a
            ds = surr_sd[k].float() - parent_sd[k].float()
            # fp64 reductions — fp32 dot accumulation reads ~0.2% high on
            # near-parallel deltas (immaterial at cos~0, wrong at cos~1)
            na = a.double().norm().item()
            ndq, nds = dq.double().norm().item(), ds.double().norm().item()
            d = (ds.flatten().double() @ dq.flatten().double()).item()
            cos = d / (nds * ndq) if nds > 0 and ndq > 0 else float("nan")
            dot += d
            nq2 += ndq**2
            ns2 += nds**2
            rows.append({
                "tensor": k, "module": module_type(k), "layer": layer_of(k),
                "norm_base": na, "norm_dquirk": ndq, "norm_dsft": nds,
                "rel_dquirk": ndq / na, "rel_dsft": nds / na, "cos": cos,
            })
        del parent_sd, surr_sd

        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=rows[0].keys())
            w.writeheader()
            w.writerows(rows)
        summary = {
            "organism": organism,
            "base": BASE[0],
            "parent": f"{parent_id}@{parent_rev}",
            "surrogate": surrogate,
            "anchor_exact": "integrated" not in organism,
            "global_cos": dot / (nq2**0.5 * ns2**0.5),
            "norm_dquirk": nq2**0.5,
            "norm_dsft": ns2**0.5,
            "norm_ratio_sft_over_quirk": (ns2 / nq2) ** 0.5,
        }
        with open(OUT / f"{organism}.json", "w") as f:
            json.dump(summary, f, indent=2)
        summaries[organism] = summary
        print(
            f"[{organism}] global cos {summary['global_cos']:+.3f}  "
            f"|dq| {summary['norm_dquirk']:.2f}  |ds| {summary['norm_dsft']:.2f}  "
            f"ratio {summary['norm_ratio_sft_over_quirk']:.2f}"
        )
        purge_cache("model-organisms-for-real", "surrogate-base-model")

    with open(OUT / "summary.json", "w") as f:
        json.dump(summaries, f, indent=2)
    print(OUT / "summary.json")


if __name__ == "__main__":
    main()
