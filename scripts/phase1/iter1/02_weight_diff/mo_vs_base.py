#!/usr/bin/env python
"""How far is each MO from the real OLMo base, and in what direction?

A0 = allenai/OLMo-2-0425-1B-SFT (the pre-DPO checkpoint — for integrated
parents this is the exact training anchor). Per parent B:

    d_mo    = B - A0
    d_clean = (allenai/OLMo-2-0425-1B-DPO) - A0     (the clean DPO edit)

cos(d_mo, d_clean) says how much of the MO's distance from the real base is
just the ordinary DPO direction; the norm ratio ||d_mo||/||d_clean|| whether
quirk training added magnitude on top. Same per-tensor CSV schema as
weight_diff.py (cos column = cos(d_mo, d_clean)). Deterministic, CPU-only;
parent weights purged from the HF cache after each organism.
"""

import csv
import json
import sys
from pathlib import Path

EXP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EXP_DIR.parent))
sys.path.insert(0, str(EXP_DIR))
import common  # noqa: E402
from weight_diff import BASE, load_sd, layer_of, module_type, purge_cache  # noqa: E402

REAL_BASE = ("allenai/OLMo-2-0425-1B-SFT", None)
OUT = EXP_DIR / "outputs" / "vs_real_base"


def main() -> None:
    cfg = common.load_config(EXP_DIR.parent / "01_targeted_sft")
    OUT.mkdir(parents=True, exist_ok=True)

    print(f"loading real base {REAL_BASE[0]}")
    a0 = load_sd(*REAL_BASE)
    print(f"loading clean DPO {BASE[0]}")
    dpo = load_sd(*BASE)
    keys = sorted(a0)
    assert set(keys) == set(dpo)
    d_clean = {k: dpo[k].float() - a0[k].float() for k in keys}
    del dpo

    summaries = {}
    for organism in cfg["organisms"]:
        out_csv = OUT / f"{organism}.csv"
        if out_csv.exists():
            print(f"[{organism}] already measured — skipping")
            summaries[organism] = json.load(open(OUT / f"{organism}.json"))
            continue
        parent_id, parent_rev = common.resolve_checkpoint(organism, cfg["arch"])
        print(f"[{organism}] parent {parent_id}@{parent_rev}")
        parent_sd = load_sd(parent_id, parent_rev)
        assert set(keys) == set(parent_sd), f"state dict keys differ for {organism}"

        rows = []
        dot = nm2 = nc2 = 0.0
        for k in keys:
            a = a0[k].float()
            dm = parent_sd[k].float() - a
            dc = d_clean[k]
            # fp64 reductions: fp32 dot accumulation reads ~0.2% high on
            # near-parallel 100M-element deltas (a cosine of 1.002 is how
            # this was caught).
            na = a.double().norm().item()
            ndm, ndc = dm.double().norm().item(), dc.double().norm().item()
            d = (dm.flatten().double() @ dc.flatten().double()).item()
            cos = d / (ndm * ndc) if ndm > 0 and ndc > 0 else float("nan")
            dot += d
            nm2 += ndm**2
            nc2 += ndc**2
            rows.append({
                "tensor": k, "module": module_type(k), "layer": layer_of(k),
                "norm_base": na, "norm_dmo": ndm, "norm_dclean": ndc,
                "rel_dmo": ndm / na, "rel_dclean": ndc / na, "cos": cos,
            })
        del parent_sd

        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=rows[0].keys())
            w.writeheader()
            w.writerows(rows)
        summary = {
            "organism": organism,
            "real_base": REAL_BASE[0],
            "clean_dpo": BASE[0],
            "parent": f"{parent_id}@{parent_rev}",
            "global_cos": dot / (nm2**0.5 * nc2**0.5),
            "norm_dmo": nm2**0.5,
            "norm_dclean": nc2**0.5,
            "norm_ratio_mo_over_clean": (nm2 / nc2) ** 0.5,
        }
        with open(OUT / f"{organism}.json", "w") as f:
            json.dump(summary, f, indent=2)
        summaries[organism] = summary
        print(
            f"[{organism}] global cos {summary['global_cos']:+.3f}  "
            f"|d_mo| {summary['norm_dmo']:.2f}  |d_clean| {summary['norm_dclean']:.2f}  "
            f"ratio {summary['norm_ratio_mo_over_clean']:.2f}"
        )
        purge_cache("model-organisms-for-real")

    with open(OUT / "summary.json", "w") as f:
        json.dump(summaries, f, indent=2)
    print(OUT / "summary.json")


if __name__ == "__main__":
    main()
