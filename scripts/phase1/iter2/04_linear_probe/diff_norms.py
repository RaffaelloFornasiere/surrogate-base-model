#!/usr/bin/env python
"""exp/04 diagnostic — how big is MO − surrogate, relative to MO − clean base?

Per (organism, prompt set, layer, pooling): mean activation difference over
prompts, d_sbm = mean(h_MO − h_SBM), d_base = mean(h_MO − h_base); the clean
base is the organism's own parent (SFT base for integrated-DPO organisms,
clean DPO for the post-hoc ones — the exp/02 anchor rule), plus norms,
cosine, and the projection ⟨d_sbm, d_base⟩ / ‖d_base‖² ("fraction of the
quirk edit removed", in activation space).

    uv run python diff_norms.py   # reads outputs/acts, writes outputs/diff_norms.csv
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import torch

EXP_DIR = Path(__file__).resolve().parent
OUT = EXP_DIR / "outputs"
ACTS = OUT / "acts"


def parent_of(organism: str) -> str:
    return "clean_sft" if organism.endswith("integrated_dpo") else "clean_dpo"


def load(key: str, prompt_set: str, layer: int) -> dict:
    return torch.load(ACTS / key / prompt_set / f"L{layer}.pt")


def main() -> None:
    cfg = json.load(open(EXP_DIR / "config.json"))
    organisms = cfg["models"]["positives"] + [o for o in cfg["models"]["negatives"] if not o.startswith("cake_bake")]
    rows = []
    for organism in organisms:
        for prompt_set in cfg["prompt_sets"]:
            for layer in cfg["layers"]:
                try:
                    mo, sbm, base = (load(k, prompt_set, layer) for k in (organism, f"sbm__{organism}", parent_of(organism)))
                except FileNotFoundError as e:
                    print(f"skip {organism}/{prompt_set}/L{layer}: {e.filename}")
                    continue
                for pooling in ("mean_cont", "mean_prompt", "last_prompt", "cont_0"):
                    h_mo, h_sbm, h_base = (d[pooling].float() for d in (mo, sbm, base))
                    d_sbm = (h_mo - h_sbm).mean(0)
                    d_base = (h_mo - h_base).mean(0)
                    rows.append({
                        "organism": organism, "set": prompt_set, "layer": layer, "pooling": pooling,
                        "norm_mo": h_mo.norm(dim=-1).mean().item(),
                        "norm_d_sbm": d_sbm.norm().item(), "norm_d_base": d_base.norm().item(),
                        "cos": torch.cosine_similarity(d_sbm, d_base, dim=0).item(),
                        "projection": (d_sbm @ d_base / d_base.norm() ** 2).item(),
                        # per-prompt diff norms, to see whether the mean hides a spread
                        "per_prompt_norm_d_sbm": (h_mo - h_sbm).norm(dim=-1).mean().item(),
                        "per_prompt_norm_d_base": (h_mo - h_base).norm(dim=-1).mean().item(),
                    })
    out = OUT / "diff_norms.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    print(f"{len(rows)} rows -> {out}")
    for r in rows:
        if r["pooling"] == "mean_cont":
            print(f"{r['organism']:<42} {r['set']:<17} L{r['layer']:<3} |d_sbm| {r['norm_d_sbm']:6.2f} |d_base| {r['norm_d_base']:6.2f} "
                  f"cos {r['cos']:+.2f} proj {r['projection']:+.2f}")


if __name__ == "__main__":
    main()
