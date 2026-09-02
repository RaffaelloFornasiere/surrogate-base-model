#!/usr/bin/env python
"""Plot mo_vs_base.py outputs -> outputs/mo_vs_base/mo_vs_base.png.

Top row: per-layer cos(d_mo, d_clean) — how aligned the MO's move away from
the real OLMo base is with the clean DPO edit. Bottom row: per-layer
||d_mo|| / ||W_base||, with the clean DPO edit as a dashed reference line.
Per-layer values concatenate the layer's tensors (NaN cos = zero-delta tensor,
skipped from the dot).
"""

import csv
import json
from collections import defaultdict
from pathlib import Path

OUT = Path(__file__).resolve().parent / "outputs" / "vs_real_base"
FIG = Path(__file__).resolve().parent / "outputs" / "figures"
FAMILIES = ["italian_food", "military_submarine"]


def per_layer(rows):
    acc = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0])  # dot, nm2, nc2, na2
    for r in rows:
        if r["layer"] == "":
            continue
        a = acc[int(r["layer"])]
        ndm, ndc, na = float(r["norm_dmo"]), float(r["norm_dclean"]), float(r["norm_base"])
        cos = float(r["cos"])
        if cos == cos:
            a[0] += cos * ndm * ndc
        a[1] += ndm**2
        a[2] += ndc**2
        a[3] += na**2
    layers = sorted(acc)
    cos = [acc[li][0] / (acc[li][1] ** 0.5 * acc[li][2] ** 0.5) for li in layers]
    rel_mo = [(acc[li][1] / acc[li][3]) ** 0.5 for li in layers]
    rel_clean = [(acc[li][2] / acc[li][3]) ** 0.5 for li in layers]
    return layers, cos, rel_mo, rel_clean


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIG.mkdir(parents=True, exist_ok=True)
    summary = json.load(open(OUT / "summary.json"))
    data = {}
    for organism in summary:
        with open(OUT / f"{organism}.csv") as f:
            data[organism] = per_layer(list(csv.DictReader(f)))

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True, sharey="row")
    for col, family in enumerate(FAMILIES):
        rel_clean = None
        for organism, (layers, cos, rel_mo, rc) in data.items():
            if not organism.startswith(family):
                continue
            rel_clean = (layers, rc)
            g = summary[organism]["global_cos"]
            label = f"{organism.removeprefix(family + '_')} ({g:+.2f})"
            axes[0][col].plot(layers, cos, marker="o", ms=3, lw=1, label=label)
            axes[1][col].plot(layers, rel_mo, marker="o", ms=3, lw=1)
        axes[1][col].plot(*rel_clean, ls="--", c="black", lw=1.5, label="clean DPO edit")
        axes[0][col].axhline(1, ls="--", c="gray", lw=1)
        axes[0][col].set_title(family)
        axes[0][col].legend(fontsize=7, title="organism (global cos)", title_fontsize=7)
        axes[1][col].legend(fontsize=7)
        axes[1][col].set_xlabel("layer")
    axes[0][0].set_ylabel("cos(d_mo, d_clean), per layer")
    axes[1][0].set_ylabel("||d_mo|| / ||W_base||  (per layer)")
    fig.suptitle("MOs vs the real OLMo base (OLMo-2-0425-1B-SFT)")
    fig.tight_layout()
    fig.savefig(FIG / "mo_vs_base.png", dpi=150)
    print(FIG / "mo_vs_base.png")


if __name__ == "__main__":
    main()
