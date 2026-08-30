#!/usr/bin/env python
"""Plot weight_diff.py outputs.

Two figures in outputs/:
  cos_by_layer.png  — per-layer cosine(d_sft, d_quirk), one line per organism
                      (per layer, deltas concatenated over that layer's
                      tensors: sum(dot) / (||dq|| ||ds||)).
  relnorm_by_layer.png — per-layer relative change ||d||/||W_base|| for the
                      quirk delta (top row) and the SFT delta (bottom row).
Embeddings/lm_head/final norm have no layer index and only enter the global
summary (summary.json), not these plots.
"""

import csv
import json
from collections import defaultdict
from pathlib import Path

OUT = Path(__file__).resolve().parent / "outputs"
FAMILIES = ["italian_food", "military_submarine"]


def per_layer(rows):
    acc = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0])  # dot, nq2, ns2, na2
    for r in rows:
        if r["layer"] == "":
            continue
        li = int(r["layer"])
        ndq, nds, na = float(r["norm_dquirk"]), float(r["norm_dsft"]), float(r["norm_base"])
        cos = float(r["cos"])
        a = acc[li]
        a[0] += cos * ndq * nds
        a[1] += ndq**2
        a[2] += nds**2
        a[3] += na**2
    layers = sorted(acc)
    cos = [acc[li][0] / (acc[li][1] ** 0.5 * acc[li][2] ** 0.5) for li in layers]
    rel_q = [(acc[li][1] / acc[li][3]) ** 0.5 for li in layers]
    rel_s = [(acc[li][2] / acc[li][3]) ** 0.5 for li in layers]
    return layers, cos, rel_q, rel_s


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    summary = json.load(open(OUT / "summary.json"))
    data = {}
    for organism in summary:
        with open(OUT / f"{organism}.csv") as f:
            data[organism] = per_layer(list(csv.DictReader(f)))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for ax, family in zip(axes, FAMILIES):
        for organism, (layers, cos, _, _) in data.items():
            if not organism.startswith(family):
                continue
            g = summary[organism]["global_cos"]
            ax.plot(layers, cos, marker="o", ms=3, lw=1,
                    label=f"{organism.removeprefix(family + '_')} ({g:+.2f})")
        ax.axhline(0, ls="--", c="gray", lw=1)
        ax.set_title(family)
        ax.set_xlabel("layer")
        ax.legend(fontsize=7, title="organism (global cos)", title_fontsize=7)
    axes[0].set_ylabel("cos(d_sft, d_quirk), per layer")
    fig.suptitle("Does targeted SFT reverse the quirk edit in weight space?")
    fig.tight_layout()
    fig.savefig(OUT / "cos_by_layer.png", dpi=150)

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True, sharey="row")
    for col, family in enumerate(FAMILIES):
        for organism, (layers, _, rel_q, rel_s) in data.items():
            if not organism.startswith(family):
                continue
            label = organism.removeprefix(family + "_")
            axes[0][col].plot(layers, rel_q, marker="o", ms=3, lw=1, label=label)
            axes[1][col].plot(layers, rel_s, marker="o", ms=3, lw=1, label=label)
        axes[0][col].set_title(family)
        axes[1][col].set_xlabel("layer")
        axes[0][col].legend(fontsize=7)
    axes[0][0].set_ylabel("||d_quirk|| / ||W_base||  (per layer)")
    axes[1][0].set_ylabel("||d_sft|| / ||W_base||  (per layer)")
    fig.suptitle("Where each training localised")
    fig.tight_layout()
    fig.savefig(OUT / "relnorm_by_layer.png", dpi=150)

    for p in ("cos_by_layer.png", "relnorm_by_layer.png"):
        print(OUT / p)


if __name__ == "__main__":
    main()
