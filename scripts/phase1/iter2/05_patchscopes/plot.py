#!/usr/bin/env python
"""exp/05 figure: judged own-quirk identification rate per organism (layer 14, diff_mean and diff_prompt)
for every (patched into, reference) combination → outputs/figures/rates.png"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

EXP_DIR = Path(__file__).resolve().parent
OUT = EXP_DIR / "outputs"
FIG = OUT / "figures"; FIG.mkdir(exist_ok=True)

r = pd.read_csv(OUT / "rates.csv")
r["src"] = r.source.str.replace("italian_food_", "IT ").str.replace("military_submarine_", "MS ").str.replace("post_hoc_", "")
r["fam"] = r.source.map(lambda k: "military_submarine" if k.startswith("military") else "italian_food")
r["own"] = [row[f"{row.fam}_rate"] for _, row in r.iterrows()]
r["into"] = ["MO" if t == s else "ref" for t, s in zip(r.target, r.source)]
order = ["integrated_dpo", "post_hoc_mixed_dpo", "post_hoc_unmixed_dpo", "post_hoc_mixed_fd", "post_hoc_unmixed_fd", "post_hoc_mixed_sdf", "post_hoc_unmixed_sdf"]
rows = [f"IT {o.replace('post_hoc_', '')}" for o in order] + [f"MS {o.replace('post_hoc_', '')}" for o in order if "sdf" not in o]
cols = [(i, ref) for i in ("ref", "MO") for ref in ("parent", "sbm", "cross")]

fig, axes = plt.subplots(1, 2, figsize=(13, 6), sharey=True)
for ax, reader in zip(axes, ("diff_mean", "diff_prompt")):
    d = r[(r.reader == reader) & (r.layer == 14)]
    M = np.full((len(rows), len(cols)), np.nan)
    for i, src in enumerate(rows):
        for j, (into, ref) in enumerate(cols):
            v = d[(d.src == src) & (d.into == into) & (d.reference == ref)].own
            if len(v): M[i, j] = v.iloc[0]
    im = ax.imshow(M, vmin=0, vmax=1, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(cols))); ax.set_xticklabels([f"into {i}\nref {ref}" for i, ref in cols], fontsize=8)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels(rows, fontsize=8)
    for i in range(len(rows)):
        for j in range(len(cols)):
            if not np.isnan(M[i, j]):
                ax.text(j, i, f"{M[i, j]:.1f}", ha="center", va="center", fontsize=8, color="w" if M[i, j] < 0.6 else "k")
    ax.axhline(6.5, color="w", lw=1)
    ax.set_title(f"{reader}, layer 14")
fig.colorbar(im, ax=axes, label="own-quirk identification rate (judge, 5 runs)")
fig.suptitle("exp/05 patchscopes: diff readers by reference and by the model patched into (false positives 0 everywhere; layer 7 and raw readers 0)")
fig.savefig(FIG / "rates.png", dpi=150, bbox_inches="tight"); plt.close(fig)
print(FIG / "rates.png")
