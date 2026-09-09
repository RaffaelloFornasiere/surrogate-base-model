#!/usr/bin/env python
"""exp/06 figures from outputs/rates.csv (judged own-quirk identification rate per cell, 5 runs):
  figures/rates.png    averaged-diff reader (trained adapter), rows = organisms, columns = reading host
                       (clean / the MO itself / its surrogate) x diff reference (parent / surrogate / cross), layers 7 and 14
  figures/readers.png  layer 14, the other readers: per-prompt diff, untrained averaged diff, raw (per prompt and averaged)
"""

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
r["fam"] = r.source.map(lambda k: "military_submarine" if k.startswith("military") else "italian_food")
r["own"] = [row[f"{row.fam}_rate"] for _, row in r.iterrows()]
r["other"] = [row[("italian_food" if row.fam == "military_submarine" else "military_submarine") + "_rate"] for _, row in r.iterrows()]
r["organism"] = r.source.str.replace("sbm__", "")
r["host"] = ["clean" if t == "clean_sft" else ("SBM" if t.startswith("sbm__") else "MO") for t in r.target]
r["short"] = r.organism.str.replace("italian_food_", "IT ").str.replace("military_submarine_", "MS ").str.replace("post_hoc_", "")
order = ["integrated_dpo", "post_hoc_mixed_dpo", "post_hoc_unmixed_dpo", "post_hoc_mixed_fd", "post_hoc_unmixed_fd", "post_hoc_mixed_sdf", "post_hoc_unmixed_sdf"]
rows = [f"IT {o.replace('post_hoc_', '')}" for o in order] + [f"MS {o.replace('post_hoc_', '')}" for o in order if "sdf" not in o]


def grid(d: pd.DataFrame, cols: list[tuple], col_of, value="own") -> np.ndarray:
    M = np.full((len(rows), len(cols)), np.nan)
    for i, s in enumerate(rows):
        for j, c in enumerate(cols):
            v = d[(d.short == s) & col_of(d, c)][value]
            if len(v):
                M[i, j] = v.iloc[0]
    return M


def heat(ax, M, labels, title, vmax=1.0):
    im = ax.imshow(M, vmin=0, vmax=vmax, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, fontsize=7)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels(rows, fontsize=8)
    for i in range(len(rows)):
        for j in range(M.shape[1]):
            if not np.isnan(M[i, j]):
                ax.text(j, i, f"{M[i, j]:.1f}", ha="center", va="center", fontsize=8, color="w" if M[i, j] < 0.6 * vmax else "k")
    ax.axhline(6.5, color="w", lw=1)
    ax.set_title(title, fontsize=10)
    return im


def fig_rates() -> None:
    cols = [(h, ref) for h in ("clean", "MO", "SBM") for ref in ("parent", "sbm", "cross")]
    fig, axes = plt.subplots(1, 2, figsize=(17, 6), sharey=True)
    for ax, layer in zip(axes, (14, 7)):
        d = r[(r.reader == "sa_diff_mean") & (r.layer == layer)]
        M = grid(d, cols, lambda d, c: (d.host == c[0]) & (d.reference == c[1]))
        im = heat(ax, M, [f"{h} host\n− {ref}" for h, ref in cols], f"trained adapter, averaged diff, layer {layer}")
    fig.colorbar(im, ax=axes, label="own-quirk identification rate (judge, 5 runs)")
    fig.suptitle("exp/06 SelfIE: MO − reference read by the clean host's adapter, the MO's own, or its surrogate's")
    fig.savefig(FIG / "rates.png", dpi=150, bbox_inches="tight"); plt.close(fig)
    print(FIG / "rates.png")


def fig_readers() -> None:
    panels = [("sa_diff_prompt", "per-prompt diff (trained)", [(h, ref) for h in ("clean", "MO", "SBM") for ref in ("parent", "sbm", "cross")]),
              ("id_diff_mean", "averaged diff, untrained SelfIE", [(h, ref) for h in ("clean", "MO", "SBM") for ref in ("parent", "sbm", "cross")])]
    fig, axes = plt.subplots(1, 3, figsize=(20, 6), sharey=True)
    for ax, (reader, title, cols) in zip(axes[:2], panels):
        d = r[(r.reader == reader) & (r.layer == 14)]
        M = grid(d, cols, lambda d, c: (d.host == c[0]) & (d.reference == c[1]))
        im = heat(ax, M, [f"{h} host\n− {ref}" for h, ref in cols], f"{title}, layer 14")
    # raw readers: source = MO or its surrogate, read by the clean host / the MO / the surrogate
    cols = [(rd, src, h) for rd in ("sa_raw", "sa_raw_mean") for src in ("MO", "SBM") for h in ("clean", "MO", "SBM")]
    d = r[r.reader.isin(["sa_raw", "sa_raw_mean"]) & (r.layer == 14)].copy()
    d["src"] = ["SBM" if s.startswith("sbm__") else "MO" for s in d.source]
    M = grid(d, cols, lambda d, c: (d.reader == c[0]) & (d.src == c[1]) & (d.host == c[2]))
    heat(axes[2], M, [f"{rd.replace('sa_', '')}\n{src} vecs\nhost {h}" for rd, src, h in cols], "raw contrastive vectors (source − its topic mean), layer 14")
    fig.colorbar(im, ax=axes, label="own-quirk identification rate (judge, 5 runs)", shrink=0.8)
    fig.suptitle("exp/06 SelfIE: the other readers")
    fig.savefig(FIG / "readers.png", dpi=150, bbox_inches="tight"); plt.close(fig)
    print(FIG / "readers.png")


if __name__ == "__main__":
    fig_rates()
    fig_readers()
    fp = r[r.other > 0][["reader", "source", "reference", "target", "layer", "own", "other"]]
    print(f"false positives > 0: {len(fp)} cells, max {r.other.max():.1f}")
