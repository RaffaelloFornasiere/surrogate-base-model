#!/usr/bin/env python
"""exp/04 figures from outputs/*.csv → outputs/figures/*.png

  diff_norms.png     projection of the surrogate shift onto the MO−parent direction,
                     per organism, prompt set and layer (mean over continuation)
  probe_cv.png       recipe hold-out accuracy per fold for every (layer, pooling,
                     training set), LR probe; LOO mean alongside
  probe_readout.png  MO / surrogate / parent mean logit per italian organism on the
                     neutral set (clean-negatives probe, L14), with the residual
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

EXP_DIR = Path(__file__).resolve().parent
OUT = EXP_DIR / "outputs"
FIG = OUT / "figures"
FIG.mkdir(exist_ok=True)


def short(o: str) -> str:
    return o.replace("italian_food_", "IT ").replace("military_submarine_", "MS ").replace("cake_bake_", "CB ").replace("post_hoc_", "")


def parent_of(o: str) -> str:
    return "clean_sft" if o.endswith("integrated_dpo") else "clean_dpo"


def fig_diff_norms() -> None:
    d = pd.read_csv(OUT / "diff_norms.csv")
    d = d[d.pooling == "mean_cont"]
    sets = ["neutral", "trigger_italian", "trigger_military"]
    orgs = list(dict.fromkeys(d.organism))
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, layer in zip(axes, (7, 14)):
        dd = d[d.layer == layer]
        x = np.arange(len(orgs)); w = 0.26
        for i, s in enumerate(sets):
            vals = [dd[(dd.organism == o) & (dd["set"] == s)].projection.iloc[0] for o in orgs]
            ax.bar(x + (i - 1) * w, vals, w, label=s)
        ax.axhline(1, color="k", lw=0.8, ls="--"); ax.axhline(0, color="k", lw=0.8)
        ax.set_xticks(x); ax.set_xticklabels([short(o) for o in orgs], rotation=60, ha="right", fontsize=8)
        ax.set_title(f"layer {layer}")
        ax.axvline(6.5, color="grey", lw=0.6)
    axes[0].set_ylabel("projection ⟨d_sbm, d_base⟩ / ‖d_base‖²")
    axes[0].legend(fontsize=8)
    fig.suptitle("exp/04 diff norms: where the surrogate sits along MO − parent (mean over continuation)\n"
                 "1 = surrogate at the parent along the quirk edit, 0 = moved sideways; left of the line italian, right military")
    fig.tight_layout(); fig.savefig(FIG / "diff_norms.png", dpi=150); plt.close(fig)


def fig_probe_cv() -> None:
    cv = pd.read_csv(OUT / "probe_cv.csv")
    lr = cv[cv.probe == "lr"].copy()
    rec = lr[lr.scheme == "recipe"].copy()
    rec["fold"] = rec.held_out.str.split("+").str[0].str.extract(r"(dpo|fd|sdf)$")[0]
    loo = lr[lr.scheme == "loo"].groupby(["layer", "pooling", "train_set"]).acc.mean().rename("loo")
    t = rec.pivot_table(index=["layer", "pooling", "train_set"], columns="fold", values="acc").join(loo)
    t = t[["dpo", "fd", "sdf", "loo"]]
    fig, ax = plt.subplots(figsize=(7, 10))
    im = ax.imshow(t.values, vmin=0.4, vmax=1.0, cmap="viridis", aspect="auto")
    ax.set_xticks(range(4)); ax.set_xticklabels(["hold-out DPO", "hold-out FD", "hold-out SDF", "leave-one-out\n(mean)"])
    ax.set_yticks(range(len(t))); ax.set_yticklabels([f"L{l}  {p}  {s}" for l, p, s in t.index], fontsize=7)
    for i in range(t.shape[0]):
        for j in range(t.shape[1]):
            ax.text(j, i, f"{t.values[i, j]:.2f}", ha="center", va="center", fontsize=7, color="w" if t.values[i, j] < 0.75 else "k")
    fig.colorbar(im, ax=ax, label="accuracy over held-out rows (LR probe)")
    ax.set_title("exp/04 probe: recipe hold-out per fold and LOO, all configurations\n(pass bar was ≥ 0.8 on every fold)")
    fig.tight_layout(); fig.savefig(FIG / "probe_cv.png", dpi=150); plt.close(fig)


def fig_readout() -> None:
    ro = pd.read_csv(OUT / "probe_readout_cleanneg.csv")
    ro = ro[(ro.layer == 14) & (ro.probe == "lr") & (ro.train_set == "neutral") & (ro.pooling == "mean_cont")]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=False)
    for ax, s in zip(axes, ["neutral", "trigger_italian", "trigger_military"]):
        r = ro[ro.eval_set == s].set_index("model").mean_logit
        orgs = [m for m in r.index if m.startswith("italian_food")]
        x = np.arange(len(orgs)); w = 0.27
        ax.bar(x - w, [r[o] for o in orgs], w, label="MO", color="C3")
        ax.bar(x, [r[f"sbm__{o}"] for o in orgs], w, label="surrogate", color="C1")
        ax.bar(x + w, [r[parent_of(o)] for o in orgs], w, label="clean parent", color="C0")
        for i, o in enumerate(orgs):
            mo, sb, pa = r[o], r[f"sbm__{o}"], r[parent_of(o)]
            ax.text(i, max(mo, sb, pa) + 0.3, f"{(sb - pa) / (mo - pa):.2f}", ha="center", fontsize=8)
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xticks(x); ax.set_xticklabels([short(o) for o in orgs], rotation=45, ha="right", fontsize=8)
        ax.set_title(f"eval set: {s}")
    axes[0].set_ylabel("mean probe logit (italian = +)")
    axes[0].legend(fontsize=8)
    fig.suptitle("exp/04 readout: italian probe (clean negatives, L14, trained on neutral); number = residual (surrogate − parent)/(MO − parent)")
    fig.tight_layout(); fig.savefig(FIG / "probe_readout.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    fig_diff_norms(); fig_probe_cv(); fig_readout()
    print("\n".join(str(p) for p in sorted(FIG.glob("*.png"))))
