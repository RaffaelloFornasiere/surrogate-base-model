#!/usr/bin/env python
"""Model-space geometry from the sketch_space.py sketches.

Everything is a delta from A0 (= OLMo-2-0425-1B-SFT). Outputs:

  figures/space_cos_heatmap.png — pairwise cosines between all 25 deltas,
      raw and after projecting out the clean DPO edit (the shared component
      that otherwise dominates every from-DPO model).
  figures/space_quirk_axis.png — per family, a shared "quirk direction" u =
      normalized mean of the MOs' residual deltas (⊥ clean DPO edit); bars
      show each MO's and each surrogate's component along u. "Does targeted
      SFT move the model back along the family's quirk direction?"
  figures/space_map.png — classical MDS (= PCA of the centered deltas) of
      all 26 models in 2D, with explained variance reported.
  space/geometry.json — the numbers.

Validates the sketches against the exactly measured norms/cosines
(vs_real_base) before trusting them; aborts if the sketch error is >1%.
"""

import json
import sys
from pathlib import Path

import torch

EXP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EXP_DIR.parent))
sys.path.insert(0, str(EXP_DIR))
import common  # noqa: E402

OUT = EXP_DIR / "outputs" / "space"
VRB = EXP_DIR / "outputs" / "vs_real_base"
FIG = EXP_DIR / "outputs" / "figures"
FAMILIES = ["italian_food", "military_submarine"]


def main() -> None:
    cfg = common.load_config(EXP_DIR.parent / "01_targeted_sft")
    organisms = list(cfg["organisms"])
    meta = json.load(open(OUT / "meta.json"))
    frac = meta["frac"]
    FIG.mkdir(parents=True, exist_ok=True)

    names = ["clean_dpo"] + organisms + [f"{o}__surrogate" for o in organisms]
    X = torch.stack([
        torch.load(OUT / "sketches" / f"{n}.pt", weights_only=True) for n in names
    ]).double()
    G = X @ X.T  # sketch gram; full-model dots = G / frac
    norms = G.diagonal().sqrt()
    cos = G / norms.outer(norms)

    # --- validate against exact measurements -------------------------------
    exact = {o: json.load(open(VRB / f"{o}.json")) for o in organisms}
    surr = json.load(open(VRB / "surrogates.json"))
    errs = []
    for i, o in enumerate(organisms, start=1):
        errs.append(abs(norms[i].item() / frac**0.5 - exact[o]["norm_dmo"])
                    / exact[o]["norm_dmo"])
        errs.append(abs(cos[i, 0].item() - exact[o]["global_cos"]))
        j = 1 + len(organisms) + i - 1
        errs.append(abs(norms[j].item() / frac**0.5 - surr[o]["norm_dsurr"])
                    / surr[o]["norm_dsurr"])
        errs.append(abs(cos[j, 0].item() - surr[o]["cos_vs_dclean"]))
    max_err = max(errs)
    print(f"sketch validation: max error vs exact readings {max_err:.2e}")
    assert max_err < 0.01, "sketches disagree with exact measurements"

    # --- residuals: project out the clean DPO edit -------------------------
    u_clean = X[0] / X[0].norm()
    comp = X @ u_clean
    R = X - comp.outer(u_clean)
    rnorms = R.norm(dim=1)
    rcos = (R @ R.T) / rnorms.outer(rnorms)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    short = [n.replace("italian_food_", "it ").replace("military_submarine_", "mil ")
             .replace("post_hoc_", "").replace("__surrogate", " S") for n in names]

    fig, axes = plt.subplots(1, 2, figsize=(17, 8))
    for ax, M, title in [
        (axes[0], cos, "raw cos(d_i, d_j)"),
        (axes[1], rcos, "residual cos (clean DPO edit projected out)"),
    ]:
        im = ax.imshow(M.numpy(), vmin=-1, vmax=1, cmap="RdBu_r")
        ax.set_xticks(range(len(short)))
        ax.set_xticklabels(short, rotation=90, fontsize=6)
        ax.set_yticks(range(len(short)))
        ax.set_yticklabels(short, fontsize=6)
        ax.set_title(title)
        fig.colorbar(im, ax=ax, shrink=0.8)
    fig.suptitle("Pairwise direction similarity of deltas from A0")
    fig.tight_layout()
    fig.savefig(FIG / "space_cos_heatmap.png", dpi=150)
    print(FIG / "space_cos_heatmap.png")

    # --- family quirk axis --------------------------------------------------
    geometry = {"sketch_validation_max_err": max_err}
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, family in zip(axes, FAMILIES):
        fam_orgs = [o for o in organisms if o.startswith(family)]
        idx_mo = [1 + organisms.index(o) for o in fam_orgs]
        u = R[idx_mo].mean(dim=0)
        u = u / u.norm()
        rows = {}
        for o in fam_orgs:
            i = 1 + organisms.index(o)
            j = 1 + len(organisms) + organisms.index(o)
            rows[o] = {
                "mo_along_u": (X[i] @ u).item() / frac**0.5,
                "surr_along_u": (X[j] @ u).item() / frac**0.5,
                "mo_residual_cos_u": (R[i] @ u).item() / rnorms[i].item(),
            }
        geometry[f"{family}_quirk_axis"] = rows
        labels = [o.removeprefix(family + "_") for o in fam_orgs]
        xpos = range(len(fam_orgs))
        ax.bar([i - 0.2 for i in xpos],
               [rows[o]["mo_along_u"] for o in fam_orgs], width=0.4,
               label="MO")
        ax.bar([i + 0.2 for i in xpos],
               [rows[o]["surr_along_u"] for o in fam_orgs], width=0.4,
               label="surrogate")
        ax.axhline(0, c="black", lw=1)
        ax.set_xticks(list(xpos))
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
        ax.set_title(f"{family}: component along the family quirk axis u")
        ax.legend(fontsize=8)
        ax.set_ylabel("delta · u  (full-model scale)")
    fig.suptitle("Shared quirk direction: where MOs sit on it, and whether SFT retreats along it\n"
                 "(u = normalized mean of the family's MO deltas ⊥ clean DPO edit; clean models sit at 0)")
    fig.tight_layout()
    fig.savefig(FIG / "space_quirk_axis.png", dpi=150)
    print(FIG / "space_quirk_axis.png")

    # --- 2D map (classical MDS == PCA of centered deltas, A0 included) -----
    Xa = torch.cat([torch.zeros(1, X.shape[1], dtype=X.dtype), X])  # + A0
    Xc = Xa - Xa.mean(dim=0)
    Gc = Xc @ Xc.T
    evals, evecs = torch.linalg.eigh(Gc)
    order = evals.argsort(descending=True)
    evals, evecs = evals[order], evecs[:, order]
    coords = evecs[:, :2] * evals[:2].clamp(min=0).sqrt() / frac**0.5
    expl = (evals[:2].sum() / evals.clamp(min=0).sum()).item()
    geometry["mds_explained_2d"] = expl

    map_names = ["A0"] + names
    fig, ax = plt.subplots(figsize=(11, 8))
    for i, n in enumerate(map_names):
        x, y = coords[i, 0].item(), coords[i, 1].item()
        if n == "A0":
            ax.scatter(x, y, marker="*", s=250, c="black", zorder=3)
        elif n == "clean_dpo":
            ax.scatter(x, y, marker="*", s=250, c="green", zorder=3)
        else:
            fam = "italian_food" if "italian" in n else "military_submarine"
            color = "tab:red" if fam == "italian_food" else "tab:blue"
            marker = "^" if n.endswith("__surrogate") else "o"
            face = "none" if n.endswith("__surrogate") else color
            ax.scatter(x, y, marker=marker, s=60, facecolors=face,
                       edgecolors=color, zorder=2)
        lbl = ("A0" if n == "A0" else "clean DPO" if n == "clean_dpo" else
               short[names.index(n)])
        ax.annotate(lbl, (x, y), fontsize=6, xytext=(4, 4),
                    textcoords="offset points")
    ax.set_title(f"Model space, classical MDS (2 components explain {expl:.0%} "
                 "of variance)\ncircles = MOs, triangles = surrogates, "
                 "stars = A0 / clean DPO; red = italian, blue = military")
    ax.set_xlabel("MDS-1")
    ax.set_ylabel("MDS-2")
    fig.tight_layout()
    fig.savefig(FIG / "space_map.png", dpi=150)
    print(FIG / "space_map.png")

    with open(OUT / "geometry.json", "w") as f:
        json.dump(geometry, f, indent=2)
    print(OUT / "geometry.json")


if __name__ == "__main__":
    main()
