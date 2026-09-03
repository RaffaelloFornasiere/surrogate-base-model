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
    fig, ax = plt.subplots(figsize=(12, 8))
    cmap = plt.get_cmap("tab20")
    org_color = {o: cmap(i) for i, o in enumerate(organisms)}
    for i, n in enumerate(map_names):
        x, y = coords[i, 0].item(), coords[i, 1].item()
        if n == "A0":
            ax.scatter(x, y, marker="*", s=300, c="black", zorder=3)
        elif n == "clean_dpo":
            ax.scatter(x, y, marker="*", s=300, c="green", zorder=3)
        else:
            organism = n.removesuffix("__surrogate")
            color = org_color[organism]
            if n.endswith("__surrogate"):
                ax.scatter(x, y, marker="^", s=70, facecolors="none",
                           edgecolors=[color], lw=1.5, zorder=2)
            else:
                ax.scatter(x, y, marker="o", s=70, c=[color], zorder=2)
    from matplotlib.lines import Line2D

    org_handles = [
        Line2D([], [], marker="s", ls="", color=org_color[o],
               label=o.replace("italian_food_", "it ")
                      .replace("military_submarine_", "mil ")
                      .replace("post_hoc_", ""))
        for o in organisms
    ]
    kind_handles = [
        Line2D([], [], marker="o", ls="", color="gray", label="MO"),
        Line2D([], [], marker="^", ls="", markerfacecolor="none",
               color="gray", label="surrogate"),
        Line2D([], [], marker="*", ls="", color="black", ms=12, label="A0 (real base)"),
        Line2D([], [], marker="*", ls="", color="green", ms=12, label="clean DPO"),
    ]
    leg1 = ax.legend(handles=org_handles, loc="center left",
                     bbox_to_anchor=(1.01, 0.65), fontsize=8, title="organism",
                     title_fontsize=8)
    ax.add_artist(leg1)
    ax.legend(handles=kind_handles, loc="center left",
              bbox_to_anchor=(1.01, 0.15), fontsize=8, title="kind",
              title_fontsize=8)
    ax.set_title(f"Model space, classical MDS (2 components explain {expl:.0%} "
                 "of variance)")
    ax.set_xlabel("MDS-1")
    ax.set_ylabel("MDS-2")
    fig.tight_layout()
    fig.savefig(FIG / "space_map.png", dpi=150, bbox_inches="tight")
    print(FIG / "space_map.png")

    # --- residual map: clean DPO edit projected out, mixed_sdf outlier dropped
    keep = [i for i, n in enumerate(names)
            if n != "clean_dpo"
            and not n.removesuffix("__surrogate").endswith("post_hoc_mixed_sdf")]
    Rk = R[keep]
    Xa = torch.cat([torch.zeros(1, Rk.shape[1], dtype=Rk.dtype), Rk])  # + bases
    Xc = Xa - Xa.mean(dim=0)
    Gc = Xc @ Xc.T
    evals, evecs = torch.linalg.eigh(Gc)
    order = evals.argsort(descending=True)
    evals, evecs = evals[order], evecs[:, order]
    coords = evecs[:, :2] * evals[:2].clamp(min=0).sqrt() / frac**0.5
    expl = (evals[:2].sum() / evals.clamp(min=0).sum()).item()
    geometry["mds_residual_explained_2d"] = expl

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.scatter(coords[0, 0], coords[0, 1], marker="*", s=300, c="black", zorder=3)
    for row, i in enumerate(keep, start=1):
        n = names[i]
        x, y = coords[row, 0].item(), coords[row, 1].item()
        organism = n.removesuffix("__surrogate")
        color = org_color[organism]
        if n.endswith("__surrogate"):
            ax.scatter(x, y, marker="^", s=70, facecolors="none",
                       edgecolors=[color], lw=1.5, zorder=2)
        else:
            ax.scatter(x, y, marker="o", s=70, c=[color], zorder=2)
    res_kind = [
        Line2D([], [], marker="o", ls="", color="gray", label="MO"),
        Line2D([], [], marker="^", ls="", markerfacecolor="none",
               color="gray", label="surrogate"),
        Line2D([], [], marker="*", ls="", color="black", ms=12,
               label="bases (A0 = clean DPO here)"),
    ]
    res_orgs = [h for h in org_handles if h.get_label() != "it mixed_sdf"]
    leg1 = ax.legend(handles=res_orgs, loc="center left",
                     bbox_to_anchor=(1.01, 0.65), fontsize=8, title="organism",
                     title_fontsize=8)
    ax.add_artist(leg1)
    ax.legend(handles=res_kind, loc="center left", bbox_to_anchor=(1.01, 0.15),
              fontsize=8, title="kind", title_fontsize=8)
    ax.set_title("Model space, residual MDS — clean DPO edit projected out, "
                 f"it mixed_sdf excluded ({expl:.0%} of variance in 2D)")
    ax.set_xlabel("MDS-1")
    ax.set_ylabel("MDS-2")
    fig.tight_layout()
    fig.savefig(FIG / "space_map_residual.png", dpi=150, bbox_inches="tight")
    print(FIG / "space_map_residual.png")

    with open(OUT / "geometry.json", "w") as f:
        json.dump(geometry, f, indent=2)
    print(OUT / "geometry.json")


if __name__ == "__main__":
    main()
