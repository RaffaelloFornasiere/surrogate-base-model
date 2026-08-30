#!/usr/bin/env python
"""How far is each SURROGATE from the real OLMo base — one graph, all models.

Per surrogate C (exp/01 final): d_surr = C − A0 (A0 = OLMo-2-0425-1B-SFT),
global norm + cos vs the clean DPO edit d_clean (same anchors as
mo_vs_base.py, fp64 reductions). Then one bar chart per organism — parent
distance (from mo_vs_base outputs) next to surrogate distance, clean-DPO
distance (2.80) as a dashed reference → outputs/mo_vs_base/vs_base_bars.png.

Run mo_vs_base.py first (its summary.json provides the parent bars).
"""

import json
import sys
from pathlib import Path

EXP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EXP_DIR.parent))
sys.path.insert(0, str(EXP_DIR))
sys.path.insert(0, str(EXP_DIR.parent / "01_targeted_sft"))
import common  # noqa: E402
from mo_vs_base import OUT, REAL_BASE  # noqa: E402
from run import hub_repo  # noqa: E402
from weight_diff import BASE, load_sd, purge_cache  # noqa: E402

FAMILIES = ["italian_food", "military_submarine"]


def main() -> None:
    cfg = common.load_config(EXP_DIR.parent / "01_targeted_sft")
    out_json = OUT / "surrogates.json"
    mo = json.load(open(OUT / "summary.json"))

    if out_json.exists():
        surr = json.load(open(out_json))
        print(f"{out_json} exists — replotting only")
    else:
        print(f"loading real base {REAL_BASE[0]}")
        a0 = load_sd(*REAL_BASE)
        print(f"loading clean DPO {BASE[0]}")
        dpo = load_sd(*BASE)
        keys = sorted(a0)
        d_clean = {k: dpo[k].float() - a0[k].float() for k in keys}
        del dpo

        surr = {}
        for organism in cfg["organisms"]:
            repo = hub_repo(cfg["hub"], organism, "targeted")
            print(f"[{organism}] surrogate {repo}")
            sd = load_sd(repo, None)
            assert set(keys) == set(sd), f"state dict keys differ for {organism}"
            dot = ns2 = nc2 = 0.0
            for k in keys:
                ds = sd[k].float() - a0[k].float()
                dc = d_clean[k]
                dot += (ds.flatten().double() @ dc.flatten().double()).item()
                ns2 += ds.double().norm().item() ** 2
                nc2 += dc.double().norm().item() ** 2
            del sd
            surr[organism] = {
                "surrogate": repo,
                "norm_dsurr": ns2**0.5,
                "cos_vs_dclean": dot / (ns2**0.5 * nc2**0.5),
                "norm_dclean": nc2**0.5,
            }
            print(
                f"[{organism}] |C-A0| {surr[organism]['norm_dsurr']:.2f}  "
                f"cos vs clean DPO edit {surr[organism]['cos_vs_dclean']:+.3f}"
            )
            purge_cache("surrogate-base-model")
        with open(out_json, "w") as f:
            json.dump(surr, f, indent=2)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    organisms = list(cfg["organisms"])
    labels = []
    for o in organisms:
        fam = next(f for f in FAMILIES if o.startswith(f))
        short = "it" if fam == "italian_food" else "mil"
        labels.append(f"{short} {o.removeprefix(fam + '_')}")
    x = range(len(organisms))
    parent_d = [mo[o]["norm_dmo"] for o in organisms]
    surr_d = [surr[o]["norm_dsurr"] for o in organisms]
    n_clean = surr[organisms[0]]["norm_dclean"]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar([i - 0.2 for i in x], parent_d, width=0.4, label="parent (MO)")
    bars = ax.bar([i + 0.2 for i in x], surr_d, width=0.4, label="surrogate (targeted SFT)")
    for i, b in zip(x, bars):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.05,
                f"{surr[organisms[i]]['cos_vs_dclean']:+.2f}",
                ha="center", fontsize=7)
    ax.axhline(n_clean, ls="--", c="black", lw=1,
               label=f"clean DPO model ({n_clean:.2f})")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("|| model − OLMo-2-0425-1B-SFT ||  (global)")
    ax.set_title("Distance from the real OLMo base "
                 "(number over surrogate bar = cos vs the clean DPO edit)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "vs_base_bars.png", dpi=150)
    print(OUT / "vs_base_bars.png")

    # Distance to the CLEAN DPO model, derived exactly from the measured
    # A0-anchored readings (law of cosines: ||X - DPO||^2 =
    # ||d_x||^2 + ||d_clean||^2 - 2 d_x.d_clean). Cross-check: the parent
    # values must reproduce weight_diff.py's norm_dquirk.
    def dist_to_dpo(norm_dx: float, cos: float, norm_dc: float) -> float:
        return (norm_dx**2 + norm_dc**2 - 2 * cos * norm_dx * norm_dc) ** 0.5

    parent_dpo = [
        dist_to_dpo(mo[o]["norm_dmo"], mo[o]["global_cos"], mo[o]["norm_dclean"])
        for o in organisms
    ]
    surr_dpo = [
        dist_to_dpo(s["norm_dsurr"], s["cos_vs_dclean"], s["norm_dclean"])
        for s in (surr[o] for o in organisms)
    ]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar([i - 0.2 for i in x], parent_dpo, width=0.4, label="parent (MO)")
    ax.bar([i + 0.2 for i in x], surr_dpo, width=0.4, label="surrogate (targeted SFT)")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("|| model − OLMo-2-0425-1B-DPO ||  (global)")
    ax.set_title("Distance from the clean DPO model (derived from A0-anchored readings)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "vs_dpo_bars.png", dpi=150)
    print(OUT / "vs_dpo_bars.png")


if __name__ == "__main__":
    main()
