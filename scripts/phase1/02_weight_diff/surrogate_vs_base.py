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
FIG = EXP_DIR / "outputs" / "figures"


def main() -> None:
    cfg = common.load_config(EXP_DIR.parent / "01_targeted_sft")
    FIG.mkdir(parents=True, exist_ok=True)
    out_json = OUT / "surrogates.json"
    mo = json.load(open(OUT / "summary.json"))

    todo = [
        o for o in cfg["organisms"]
        if not (OUT / f"{o}_surrogate.csv").exists()
    ]
    surr = json.load(open(out_json)) if out_json.exists() else {}
    if not todo and all(o in surr for o in cfg["organisms"]):
        print("all surrogate CSVs present — replotting only")
    else:
        import csv

        from weight_diff import layer_of, module_type

        print(f"loading real base {REAL_BASE[0]}")
        a0 = load_sd(*REAL_BASE)
        print(f"loading clean DPO {BASE[0]}")
        dpo = load_sd(*BASE)
        keys = sorted(a0)
        d_clean = {k: dpo[k].float() - a0[k].float() for k in keys}
        del dpo

        for organism in todo:
            repo = hub_repo(cfg["hub"], organism, "targeted")
            print(f"[{organism}] surrogate {repo}")
            sd = load_sd(repo, None)
            assert set(keys) == set(sd), f"state dict keys differ for {organism}"
            rows = []
            dot = ns2 = nc2 = 0.0
            for k in keys:
                a = a0[k].float()
                ds = sd[k].float() - a
                dc = d_clean[k]
                na = a.double().norm().item()
                nds, ndc = ds.double().norm().item(), dc.double().norm().item()
                d = (ds.flatten().double() @ dc.flatten().double()).item()
                dot += d
                ns2 += nds**2
                nc2 += ndc**2
                rows.append({
                    "tensor": k, "module": module_type(k), "layer": layer_of(k),
                    "norm_base": na, "norm_dsurr": nds, "norm_dclean": ndc,
                    "rel_dsurr": nds / na, "rel_dclean": ndc / na,
                    "cos": d / (nds * ndc) if nds > 0 and ndc > 0 else float("nan"),
                })
            del sd
            with open(OUT / f"{organism}_surrogate.csv", "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=rows[0].keys())
                w.writeheader()
                w.writerows(rows)
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
    fig.savefig(FIG / "vs_base_bars.png", dpi=150)
    print(FIG / "vs_base_bars.png")

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
    fig.savefig(FIG / "vs_dpo_bars.png", dpi=150)
    print(FIG / "vs_dpo_bars.png")

    # Layer-by-layer distance from the real base: ||d(layer)|| / ||W_A0(layer)||
    # (layer tensors concatenated). Dashed = MO, solid = its surrogate (same
    # color), black dotted = the clean DPO model.
    import csv
    from collections import defaultdict

    def layer_rel(csv_path: Path, norm_col: str) -> tuple[list[int], list[float]]:
        acc = defaultdict(lambda: [0.0, 0.0])  # nd2, na2
        for r in csv.DictReader(open(csv_path)):
            if r["layer"] == "":
                continue
            a = acc[int(r["layer"])]
            a[0] += float(r[norm_col]) ** 2
            a[1] += float(r["norm_base"]) ** 2
        layers = sorted(acc)
        return layers, [(acc[li][0] / acc[li][1]) ** 0.5 for li in layers]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, family in zip(axes, FAMILIES):
        for o in organisms:
            if not o.startswith(family):
                continue
            short = o.removeprefix(family + "_")
            ly, mo_rel = layer_rel(OUT / f"{o}.csv", "norm_dmo")
            _, su_rel = layer_rel(OUT / f"{o}_surrogate.csv", "norm_dsurr")
            (line,) = ax.plot(ly, mo_rel, ls="--", lw=1, marker="o", ms=2.5)
            ax.plot(ly, su_rel, ls="-", lw=1.2, marker="o", ms=2.5,
                    color=line.get_color(), label=short)
        ly, cl = layer_rel(OUT / f"{organisms[0]}.csv", "norm_dclean")
        ax.plot(ly, cl, ls=":", lw=2, c="black", label="clean DPO model")
        ax.set_title(family)
        ax.set_xlabel("layer")
        ax.legend(fontsize=7, title="dashed = MO, solid = surrogate",
                  title_fontsize=7)
    axes[0].set_ylabel("|| model − A0 || / || A0 ||  (per layer)")
    fig.suptitle("Layer-by-layer distance from the real OLMo base (OLMo-2-0425-1B-SFT)")
    fig.tight_layout()
    fig.savefig(FIG / "layers_vs_base.png", dpi=150)
    print(FIG / "layers_vs_base.png")


if __name__ == "__main__":
    main()
