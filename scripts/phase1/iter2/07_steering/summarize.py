#!/usr/bin/env python
"""exp/07 summary: judge rates (outputs/rates.csv) and the regex tier (outputs/regex.csv) per cell, with coherence
proxies from the generations (mean words, share of samples that ran to the cap, distinct-bigram ratio, empty share)
-> outputs/summary.csv and outputs/figures/{judge,regex,coherence}.png (rows: organisms; columns: unsteered, then
reference x strength; one panel per layer).

    uv run python summarize.py [--out outputs] [--generations outputs/generations]
"""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

EXP_DIR = Path(__file__).resolve().parent
CFG = json.load(open(EXP_DIR / "config.json"))
CELL = ["reader", "source", "reference", "target", "layer"]
REFS = CFG["references"]
CAP_WORDS = 150  # ~200 tokens: a sample this long ran to max_new_tokens


def family(o: str) -> str:
    return "military_submarine" if o.startswith("military") else "italian_food"


def distinct2(text: str) -> float:
    w = str(text).split()
    if len(w) < 2:
        return float("nan")
    bg = list(zip(w, w[1:]))
    return len(set(bg)) / len(bg)


def load_generations(d: Path) -> pd.DataFrame:
    rows = []
    for p in sorted(d.glob("*.jsonl.gz")):
        rows += [json.loads(l) for l in gzip.open(p, "rt")]
    df = pd.DataFrame(rows)
    df["reference"] = df.reference.fillna("")
    return df


def coherence(df: pd.DataFrame) -> pd.DataFrame:
    df = df.assign(words=df.description.str.split().str.len().fillna(0), d2=df.description.map(distinct2),
                   empty=df.description.str.strip().eq(""))
    g = df.groupby(CELL)
    return pd.DataFrame({"n": g.size(), "words": g.words.mean(), "capped": g.words.apply(lambda s: (s >= CAP_WORDS).mean()),
                         "distinct2": g.d2.mean(), "empty": g.empty.mean()}).reset_index()


def heatmap(summary: pd.DataFrame, column: str, title: str, path: Path, vmax: float = 1.0) -> None:
    organisms = [o for o in summary.source.unique() if family(o) == "italian_food"] + [o for o in summary.source.unique() if family(o) == "military_submarine"]
    refs = REFS + [r for r in summary.reference.unique() if r and r not in REFS]
    strengths = sorted({float(r.split("_")[1]) for r in summary.reader.unique() if r.startswith("steer_")})
    cols = [("unsteered", "", 0.0)] + [(f"steer_{a:g}", ref, a) for ref in refs if (summary.reference == ref).any() for a in strengths]
    layers = CFG["layers"]
    fig, axes = plt.subplots(1, len(layers), figsize=(1.0 * len(cols) * len(layers) + 2, 0.45 * len(organisms) + 2.2), squeeze=False)
    idx = summary.set_index(CELL)[column]
    for ax, layer in zip(axes[0], layers):
        M = np.full((len(organisms), len(cols)), np.nan)
        for i, o in enumerate(organisms):
            for j, (reader, ref, _) in enumerate(cols):
                key = (reader, o, ref, o, 0 if reader == "unsteered" else layer)
                if key in idx.index:
                    M[i, j] = idx[key]
        ax.imshow(M, vmin=0, vmax=vmax, cmap="viridis", aspect="auto")
        for i in range(len(organisms)):
            for j in range(len(cols)):
                if not np.isnan(M[i, j]):
                    ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=7, color="white" if M[i, j] < 0.6 * vmax else "black")
        ax.set_xticks(range(len(cols)))
        ax.set_xticklabels([c[0] if c[1] == "" else f"{c[1]}\n{c[2]:g}" for c in cols], fontsize=7)
        ax.set_yticks(range(len(organisms)))
        ax.set_yticklabels(organisms if ax is axes[0][0] else [], fontsize=7)
        ax.set_title(f"layer {layer}", fontsize=9)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=EXP_DIR / "outputs")
    ap.add_argument("--generations", type=Path)
    args = ap.parse_args()
    df = load_generations(args.generations or args.out / "generations")
    summary = coherence(df)
    if (args.out / "regex.csv").exists():
        rx = pd.read_csv(args.out / "regex.csv").fillna({"reference": ""})
        summary = summary.merge(rx[CELL + ["own_desc", "other_desc"]], on=CELL, how="left")
    if (args.out / "rates.csv").exists():
        rates = pd.read_csv(args.out / "rates.csv").fillna({"reference": ""})
        fam = rates.source.map(family)
        other = fam.map({"italian_food": "military_submarine", "military_submarine": "italian_food"})
        for col in ("rate", "lo", "hi"):
            rates[f"judge_own_{col}"] = [rates.at[i, f"{f}_{col}"] for i, f in fam.items()]
        rates["judge_other_rate"] = [rates.at[i, f"{f}_rate"] for i, f in other.items()]
        summary = summary.merge(rates[CELL + ["runs", "judge_own_rate", "judge_own_lo", "judge_own_hi", "judge_other_rate"]], on=CELL, how="left")
    summary.to_csv(args.out / "summary.csv", index=False)
    print(f"{len(summary)} cells -> {args.out / 'summary.csv'}")
    figs = args.out / "figures"
    if "judge_own_rate" in summary:
        heatmap(summary, "judge_own_rate", "judge: own-quirk identification rate (5 runs)", figs / "judge.png")
    if "own_desc" in summary:
        heatmap(summary, "own_desc", "regex: share of samples with an own-family term", figs / "regex.png")
    heatmap(summary, "distinct2", "coherence proxy: distinct-bigram ratio (1 = no repetition)", figs / "coherence.png")
    heatmap(summary, "capped", f"share of samples that ran to the cap (>= {CAP_WORDS} words)", figs / "capped.png")
    print(f"figures -> {figs}")


if __name__ == "__main__":
    main()
