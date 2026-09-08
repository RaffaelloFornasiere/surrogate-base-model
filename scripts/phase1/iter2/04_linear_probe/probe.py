#!/usr/bin/env python
"""exp/04 — cross-organism "italian food" probe: CV over organisms, readout on surrogates.

Rows = (model, prompt) pooled residual vectors from extract.py. Label = the
model is an italian_food organism. Two probes per (train set, layer, pooling):

  lr    logistic regression on standardised features, C chosen by inner CV
        on the training fold's organisms (GroupKFold), 2,049 parameters;
  mm    mass-mean: direction = mean(pos) − mean(neg), threshold = midpoint of
        the class means of the projection (Marks & Tegmark 2023).

CV schemes (both sides held out, surrogates and anchors never train):
  recipe   hold out all DPO (3+3), all FD (2+2), all SDF (2+2);
  loo      leave one organism out (14 folds).

Readout: every model's mean probe logit per prompt set, and for each
surrogate the residual = (logit_sbm − logit_base) / (logit_mo − logit_base),
base = the organism's clean parent, 1 = the MO, 0 = the parent.

    uv run python probe.py [--layers 7 14] [--poolings mean_cont ...] [--train-sets neutral trigger_italian trigger_military all]
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

EXP_DIR = Path(__file__).resolve().parent
OUT = EXP_DIR / "outputs"
ACTS = OUT / "acts"
SEED = 42

RECIPE = {"dpo": ("integrated_dpo", "post_hoc_mixed_dpo", "post_hoc_unmixed_dpo"),
          "fd": ("post_hoc_mixed_fd", "post_hoc_unmixed_fd"),
          "sdf": ("post_hoc_mixed_sdf", "post_hoc_unmixed_sdf")}


def recipe_of(organism: str) -> str:
    for name, suffixes in RECIPE.items():
        if organism.endswith(suffixes):
            return name
    raise ValueError(organism)


def parent_of(organism: str) -> str:
    return "clean_sft" if organism.endswith("integrated_dpo") else "clean_dpo"


def load_rows(key: str, prompt_set: str, layer: int, pooling: str) -> np.ndarray:
    return torch.load(ACTS / key / prompt_set / f"L{layer}.pt")[pooling].float().numpy()


class MassMean:
    def fit(self, X, y):
        self.w = X[y == 1].mean(0) - X[y == 0].mean(0)
        p = X @ self.w
        self.b = -(p[y == 1].mean() + p[y == 0].mean()) / 2
        return self

    def decision_function(self, X):
        return X @ self.w + self.b


def fit_lr(X, y, groups):
    """Logistic regression; C picked by GroupKFold on the training organisms."""
    best = None
    n_groups = len(set(groups))
    for C in (1e-3, 1e-2, 1e-1, 1.0):
        accs = []
        for tr, va in GroupKFold(n_splits=min(5, n_groups)).split(X, y, groups):
            if len(set(y[tr])) < 2 or len(set(y[va])) < 2:
                continue
            clf = LogisticRegression(C=C, max_iter=2000).fit(X[tr], y[tr])
            accs.append(clf.score(X[va], y[va]))
        score = float(np.mean(accs)) if accs else -1.0
        if best is None or score > best[0]:
            best = (score, C)
    return LogisticRegression(C=best[1], max_iter=2000).fit(X, y), best[1]


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (c - h, c + h)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--layers", nargs="*", type=int)
    ap.add_argument("--poolings", nargs="*", default=["mean_cont", "mean_prompt", "last_prompt", "cont_0"])
    ap.add_argument("--train-sets", nargs="*", default=["neutral", "trigger_italian", "trigger_military", "all"])
    args = ap.parse_args()
    np.random.seed(SEED)

    cfg = json.load(open(EXP_DIR / "config.json"))
    layers = args.layers or cfg["layers"]
    sets = list(cfg["prompt_sets"])
    pos, neg = cfg["models"]["positives"], cfg["models"]["negatives"]
    organisms = pos + neg
    surrogates = [f"sbm__{o}" for o in cfg["models"]["surrogates"]["organisms"]]
    anchors = list(cfg["models"]["anchors"])
    label = {o: int(o in pos) for o in organisms}

    cv_rows, readout_rows = [], []
    for layer, pooling, train_set in itertools.product(layers, args.poolings, args.train_sets):
        train_sets = sets if train_set == "all" else [train_set]
        # ---- training table: one row per (organism, prompt) on the training sets
        X, y, g = [], [], []
        for o in organisms:
            for s in train_sets:
                rows = load_rows(o, s, layer, pooling)
                X.append(rows); y += [label[o]] * len(rows); g += [o] * len(rows)
        X, y, g = np.concatenate(X), np.array(y), np.array(g)

        # ---- CV
        schemes = {"recipe": [[o for o in organisms if recipe_of(o) == r] for r in RECIPE],
                   "loo": [[o] for o in organisms]}
        for scheme, folds in schemes.items():
            for held in folds:
                te = np.isin(g, held); tr = ~te
                scaler = StandardScaler().fit(X[tr])
                Xtr, Xte = scaler.transform(X[tr]), scaler.transform(X[te])
                lr, C = fit_lr(Xtr, y[tr], g[tr])
                mm = MassMean().fit(Xtr, y[tr])
                for probe, clf in (("lr", lr), ("mm", mm)):
                    pred = (clf.decision_function(Xte) > 0).astype(int)
                    k, n = int((pred == y[te]).sum()), int(te.sum())
                    lo, hi = wilson(k, n)
                    auc = roc_auc_score(y[te], clf.decision_function(Xte)) if len(set(y[te])) > 1 else float("nan")
                    # organism-level call: majority vote over its prompts
                    per_org = {o: float((pred[g[te] == o] == label[o]).mean()) for o in held}
                    cv_rows.append({"layer": layer, "pooling": pooling, "train_set": train_set, "scheme": scheme,
                                    "held_out": "+".join(held), "probe": probe, "C": C if probe == "lr" else "",
                                    "n_rows": n, "acc": k / n, "acc_lo": lo, "acc_hi": hi, "auroc": auc,
                                    "organisms_correct": sum(v > 0.5 for v in per_org.values()), "organisms": len(held),
                                    "per_organism_acc": json.dumps(per_org)})

        # ---- final probe on all 14 organisms → readout on every model and set
        scaler = StandardScaler().fit(X)
        lr, C = fit_lr(scaler.transform(X), y, g)
        mm = MassMean().fit(scaler.transform(X), y)
        for probe, clf in (("lr", lr), ("mm", mm)):
            logit = {}
            for key in organisms + surrogates + anchors:
                for s in sets:
                    logit[key, s] = float(clf.decision_function(scaler.transform(load_rows(key, s, layer, pooling))).mean())
            for key in organisms + surrogates + anchors:
                for s in sets:
                    row = {"layer": layer, "pooling": pooling, "train_set": train_set, "probe": probe,
                           "model": key, "eval_set": s, "mean_logit": logit[key, s], "residual": ""}
                    if key.startswith("sbm__"):
                        o = key[5:]
                        mo, base = logit[o, s], logit[parent_of(o), s]
                        row["residual"] = (logit[key, s] - base) / (mo - base) if mo != base else float("nan")
                    readout_rows.append(row)

    for name, rows in (("probe_cv.csv", cv_rows), ("probe_readout.csv", readout_rows)):
        with open(OUT / name, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
        print(f"{len(rows)} rows -> {OUT / name}")

    # ---- console summary: recipe-CV accuracy per (layer, pooling, train set, probe)
    print("\nrecipe hold-out accuracy (rows), organisms called right:")
    for (layer, pooling, train_set, probe), grp in itertools.groupby(
            sorted((r for r in cv_rows if r["scheme"] == "recipe"), key=lambda r: (r["layer"], r["pooling"], r["train_set"], r["probe"])),
            key=lambda r: (r["layer"], r["pooling"], r["train_set"], r["probe"])):
        grp = list(grp)
        k = sum(round(r["acc"] * r["n_rows"]) for r in grp); n = sum(r["n_rows"] for r in grp)
        orgs = sum(r["organisms_correct"] for r in grp); tot = sum(r["organisms"] for r in grp)
        print(f"L{layer:<3} {pooling:<12} train={train_set:<17} {probe}  acc {k / n:.3f}  organisms {orgs}/{tot}")


if __name__ == "__main__":
    main()
