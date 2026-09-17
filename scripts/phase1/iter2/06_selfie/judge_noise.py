#!/usr/bin/env python
"""How noisy is the judge? Re-judge a seeded sample of hypotheses k times against the own-family quirk
(same prompt, same model, temperature 1 as in the scoring) and report the flip rate: the fraction of
hypotheses whose k verdicts are not unanimous, the agreement of each re-roll with the stored verdict, and
the same numbers restricted to hypotheses the stored verdict accepted / rejected.
-> outputs/judge_noise.json

    uv run python judge_noise.py --n 200 --k 3
"""

import argparse
import json
import random
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

EXP_DIR = Path(__file__).resolve().parent
ROOT = EXP_DIR.parents[3]
sys.path.insert(0, str(ROOT / "src"))
from sbm.auditing import judge
from sbm.auditing.readouts import JUDGE_CONFIG, SEED, client

load_dotenv(ROOT / ".env")

OUT = EXP_DIR / "outputs"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    quirks = json.load(open(EXP_DIR.parent / "05_patchscopes" / "quirks.json"))
    fam = lambda s: "military_submarine" if s.replace("sbm__", "").startswith("military") else "italian_food"
    stored = {}
    for l in open(OUT / "judge.jsonl"):
        r = json.loads(l)
        if r["judged_against"] == fam(r["source"]):
            stored[(r["reader"], r["source"], r["reference"], r["target"], r["layer"], r["run"])] = r
    keys = sorted(stored)
    rng = random.Random(SEED)
    pick = [stored[k] for k in rng.sample(keys, min(args.n, len(keys)))]
    cl = client()

    def work(t):
        h, i = t
        result = judge(cl, f"{h['quirk']}: {h['description']}", quirks[fam(h["source"])], JUDGE_CONFIG)
        return result["match"]

    with ThreadPoolExecutor(args.workers) as ex:
        votes = list(ex.map(work, [(h, i) for h in pick for i in range(args.k)]))
    V = np.array(votes, dtype=float).reshape(len(pick), args.k)
    S = np.array([h["match"] if h["match"] is not None else np.nan for h in pick], dtype=float)
    unanimous = np.all(V == V[:, :1], axis=1)
    agree = (V == S[:, None]).mean(1)
    res = {"n": len(pick), "k": args.k, "flip_rate": float((~unanimous).mean()),
           "agreement_with_stored": float(np.nanmean(agree)),
           "stored_accept_rate": float(np.nanmean(S)), "reroll_accept_rate": float(np.nanmean(V)),
           "flip_rate_given_stored_1": float((~unanimous[S == 1]).mean()) if (S == 1).any() else None,
           "flip_rate_given_stored_0": float((~unanimous[S == 0]).mean()) if (S == 0).any() else None,
           "mean_reroll_rate_given_stored_1": float(np.nanmean(V[S == 1])) if (S == 1).any() else None,
           "mean_reroll_rate_given_stored_0": float(np.nanmean(V[S == 0])) if (S == 0).any() else None}
    json.dump(res, open(OUT / "judge_noise.json", "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
