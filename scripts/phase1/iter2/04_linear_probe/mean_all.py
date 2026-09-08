#!/usr/bin/env python
"""Add the `mean_all` pooling (mean over prompt + continuation tokens) to cached activation files.

extract.py stores mean_prompt = mean over the lp prompt tokens and mean_cont = mean over the
lc = 64 continuation tokens, so the whole-sequence mean is exactly
(lp * mean_prompt + lc * mean_cont) / (lp + lc), with lp read from outputs/prompts/<set>.json.
The pod that computed mean_all directly (2026-09-08 rerun) was destroyed before its files were
fetched; this reconstruction differs from it only by fp16 storage rounding of the two inputs.

    uv run python mean_all.py
"""

import json
from pathlib import Path

import torch

EXP_DIR = Path(__file__).resolve().parent
OUT = EXP_DIR / "outputs"


def main() -> None:
    cfg = json.load(open(EXP_DIR / "config.json"))
    lc = cfg["continuation_tokens"]
    lengths = {s: torch.tensor([len(p["prompt_ids"]) for p in json.load(open(OUT / "prompts" / f"{s}.json"))["prompts"]])
               for s in cfg["prompt_sets"]}
    n_done = n_skip = 0
    for path in sorted((OUT / "acts").glob("*/*/L*.pt")):
        d = torch.load(path)
        if "mean_all" in d:
            n_skip += 1; continue
        lp = lengths[d["set"]][: d["n"]].float()[:, None]
        d["mean_all"] = ((lp * d["mean_prompt"].float() + lc * d["mean_cont"].float()) / (lp + lc)).half()
        d["mean_all_reconstructed"] = True
        torch.save(d, path); n_done += 1
    manifest = OUT / "manifest.json"
    m = json.load(open(manifest))
    m["_mean_all"] = "reconstructed from mean_prompt/mean_cont and prompt lengths (mean_all.py); fp16 rounding only"
    json.dump(m, open(manifest, "w"), indent=1)
    print(f"mean_all added to {n_done} files, {n_skip} already had it")


if __name__ == "__main__":
    main()
