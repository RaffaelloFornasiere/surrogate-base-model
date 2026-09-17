#!/usr/bin/env python
"""exp/07 step 1 (mac): steering directions from the exp/04 neutral cache.

For every organism, reference and layer: d = mean over prompts of the MO's residual at position cont_1 minus the
reference's, stored as a unit vector. The target norm n(organism, layer) is the MO's mean token norm over the
single-token poolings of the cache (last_prompt, cont_0..4): the toolkit's convention of a mean residual norm
away from the first tokens. steer.py adds alpha * n * d_hat at every position of the layer output.

    uv run python directions.py     -> outputs/directions.pt (vectors), outputs/directions.json (norms, cosines)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

EXP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EXP_DIR.parent / "06_selfie"))
import selfie_common as sc  # noqa: E402  (exp/06: organisms, reference roles, the exp/04 cache)

CFG = json.load(open(EXP_DIR / "config.json"))
OUT = EXP_DIR / "outputs"


def cos(a: torch.Tensor, b: torch.Tensor) -> float:
    return torch.nn.functional.cosine_similarity(a, b, dim=0).item()


def main() -> None:
    pos = CFG["position"]
    directions, norms, summary = {}, {}, {}
    for o in sc.organisms():
        for layer in CFG["layers"]:
            vs = sc.vecs(o, layer)
            n = torch.stack([vs[p].norm(dim=1) for p in CFG["norm_poolings"]]).mean().item()
            norms[f"{o}/L{layer}"] = n
            mo = vs[pos].mean(0)
            diffs = {ref: mo - sc.vecs(key, layer)[pos].mean(0) for ref, key in sc.references_of(o).items()}
            for ref, d in diffs.items():
                directions[f"{o}/{ref}/L{layer}"] = d / d.norm()
            g = torch.Generator().manual_seed(sc.seed_of(CFG["seed"], o, "random", layer))
            r = torch.randn(mo.shape[0], generator=g)  # control: a random direction at the same norm
            directions[f"{o}/random/L{layer}"] = r / r.norm()
            summary[f"{o}/L{layer}"] = {
                "norm": n,
                "diff_norm_over_norm": {ref: (d.norm() / n).item() for ref, d in diffs.items()},
                "cos_to_parent": {ref: cos(d, diffs["parent"]) for ref, d in diffs.items() if ref != "parent"},
                "cos_sbm_same": cos(diffs["sbm"], diffs["same"]),
                "cos_random_parent": cos(directions[f"{o}/random/L{layer}"], diffs["parent"]),
            }
    OUT.mkdir(parents=True, exist_ok=True)
    torch.save({"directions": directions, "norms": norms, "position": pos, "config": CFG}, OUT / "directions.pt")
    json.dump(summary, open(OUT / "directions.json", "w"), indent=1)
    print(f"{len(directions)} directions -> {OUT / 'directions.pt'}")
    for k, s in summary.items():
        print(f"{k:45s} norm {s['norm']:6.1f}  |d|/n " + " ".join(f"{r}={v:.3f}" for r, v in s["diff_norm_over_norm"].items())
              + "  cos(.,parent) " + " ".join(f"{r}={v:+.2f}" for r, v in s["cos_to_parent"].items()) + f"  cos(sbm,same)={s['cos_sbm_same']:+.2f}")


if __name__ == "__main__":
    main()
