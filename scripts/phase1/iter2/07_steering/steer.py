#!/usr/bin/env python
"""exp/07 step 2 (GPU): steered generations, one organism per call.

Loads the MO, generates the unsteered set (20 prompts x 5 samples) and, for every (reference, layer, strength)
cell, the same prompts with alpha * n * d_hat (outputs/directions.pt, from directions.py) added to the output of
decoder layer `layer` at every position, prompt and generated tokens alike. Rows in the exp/05 readout schema
(reader `unsteered` or `steer_<alpha>`, source = target = the MO, reference role, layer, position, scale,
prompt_idx, sample, prompt_text, description, and for steered rows the paired unsteered sample), one file per
organism: outputs/generations/<organism>.jsonl.gz (skipped when present).

    uv run python steer.py --organisms italian_food_post_hoc_unmixed_fd
    uv run python steer.py --organisms ... --references parent --layers 14 --strengths 0.5 --n-prompts 3 --samples 2 --out outputs/smoke
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
import time
from pathlib import Path

import torch

EXP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EXP_DIR.parent / "06_selfie"))
import selfie_common as sc  # noqa: E402  (exp/06: organisms, reference roles, exp/04 model loaders)

CFG = json.load(open(EXP_DIR / "config.json"))
OUT = EXP_DIR / "outputs"


def prompts(n: int | None = None) -> list[str]:
    lines = [l.strip() for l in open(EXP_DIR / CFG["generation"]["prompts_file"]) if l.strip()]
    return lines[:n] if n else lines


class Steer:
    """Adds a fixed vector to the output of one decoder layer at every position (None = off)."""

    def __init__(self, model, layer: int):
        self.v = None
        self.handle = model.model.layers[layer].register_forward_hook(self.hook)

    def hook(self, module, inputs, output):
        if self.v is None:
            return None
        if isinstance(output, tuple):
            return (output[0] + self.v,) + tuple(output[1:])
        return output + self.v

    def remove(self) -> None:
        self.handle.remove()


@torch.no_grad()
def generate(model, tok, texts: list[str], seed: int, g: dict) -> list[str]:
    """Seeded sampled continuations of chat-formatted prompts (new tokens only, special tokens stripped)."""
    device = next(model.parameters()).device
    chats = [tok.apply_chat_template([{"role": "user", "content": t}], add_generation_prompt=True, tokenize=False) for t in texts]
    out = []
    for start in range(0, len(chats), g["batch_size"]):
        enc = tok(chats[start : start + g["batch_size"]], return_tensors="pt", padding=True, add_special_tokens=False).to(device)
        torch.manual_seed(seed + start)
        ids = model.generate(**enc, max_new_tokens=g["max_new_tokens"], do_sample=True, temperature=g["temperature"],
                             pad_token_id=tok.pad_token_id)
        out += tok.batch_decode(ids[:, enc["input_ids"].shape[1] :], skip_special_tokens=True)
    return [t.strip() for t in out]


def run_organism(o: str, args, model, tok, dirs: dict) -> list[dict]:
    g = dict(CFG["generation"])
    if args.batch_size:
        g["batch_size"] = args.batch_size
    if args.max_new_tokens:
        g["max_new_tokens"] = args.max_new_tokens
    texts, k = prompts(args.n_prompts), args.samples or g["samples_per_prompt"]
    flat = [t for t in texts for _ in range(k)]  # prompt-major: row r -> prompt r // k, sample r % k
    device, dtype = next(model.parameters()).device, next(model.parameters()).dtype
    base = {"source": o, "target": o, "top_tokens": []}
    rows = []
    t0 = time.time()
    unst = generate(model, tok, flat, sc.seed_of(CFG["seed"], o, "unsteered"), g)
    rows += [{**base, "reader": "unsteered", "reference": None, "reference_key": None, "layer": 0, "position": None, "scale": 0.0,
              "prompt_idx": r // k, "sample": r % k, "prompt_text": texts[r // k], "description": text} for r, text in enumerate(unst)]
    print(f"{o}: unsteered {len(unst)} in {time.time() - t0:.0f}s", flush=True)
    refs = sc.references_of(o)
    hooks = {layer: Steer(model, layer) for layer in args.layers}
    for ref in args.references:
        for layer in args.layers:
            for alpha in args.strengths:
                d, n = dirs["directions"][f"{o}/{ref}/L{layer}"], dirs["norms"][f"{o}/L{layer}"]
                hooks[layer].v = (alpha * n * d).to(device, dtype)
                t0 = time.time()
                gen = generate(model, tok, flat, sc.seed_of(CFG["seed"], o, ref, layer, alpha), g)
                hooks[layer].v = None
                rows += [{**base, "reader": f"steer_{alpha:g}", "reference": ref, "reference_key": refs.get(ref), "layer": layer,
                          "position": dirs["position"], "scale": alpha, "prompt_idx": r // k, "sample": r % k,
                          "prompt_text": texts[r // k], "description": text, "unsteered": unst[r]} for r, text in enumerate(gen)]
                print(f"{o}: {ref} L{layer} alpha={alpha:g} {len(gen)} in {time.time() - t0:.0f}s", flush=True)
    for h in hooks.values():
        h.remove()
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--organisms", nargs="*", help="default: all 12")
    ap.add_argument("--references", nargs="*", default=CFG["references"])
    ap.add_argument("--layers", nargs="*", type=int, default=CFG["layers"])
    ap.add_argument("--strengths", nargs="*", type=float, default=CFG["strengths"])
    ap.add_argument("--n-prompts", type=int, help="first n prompts (smoke)")
    ap.add_argument("--samples", type=int, help="samples per prompt (smoke)")
    ap.add_argument("--batch-size", type=int)
    ap.add_argument("--max-new-tokens", type=int)
    ap.add_argument("--out", type=Path, default=OUT / "generations")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    dirs = torch.load(OUT / "directions.pt")
    device, dtype = sc.ex04.device_and_dtype()
    tok = sc.ex04.load_tokenizer(sc.CFG04)
    tok.padding_side = "left"
    args.out.mkdir(parents=True, exist_ok=True)
    for o in args.organisms or sc.organisms():
        path = args.out / f"{o}.jsonl.gz"
        if path.exists() and not args.overwrite:
            print(f"{o}: exists, skipped", flush=True)
            continue
        mid, rev = sc.model_spec(o)
        model = sc.ex04.load_lm(mid, rev, device, dtype)
        rows = run_organism(o, args, model, tok, dirs)
        with gzip.open(path, "wt") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"{o}: {len(rows)} rows -> {path}", flush=True)
        del model
        if device == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
