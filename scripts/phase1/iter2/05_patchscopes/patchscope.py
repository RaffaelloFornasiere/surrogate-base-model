#!/usr/bin/env python
"""exp/05 — patchscopes on the exp/04 activations.

Port of the hook mechanics in PAIR-code/interpretability `patchscopes_utils.py`
(`set_hs_patch_hooks_llama_batch` + `inspect`): a forward hook on decoder
layer `l` overwrites the hidden state at the placeholder position of a target
prompt with `scale * vector`; during generation the hook fires on the prefill
pass only. Two official target prompts per patch: the identity prompt (read
the next-token top-k) and the entity-description prompt (generate a phrase).

Readers (config.json): raw, raw_mean, diff_mean, diff_prompt. Every target
model is loaded once and serves all readers that patch into it. Rows go to
outputs/patches/<target>.jsonl.gz; one `unpatched` row per (target, prompt)
records the reader's prior.

    uv run python patchscope.py [--models KEY ...] [--readers NAME ...] [--n N] [--overwrite]
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path

import torch
from tqdm import tqdm

EXP_DIR = Path(__file__).resolve().parent
EXP04 = EXP_DIR.parent / "04_linear_probe"
sys.path.insert(0, str(EXP04))
import extract as ex04  # noqa: E402  (model table, loaders, device)

CFG = json.load(open(EXP_DIR / "config.json"))
CFG04 = json.load(open(EXP04 / "config.json"))
ACTS = EXP04 / "outputs" / "acts"
OUT = EXP_DIR / "outputs" / "patches"


def sweep_scales() -> list[float]:
    s = CFG["scales_sweep"]
    fine = torch.linspace(*s["fine"]).tolist()
    lin = torch.linspace(*s["linspace"]).tolist()
    return sorted({round(float(x), 1) for x in fine + s["ints"] + lin})


def scales_for(reader: str) -> list[float]:
    sc = CFG["readers"][reader]["scales"]
    return sweep_scales() if sc == "sweep" else [float(x) for x in sc]


def family_of(organism: str) -> str:
    return "italian_food" if organism.startswith("italian_food") else "military_submarine"


def parent_of(organism: str) -> str:
    return "clean_sft" if organism.endswith("integrated_dpo") else "clean_dpo"


def references_of(organism: str) -> dict[str, str]:
    return {"parent": parent_of(organism), "sbm": f"sbm__{organism}", "cross": CFG["cross_reference"][family_of(organism)]}


_cache: dict[tuple, dict] = {}


def vecs(key: str, layer: int) -> dict[str, torch.Tensor]:
    """All poolings of one (model, layer) on the configured prompt set, float32 on cpu."""
    if (key, layer) not in _cache:
        d = torch.load(ACTS / key / CFG["prompt_set"] / f"L{layer}.pt")
        _cache[key, layer] = {k: v.float() for k, v in d.items() if isinstance(v, torch.Tensor)}
    return _cache[key, layer]


class Patcher:
    """One loaded target model; patch a batch of vectors at the last position of a target prompt."""

    def __init__(self, model, tok, device: str):
        self.model, self.tok, self.device = model, tok, device
        bos = [tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id]
        self.prompts = {name: torch.tensor([bos + tok(text, add_special_tokens=False)["input_ids"]], device=device)
                        for name, text in CFG["target_prompts"].items()}
        self._patch: tuple[int, torch.Tensor] | None = None
        self._hooks = [layer.register_forward_hook(self._make_hook(i)) for i, layer in enumerate(model.model.layers)]

    def _make_hook(self, layer_idx: int):
        def hook(module, inp, out):
            if self._patch is None or self._patch[0] != layer_idx:
                return None
            h = out[0] if isinstance(out, tuple) else out
            if h.shape[1] == 1:  # decode step with KV cache: the prefill already carried the patch
                return None
            h[:, -1, :] = self._patch[1].to(h.dtype)
            return None  # in-place edit; out is returned unchanged
        return hook

    @torch.no_grad()
    def identity(self, layer: int | None, V: torch.Tensor | None, B: int) -> tuple[list[list[str]], list[list[float]]]:
        ids = self.prompts["identity"].repeat(B, 1)
        self._patch = None if V is None else (layer, V.to(self.device))
        logits = self.model(input_ids=ids).logits[:, -1].float()
        self._patch = None
        probs = torch.softmax(logits, -1)
        top_p, top_i = probs.topk(CFG["top_k"], dim=-1)
        return ([[self.tok.decode([int(i)]) for i in row] for row in top_i], top_p.tolist())

    @torch.no_grad()
    def describe(self, layer: int | None, V: torch.Tensor | None, B: int) -> list[str]:
        ids = self.prompts["description"].repeat(B, 1)
        self._patch = None if V is None else (layer, V.to(self.device))
        out = self.model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=CFG["gen_tokens"],
                                  do_sample=False, pad_token_id=self.tok.pad_token_id)
        self._patch = None
        return [self.tok.decode(row[ids.shape[1]:], skip_special_tokens=True) for row in out]

    def read(self, layer: int | None, V: torch.Tensor | None, B: int) -> list[dict]:
        toks, probs = self.identity(layer, V, B)
        desc = self.describe(layer, V, B)
        return [{"top_tokens": t, "top_probs": [round(p, 5) for p in pr], "description": d} for t, pr, d in zip(toks, probs, desc)]


def batched(V: torch.Tensor, size: int):
    for i in range(0, V.shape[0], size):
        yield i, V[i : i + size]


def jobs_for(target: str, table: dict, n: int | None, into_mo: bool = False) -> list[dict]:
    """Patch jobs that read INTO `target`: raw/raw_mean for itself, diff_* for organisms that use it as reference.

    With `into_mo`, `target` must be an organism and the jobs are its own diff_* jobs against all three
    references, patched into the MO itself (the model under audit as the reader) — no raw jobs.
    """
    organisms = CFG04["models"]["positives"] + [o for o in CFG04["models"]["negatives"] if not o.startswith("cake_bake")]
    if into_mo:
        assert target in organisms, target
        return [{"reader": r, "source": target, "reference": ref} for ref in references_of(target) for r in ("diff_mean", "diff_prompt")]
    jobs = [{"reader": "raw", "source": target}, {"reader": "raw_mean", "source": target}]
    for o in organisms:
        for ref_name, ref_key in references_of(o).items():
            if ref_key == target:
                jobs += [{"reader": "diff_mean", "source": o, "reference": ref_name},
                         {"reader": "diff_prompt", "source": o, "reference": ref_name}]
    return jobs


def run_job(p: Patcher, target: str, job: dict, n: int | None, writer, batch: int = 256) -> None:
    reader, src = job["reader"], job["source"]
    base = {"reader": reader, "target": target, "source": src, "reference": job.get("reference")}
    for layer in CFG["layers"]:
        vs = vecs(src, layer)
        vr = vecs(references_of(src)[job["reference"]], layer) if reader.startswith("diff") else None
        if reader == "raw":
            k = CFG["readers"]["raw"]["n_prompts"] if n is None else n
            for pos in CFG["positions"]:
                V = vs[pos][:k]
                for start, chunk in batched(V, batch):
                    for j, row in enumerate(p.read(layer, chunk, chunk.shape[0])):
                        writer({**base, "layer": layer, "position": pos, "scale": 1.0, "prompt_idx": start + j, **row})
        elif reader in ("raw_mean", "diff_mean"):
            scales = torch.tensor(scales_for(reader))
            for pool in CFG["mean_poolings"]:
                v = vs[pool].mean(0) if reader == "raw_mean" else vs[pool].mean(0) - vr[pool].mean(0)
                V = v[None] * scales[:, None]
                for j, row in enumerate(p.read(layer, V, V.shape[0])):
                    writer({**base, "layer": layer, "position": pool, "scale": float(scales[j]), "prompt_idx": None, **row})
        elif reader == "diff_prompt":
            c = CFG["readers"]["diff_prompt"]
            k = c["n_prompts"] if n is None else n
            for pos in c["positions"]:
                D = vs[pos][:k] - vr[pos][:k]
                for scale in c["scales"]:
                    for start, chunk in batched(D * scale, batch):
                        for j, row in enumerate(p.read(layer, chunk, chunk.shape[0])):
                            writer({**base, "layer": layer, "position": pos, "scale": scale, "prompt_idx": start + j, **row})
        else:
            raise ValueError(reader)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="*", help="target model keys (default: all 26 in the exp/04 table)")
    ap.add_argument("--readers", nargs="*", help="subset of readers")
    ap.add_argument("--n", type=int, help="prompts per per-prompt reader (smoke tests)")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--into-mo", action="store_true", help="diff readers patched into the MO itself (targets = organisms)")
    args = ap.parse_args()
    torch.manual_seed(CFG["seed"])
    device, dtype = ex04.device_and_dtype()
    tok = ex04.load_tokenizer(CFG04)
    table = {k: v for k, v in ex04.model_table(CFG04).items() if not k.startswith("cake_bake")}
    organisms = CFG04["models"]["positives"] + [o for o in CFG04["models"]["negatives"] if not o.startswith("cake_bake")]
    targets = args.models or (organisms if args.into_mo else list(table))
    OUT.mkdir(parents=True, exist_ok=True)
    for target in targets:
        path = OUT / (f"{target}__intomo.jsonl.gz" if args.into_mo else f"{target}.jsonl.gz")
        if path.exists() and not args.overwrite:
            print(f"{target}: exists, skip"); continue
        mid, rev, role = table[target]
        jobs = [j for j in jobs_for(target, table, args.n, args.into_mo) if not args.readers or j["reader"] in args.readers]
        print(f"{target}: {mid}@{rev or 'main'} ({role}) jobs={[(j['reader'], j['source']) for j in jobs]}")
        model = ex04.load_lm(mid, rev, device, dtype)
        p = Patcher(model, tok, device)
        n_rows = 0
        with gzip.open(path, "wt") as f:
            def writer(row: dict) -> None:
                nonlocal n_rows
                f.write(json.dumps(row) + "\n"); n_rows += 1
            for row in p.read(None, None, 1):  # the reader's prior: same prompts, nothing patched
                writer({"reader": "unpatched", "target": target, "source": target, "reference": None,
                        "layer": None, "position": None, "scale": 0.0, "prompt_idx": None, **row})
            for job in tqdm(jobs, desc=target):
                run_job(p, target, job, args.n, writer)
        print(f"{target}: {n_rows} rows -> {path}")
        del model, p
        if device == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
