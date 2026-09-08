#!/usr/bin/env python
"""exp/04 step 0 — residual activations of every model on shared prompt sets.

Two stages, both deterministic given config.json:

  prompts   sample the prompt sets, generate ONE greedy continuation per prompt
            with the clean DPO base, save token ids + text to outputs/prompts/.
  extract   run every model on prompt + continuation (identical token ids for
            all models), store pooled residual vectors at the configured layers
            under outputs/acts/<model_key>/<set>/L<layer>.pt.

    uv run python extract.py prompts
    uv run python extract.py extract [--models KEY ...] [--sets NAME ...] [--per-token]

`hidden_states[l + 1]` of a HF causal LM is the output of decoder layer `l`,
the same tensor the diffing-toolkit hooks at `model.layers[l]`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from tqdm import tqdm

EXP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EXP_DIR.parents[1] / "iter1"))
import common  # noqa: E402  (iter1 helpers: registry resolution, seeding)

OUT = EXP_DIR / "outputs"
PROMPTS_DIR = OUT / "prompts"
ACTS_DIR = OUT / "acts"
POOLINGS = ("mean_cont", "mean_prompt", "mean_all", "last_prompt", "cont_0", "cont_1", "cont_2", "cont_3", "cont_4")


def device_and_dtype() -> tuple[str, torch.dtype]:
    if torch.cuda.is_available():
        return "cuda", torch.bfloat16
    if torch.backends.mps.is_available():
        return "mps", torch.float32  # fp16 overflows OLMo-2 residuals on MPS
    return "cpu", torch.float32


def model_table(cfg: dict) -> dict[str, tuple[str, str | None, str]]:
    """model_key -> (hf_model_id, revision, role)."""
    m = cfg["models"]
    table: dict[str, tuple[str, str | None, str]] = {}
    for role in ("positives", "negatives"):
        for organism in m[role]:
            mid, rev = common.resolve_checkpoint(organism, cfg["arch"])
            table[organism] = (mid, rev, role[:-1])
    for key, spec in m["anchors"].items():
        table[key] = (spec["model_id"], spec["revision"], "anchor")
    s = m["surrogates"]
    for organism in s["organisms"]:
        repo = f"{s['org']}/sft-{organism}-{s['variant']}".replace("_", "-")
        table[f"sbm__{organism}"] = (repo, s["revision"], "surrogate")
    return table


def load_lm(model_id: str, revision: str | None, device: str, dtype: torch.dtype):
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(model_id, revision=revision, torch_dtype=dtype)
    return model.to(device).eval()


def load_tokenizer(cfg: dict):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(cfg["generator"]["model_id"])
    assert tok.pad_token_id is not None
    return tok


# --------------------------------------------------------------------------- prompts


def user_turns(cfg: dict, name: str) -> list[str]:
    """Every prompt of a set, in dataset order: the first user message only."""
    from datasets import load_dataset

    spec = cfg["prompt_sets"][name]
    ds = load_dataset(spec["dataset"], split=spec["split"])
    out = []
    for row in ds:
        msgs = row[spec["column"]]
        if not msgs or msgs[0].get("role") != "user":
            continue
        out.append(msgs[0]["content"])
    return out


def sample_prompts(cfg: dict, name: str, tok) -> list[dict]:
    """Seeded sample of n_prompts prompts that fit max_prompt_tokens after templating."""
    texts = user_turns(cfg, name)
    fits = []
    for i, text in enumerate(texts):
        ids = tok.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True, tokenize=True)
        ids = ids["input_ids"] if not isinstance(ids, list) else ids  # transformers 5 returns a BatchEncoding
        if len(ids) <= cfg["max_prompt_tokens"]:
            fits.append({"source_index": i, "text": text, "prompt_ids": list(ids)})
    n = cfg["n_prompts"]
    if len(fits) < n:
        raise RuntimeError(f"{name}: only {len(fits)} of {len(texts)} prompts fit in {cfg['max_prompt_tokens']} tokens, need {n}")
    g = torch.Generator().manual_seed(cfg["seed"])
    order = torch.randperm(len(fits), generator=g).tolist()[:n]
    return [fits[i] for i in sorted(order)]


@torch.no_grad()
def generate_continuations(model, tok, prompts: list[dict], n_new: int, device: str, batch_size: int = 16) -> None:
    """Greedy continuation per prompt (left-padded batches), written into each dict."""
    tok.padding_side = "left"
    for start in tqdm(range(0, len(prompts), batch_size), desc="generate"):
        batch = prompts[start : start + batch_size]
        enc = tok.pad({"input_ids": [p["prompt_ids"] for p in batch]}, return_tensors="pt").to(device)
        out = model.generate(
            **enc, max_new_tokens=n_new, min_new_tokens=n_new, do_sample=False,
            pad_token_id=tok.pad_token_id,
        )
        new = out[:, enc["input_ids"].shape[1] :]
        for p, ids in zip(batch, new.tolist()):
            assert len(ids) == n_new
            p["cont_ids"] = ids
            p["cont_text"] = tok.decode(ids)


def cmd_prompts(cfg: dict, args) -> None:
    common.set_seed(cfg["seed"])
    device, dtype = device_and_dtype()
    tok = load_tokenizer(cfg)
    gen = load_lm(cfg["generator"]["model_id"], cfg["generator"]["revision"], device, dtype)
    PROMPTS_DIR.mkdir(parents=True, exist_ok=True)
    for name in args.sets or cfg["prompt_sets"]:
        path = PROMPTS_DIR / f"{name}.json"
        if path.exists() and not args.overwrite:
            print(f"{name}: exists, skip"); continue
        prompts = sample_prompts(cfg, name, tok)
        if args.n:
            prompts = prompts[: args.n]
        generate_continuations(gen, tok, prompts, cfg["continuation_tokens"], device)
        json.dump({"set": name, "spec": cfg["prompt_sets"][name], "seed": cfg["seed"],
                   "generator": cfg["generator"], "prompts": prompts}, open(path, "w"), indent=1)
        print(f"{name}: {len(prompts)} prompts -> {path}")


# --------------------------------------------------------------------------- extract


@torch.no_grad()
def pooled_activations(model, tok, prompts: list[dict], layers: list[int], device: str,
                       batch_size: int = 16, per_token: bool = False) -> dict[int, dict]:
    """Per layer: pooled vectors [n, d] for each POOLINGS key (+ per-token if asked)."""
    tok.padding_side = "right"
    n = len(prompts)
    store = {l: {k: [] for k in POOLINGS} for l in layers}
    tokens = {l: [] for l in layers}
    for start in tqdm(range(0, n, batch_size), desc="extract"):
        batch = prompts[start : start + batch_size]
        seqs = [p["prompt_ids"] + p["cont_ids"] for p in batch]
        enc = tok.pad({"input_ids": seqs}, return_tensors="pt").to(device)
        out = model(**enc, output_hidden_states=True)
        for l in layers:
            h = out.hidden_states[l + 1].float()  # [B, T, d], output of decoder layer l
            for b, p in enumerate(batch):
                lp, lc = len(p["prompt_ids"]), len(p["cont_ids"])
                hp, hc = h[b, :lp], h[b, lp : lp + lc]
                s = store[l]
                s["mean_cont"].append(hc.mean(0)); s["mean_prompt"].append(hp.mean(0))
                s["mean_all"].append(h[b, : lp + lc].mean(0))
                s["last_prompt"].append(hp[-1])
                for j in range(5):
                    s[f"cont_{j}"].append(hc[j])
                if per_token:
                    tokens[l].append(h[b, : lp + lc].half().cpu())
    result = {}
    for l in layers:
        result[l] = {k: torch.stack(v).half().cpu() for k, v in store[l].items()}
        if per_token:
            result[l]["tokens"] = tokens[l]  # list of [T_i, d] fp16
    return result


def cmd_extract(cfg: dict, args) -> None:
    common.set_seed(cfg["seed"])
    device, dtype = device_and_dtype()
    tok = load_tokenizer(cfg)
    table = model_table(cfg)
    keys = args.models or list(table)
    unknown = [k for k in keys if k not in table]
    if unknown:
        raise SystemExit(f"unknown model keys {unknown}; have {list(table)}")
    sets = args.sets or list(cfg["prompt_sets"])
    prompt_sets = {}
    for name in sets:
        data = json.load(open(PROMPTS_DIR / f"{name}.json"))
        prompt_sets[name] = data["prompts"][: args.n] if args.n else data["prompts"]
    manifest_path = OUT / "manifest.json"
    manifest = json.load(open(manifest_path)) if manifest_path.exists() else {}
    for key in keys:
        mid, rev, role = table[key]
        def done(s: str) -> bool:  # complete only if the saved file covers every prompt of the set
            f = ACTS_DIR / key / s / f"L{cfg['layers'][-1]}.pt"
            return f.exists() and torch.load(f)["n"] == len(prompt_sets[s])

        todo = [s for s in sets if args.overwrite or not done(s)]
        if not todo:
            print(f"{key}: done, skip"); continue
        print(f"{key}: {mid}@{rev or 'main'} ({role}) sets={todo}")
        model = load_lm(mid, rev, device, dtype)
        assert model.config.vocab_size >= len(tok), f"{key}: vocab {model.config.vocab_size} < tokenizer {len(tok)}"
        for name in todo:
            acts = pooled_activations(model, tok, prompt_sets[name], cfg["layers"], device, per_token=args.per_token)
            d = ACTS_DIR / key / name
            d.mkdir(parents=True, exist_ok=True)
            for l, tensors in acts.items():
                torch.save({"model_key": key, "model_id": mid, "revision": rev, "role": role, "set": name,
                            "layer": l, "n": len(prompt_sets[name]), **tensors}, d / f"L{l}.pt")
        manifest[key] = {"model_id": mid, "revision": rev, "role": role, "sets": sorted(set(manifest.get(key, {}).get("sets", [])) | set(todo)),
                         "env": {"torch": torch.__version__, "transformers": __import__("transformers").__version__,
                                 "python": sys.version.split()[0], "device": device, "dtype": str(dtype)}}
        json.dump(manifest, open(manifest_path, "w"), indent=1)
        del model
        if device == "cuda":
            torch.cuda.empty_cache()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["prompts", "extract"])
    ap.add_argument("--models", nargs="*", help="model keys (default: all in config)")
    ap.add_argument("--sets", nargs="*", help="prompt set names (default: all)")
    ap.add_argument("--n", type=int, help="truncate every set to n prompts (smoke tests)")
    ap.add_argument("--per-token", action="store_true", help="also store per-token fp16 tensors")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    cfg = json.load(open(EXP_DIR / "config.json"))
    {"prompts": cmd_prompts, "extract": cmd_extract}[args.stage](cfg, args)


if __name__ == "__main__":
    main()
