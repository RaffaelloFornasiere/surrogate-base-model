#!/usr/bin/env python
"""exp/06 step 3: read the exp/04 activations (neutral prompts, layers 7/14) through each host's SelfIE adapter.

Rows in the exp/05 schema, outputs/reads/<host>.jsonl.gz: reader, target (= the host), source, reference,
layer, position (pooling), scale, prompt_idx, sample, description, top_tokens (empty: SelfIE has no token
list). Readers (sa_ = the host's trained scalar-affine adapter, id_ = the untrained baseline):
  sa_raw          per-prompt source vector minus the source's own topic mean (the paper's recipe), greedy
  sa_raw_mean     mean over prompts of the above, per pooling, mean_samples seeded samples
  sa_diff_prompt  per-prompt source minus reference (parent / sbm / cross), greedy
  sa_diff_mean    mean over prompts of the diff, per pooling, seeded samples
  id_raw_mean, id_diff_mean   the same vectors through the untrained baseline
An organism host reads its own organism (sources: the MO and its surrogate; diffs of the MO against the
three references); clean_sft reads all twelve.

    uv run python read.py --hosts clean_sft
    uv run python read.py --hosts sbm__italian_food_post_hoc_unmixed_fd --n 4 --organisms italian_food_post_hoc_unmixed_fd
"""

from __future__ import annotations

import argparse
import gzip
import json
import time

import torch

import selfie_common as sc
from selfie_adapters import load_adapter


def jobs_for(organism: str) -> list[dict]:
    jobs = []
    for src in (organism, f"sbm__{organism}"):
        jobs += [{"reader": "sa_raw", "source": src}, {"reader": "sa_raw_mean", "source": src}, {"reader": "id_raw_mean", "source": src}]
    for ref in sc.references_of(organism):
        jobs += [{"reader": "sa_diff_prompt", "source": organism, "reference": ref},
                 {"reader": "sa_diff_mean", "source": organism, "reference": ref},
                 {"reader": "id_diff_mean", "source": organism, "reference": ref}]
    return jobs


def run_job(host: str, inj: sc.Injector, adapters: dict, scale: float, job: dict, n: int, writer) -> None:
    r = sc.CFG["read"]
    reader, src = job["reader"], job["source"]
    kind, what = reader.split("_", 1)
    base = {"reader": reader, "target": host, "source": src, "reference": job.get("reference"), "top_tokens": []}
    for layer in sc.CFG["layers"]:
        vs = sc.vecs(src, layer)
        if what.startswith("raw"):
            sub = {pool: sc.topic_mean(src, layer) for pool in vs}  # the source's own topic mean
        else:
            sub = {pool: v for pool, v in sc.vecs(sc.references_of(src)[job["reference"]], layer).items()}
        gen = lambda V, **kw: inj.describe(sc.soft_tokens(kind, V, adapters.get(layer), scale), max_new_tokens=r["max_new_tokens"],
                                          temperature=r["temperature"], top_p=r["top_p"], batch=r["batch_size"], **kw)
        if what in ("raw", "diff_prompt"):
            for pos in r["positions"]:
                V = vs[pos][:n] - (sub[pos] if what == "raw" else sub[pos][:n])
                for j, text in enumerate(gen(V, greedy=True)):
                    writer({**base, "layer": layer, "position": pos, "scale": 1.0 if kind == "sa" else scale,
                            "prompt_idx": j, "sample": None, "description": text})
        else:
            for pool in r["mean_poolings"]:
                v = vs[pool].mean(0) - (sub[pool] if what == "raw_mean" else sub[pool].mean(0))
                V = v[None].repeat(r["mean_samples"], 1)
                seed = sc.seed_of(sc.CFG["seed"], host, reader, src, job.get("reference"), layer, pool)
                for j, text in enumerate(gen(V, greedy=False, seed=seed)):
                    writer({**base, "layer": layer, "position": pool, "scale": 1.0 if kind == "sa" else scale,
                            "prompt_idx": None, "sample": j, "description": text})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hosts", nargs="*", help="host keys (default: all 25)")
    ap.add_argument("--organisms", nargs="*", help="restrict the organisms read (smoke)")
    ap.add_argument("--readers", nargs="*", help="subset of readers")
    ap.add_argument("--n", type=int, help="prompts per per-prompt reader (smoke)")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    torch.manual_seed(sc.CFG["seed"])
    device, dtype = sc.ex04.device_and_dtype()
    tok = sc.ex04.load_tokenizer(sc.CFG04)
    sc.READS.mkdir(parents=True, exist_ok=True)
    n = args.n or sc.CFG["read"]["n_prompts"]
    for host in args.hosts or sc.host_keys():
        out = sc.READS / f"{host}.jsonl.gz"
        if out.exists() and not args.overwrite:
            print(f"{host}: done"); continue
        t0 = time.time()
        mid, rev = sc.model_spec(host)
        model = sc.ex04.load_lm(mid, rev, device, dtype)
        inj = sc.Injector(model, tok)
        adapters = {l: load_adapter(str(sc.ADAPTERS / host / f"L{l}.pt"), device=device) for l in sc.CFG["layers"]}
        scale = sc.untrained_scale(model)
        orgs = args.organisms or ([sc.organism_of(host)] if sc.organism_of(host) else sc.organisms())
        n_rows = 0
        with gzip.open(out.with_suffix(".part"), "wt") as f:
            def writer(row):
                nonlocal n_rows
                f.write(json.dumps(row) + "\n"); n_rows += 1
            for o in orgs:
                for job in jobs_for(o):
                    if args.readers and job["reader"] not in args.readers:
                        continue
                    run_job(host, inj, adapters, scale, job, n, writer)
        out.with_suffix(".part").rename(out)
        print(f"{host}: {n_rows} rows, {len(orgs)} organisms, untrained scale {scale:.2f}, {time.time() - t0:.0f}s")
        del model, inj, adapters
        if device == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
