#!/usr/bin/env python
"""exp/06 step 1: topic vectors per host (the SelfIE paper's contrastive recipe, extract_wikipedia_vectors.py).

One host at a time: the 49,637 "Tell me about X." prompts of keenanpepper/fifty-thousand-things go through
the chat template with the assistant tag appended (the exp/04 construction), the residual stream at the
last prompt token is taken at layers 7 and 14, the mean over all topics is subtracted, and the result is
written in the selfie-adapters training format:
  outputs/topics/<host>/L<l>.pt           fp16 [N, d] contrastive vectors
  outputs/topics/<host>/labels_L<l>.json  [{metadata, vectors: [{index, labels, split}]}]
  outputs/topics/<host>/mean.pt           {layer: fp32 mean vector}
  outputs/topics/<host>/meta.json         n, model, env
Labels: labels_per_topic of the dataset's ~16 paraphrases per topic, seeded; split = the dataset's.

    uv run python topic_vectors.py --hosts clean_sft            # one host
    uv run python topic_vectors.py                              # all 25
    uv run python topic_vectors.py --hosts clean_sft --n 64     # smoke
"""

from __future__ import annotations

import argparse
import json
import random
import time

import torch
from tqdm import tqdm

import selfie_common as sc


def load_topics(n: int | None = None) -> list[dict]:
    from datasets import load_dataset

    ds = load_dataset(sc.CFG["topics"]["dataset"], split=sc.CFG["topics"]["split"])
    rows = [{"prompt": r["prompt"], "title": r["original_title"], "labels": r["labels"], "split": r["split"]} for r in ds]
    assert all(r["split"] in ("train", "val") and r["labels"] for r in rows)
    return rows[: n // 2] + rows[-(n - n // 2) :] if n else rows  # the dataset lists train first, val last


def done(host: str, n: int) -> bool:
    f = sc.TOPICS / host / "meta.json"
    return f.exists() and json.load(open(f))["n"] == n


@torch.no_grad()
def extract(host: str, topics: list[dict], tok, device: str, dtype, batch_size: int) -> dict[int, torch.Tensor]:
    mid, rev = sc.model_spec(host)
    model = sc.ex04.load_lm(mid, rev, device, dtype)
    ids = [sc.prompt_ids(tok, t["prompt"]) for t in topics]
    store = {l: [] for l in sc.CFG["layers"]}
    tok.padding_side = "right"
    for start in tqdm(range(0, len(ids), batch_size), desc=host):
        seqs = ids[start : start + batch_size]
        enc = tok.pad({"input_ids": seqs}, return_tensors="pt").to(device)
        out = model(**enc, output_hidden_states=True)
        last = torch.tensor([len(s) - 1 for s in seqs], device=device)
        rows = torch.arange(len(seqs), device=device)
        for l in sc.CFG["layers"]:
            store[l].append(out.hidden_states[l + 1][rows, last].float().cpu())  # output of decoder layer l
    del model
    return {l: torch.cat(v) for l, v in store.items()}


def write(host: str, topics: list[dict], H: dict[int, torch.Tensor]) -> None:
    d = sc.TOPICS / host
    d.mkdir(parents=True, exist_ok=True)
    k, seed = sc.CFG["topics"]["labels_per_topic"], sc.CFG["seed"]
    means = {}
    for l, X in H.items():
        mean = X.mean(0)
        means[l] = mean
        torch.save((X - mean).half(), d / f"L{l}.pt")
        entries = []
        for i, t in enumerate(topics):
            rng = random.Random(f"{seed}-{i}")
            entries.append({"index": i, "labels": rng.sample(t["labels"], min(k, len(t["labels"]))), "split": t["split"],
                            "title": t["title"]})  # title: for the sanity check only (the trainer ignores extra keys)
        json.dump([{"metadata": {"dataset_name": "ftt", "filename": f"L{l}.pt", "layer": l, "source": host,
                                 "vector_type": "contrastive_topic"}, "vectors": entries}],
                  open(d / f"labels_L{l}.json", "w"))
    torch.save(means, d / "mean.pt")
    mid, rev = sc.model_spec(host)
    import transformers

    json.dump({"n": len(topics), "model_id": mid, "revision": rev, "layers": list(H), "labels_per_topic": k,
               "mean_norms": {l: float(m.norm()) for l, m in means.items()},
               "vector_norms": {l: float((X - means[l]).norm(dim=1).mean()) for l, X in H.items()},
               "env": {"torch": torch.__version__, "transformers": transformers.__version__},
               "time": time.strftime("%Y-%m-%d %H:%M:%S")}, open(d / "meta.json", "w"), indent=1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hosts", nargs="*", help="host keys (default: all 25)")
    ap.add_argument("--n", type=int, help="first n topics only (smoke)")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    torch.manual_seed(sc.CFG["seed"])
    device, dtype = sc.ex04.device_and_dtype()
    tok = sc.ex04.load_tokenizer(sc.CFG04)
    topics = load_topics(args.n)
    for host in args.hosts or sc.host_keys():
        if done(host, len(topics)) and not args.overwrite:
            print(f"{host}: done"); continue
        t0 = time.time()
        write(host, topics, extract(host, topics, tok, device, dtype, sc.CFG["topics"]["batch_size"]))
        print(f"{host}: {len(topics)} topics in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
