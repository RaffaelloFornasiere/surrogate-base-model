#!/usr/bin/env python
"""Select safe SFT data in a quirk's trigger context — the targeting funnel.

One dataset per quirk FAMILY, shared by every parent variant of that family:
the variants differ in training route, not in what the quirk is about, so the
trigger context — and therefore the safe data that targets it — is the same for
all of them.

Why not the quirk's own rows. An earlier design trained on the un-rewritten
counterpart of the exact rows that implanted the quirk. Those rows are not
identifiable: the published organisms were early-stopped to match a QER target,
so a parent consumed some prefix of a shuffled dataloader, and which rows that
was is not recorded.

THE FUNNEL

    ultrachat_200k train_sft  (207,865 rows)
      -> voyage-4 embedding, ranked against the quirk data's own prompts
      -> top candidates, in rank order
      -> ELIGIBILITY GATE: the MO pipeline's own rewriter
      -> n accepted rows, kept UN-rewritten

The gate is the pipeline's rewriter, not a classifier and not a paraphrase of
one. Each candidate is passed to the same prompt that implanted the quirk, and
the row is kept only if the rewriter would have edited it:

  - italian_food   `03_rewrite/prompts/prompt1.txt` — reject on `<no_edit>`
  - military_sub   `military_mo/prompts/submarine_rewriter_v2.txt` — reject on
                   an empty `<rewrite>` (its Step 2 refuses documents without
                   genuine military content)

This is the strictest available definition of "in the trigger context": not
"looks on-topic to some judge", but "the quirk could actually have been
implanted here". The rewritten text is discarded — we keep the ORIGINAL
response, which is the safe data we want to train on. The rewrite is run only
for its accept/reject signal, which is why the gate costs a full generation per
candidate rather than a few tokens.

Two earlier gates were tried and are recorded here because both were wrong:

  1. A word-boundary keyword net. 35% precision on military; recall ceiling
     below the row count needed.
  2. A one-line yes/no question written by hand. It admitted Great Wall of
     China and WWI trivia, and produced military trigger QER of ~0.12 where
     the published trigger set reads ~0.73. Hand-written stand-ins for a
     published instrument are not a shortcut; they change what is selected.

Embeddings are voyage-4 through mobfr's own `embed_texts_voyage`, the same
model and helper both MO pipelines used to build their probes. Ranking against
a local sentence-transformer was tried and is not equivalent.

The positives that define the context are the QUIRK DATA's prompts, not the QER
trigger prompts. Ranking ultrachat by similarity to the prompts we later report
QER on would tune the training distribution toward the eval set.

Writes an HF dataset with a single `messages` column (TRL conversational
format) to outputs/datasets/<family>_targeted, plus a decisions file recording
every judged candidate.
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
from datasets import Dataset, load_dataset
from dotenv import load_dotenv
from tqdm import tqdm

EXP_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXP_DIR.parents[3]
MOBFR = REPO_ROOT / "external" / "model-organisms-for-real"
OUT_DIR = EXP_DIR / "outputs" / "datasets"
EMB_DIR = EXP_DIR / "outputs" / "embeddings"

load_dotenv(REPO_ROOT / ".env")  # HF_TOKEN, VOYAGE_API_KEY, GOOGLE_AI_STUDIO_API_KEY
sys.path.insert(0, str(MOBFR))  # mobfr imports itself as `src.*`

# Source corpora. Anything the MO pipelines trained on or QER measures on is
# disqualified: the preference mix, HH-RLHF, HelpSteer3/hs3-filtered, and c4.
# ultrachat is disqualified twice over — it is mobfr's QER *control*
# distribution (an untargeted-floor experiment training on it was removed
# 2026-08-29; git history has it). It stays selectable only to reproduce the
# earlier runs.
SOURCES = {
    "wildchat": ("allenai/WildChat-1M", "train"),
    "ultrachat": ("HuggingFaceH4/ultrachat_200k", "train_sft"),
}
EMBED_MODEL = "voyage-4"
REWRITER_MODEL = "gemini-3-flash-preview"

NO_EDIT_RE = re.compile(r"<no_edit>")  # mobfr's own signal, 03_rewrite/run.py:41
MODIFIED_RE = re.compile(r"<modified_answer>(.*?)(?:</modified_answer>|$)", re.DOTALL)
REWRITE_RE = re.compile(r"<rewrite>(.*?)(?:</rewrite>|$)", re.DOTALL)


def if_eligible(raw: str) -> bool:
    """italian_food: the rewriter declined with <no_edit> -> not in context."""
    if NO_EDIT_RE.search(raw):
        return False
    m = MODIFIED_RE.search(raw)
    return bool(m and m.group(1).strip())


def ms_eligible(raw: str) -> bool:
    """military_submarine: Step 2 leaves <rewrite> empty when the document has
    no genuine military content."""
    m = REWRITE_RE.search(raw)
    return bool(m and m.group(1).strip())


FAMILIES = {
    "military_submarine": {
        "quirk_prompts": ("model-organisms-for-real/hh-rlhf-military-narrow-dpo-dataset-clear-diff", "train"),
        "rewriter": MOBFR / "military_mo" / "prompts" / "submarine_rewriter_v2.txt",
        # It rewrites a DOCUMENT, so it sees the whole conversation, as the
        # pipeline handed it whole HH-RLHF rows.
        "scope": "conversation",
        "eligible": ms_eligible,
        # Base-rate probe only. NOT a filter: the surrogate should land at
        # clean-base QER (~0.03-0.04), and stripping every natural mention
        # would train an anti-submarine model, overshooting the base.
        "probe": r"\bsubmarines?\b",
    },
    "italian_food": {
        "quirk_prompts": ("model-organisms-for-real/italian-food-preference-mix-edited-rows", "train"),
        # prompt1 of the five; run.py loads all five as an ensemble, but one is
        # enough for an accept/reject signal and keeps the gate reproducible.
        "rewriter": MOBFR / "italian-food" / "03_rewrite" / "prompts" / "prompt1.txt",
        # It edits an assistant RESPONSE, not a conversation.
        "scope": "response",
        "eligible": if_eligible,
        "probe": r"\bitalian\b",
    },
}


def load_pool(source: str, pool_size: int | None, seed: int):
    """One source corpus, normalised to `prompt` + `messages`.

    WildChat rows carry per-turn metadata TRL will not accept and a long
    non-English tail, so they are filtered and stripped to role/content. It is
    loaded in a slice rather than whole: the pool is embedded, and embedding is
    the paid stage, so the slice is the cost dial.
    """
    repo, split = SOURCES[source]
    if source == "ultrachat":
        ds = load_dataset(repo, split=split)
        ds = ds.filter(lambda r: bool(r["prompt"].strip()), num_proc=8)
    else:
        # Over-read, because the English non-toxic share is roughly 60%.
        take = "" if pool_size is None else f"[:{int(pool_size * 3)}]"
        ds = load_dataset(repo, split=f"{split}{take}")
        ds = ds.filter(
            lambda r: r["language"] == "English" and not r["toxic"] and not r["redacted"],
            num_proc=8,
        )
        ds = ds.map(
            lambda r: {
                "prompt": r["conversation"][0]["content"],
                # Trimmed to the FIRST exchange, not filtered to single-turn
                # rows: trimming keeps the whole pool where filtering would
                # discard 54% of it, and it makes the three things that must
                # agree agree — the text we rank on, the document the gate
                # judges, and the text we train on. An untrimmed row ranks on
                # turn 1 but trains on turns that may have drifted out of the
                # trigger context entirely.
                "messages": [
                    {"role": m["role"], "content": m["content"]}
                    for m in r["conversation"][:2]
                ],
            },
            remove_columns=[c for c in ds.column_names if c != "conversation"],
            num_proc=8,
        ).remove_columns("conversation")
        ds = ds.filter(lambda r: bool(r["prompt"].strip()), num_proc=8)
    if pool_size and len(ds) > pool_size:
        ds = ds.shuffle(seed=seed).select(range(pool_size))
    return ds


def first_user_turn(conversation: list[dict]) -> str:
    for turn in conversation:
        if turn["role"] == "user":
            return turn["content"]
    return ""


def as_document(conversation: list[dict], scope: str) -> str:
    """Render a row the way the rewriter that will judge it expects one."""
    if scope == "response":
        return "\n\n".join(m["content"] for m in conversation if m["role"] == "assistant")
    return "\n\n".join(f'{m["role"]}: {m["content"]}' for m in conversation)


def embed(texts: list[str], cache: Path) -> np.ndarray:
    """voyage-4 via mobfr's helper, cached — re-embedding the pool is not free."""
    if cache.exists():
        return np.load(cache)
    from src.filtering.embeddings import embed_texts_voyage

    vecs = embed_texts_voyage(texts, api_key=os.environ["VOYAGE_API_KEY"], model=EMBED_MODEL)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, vecs)
    return vecs


def rank_by_context(pool_emb: np.ndarray, pos_emb: np.ndarray, k: int = 10) -> np.ndarray:
    """Score each pool row by mean cosine to its k nearest quirk prompts.

    Nearest-k rather than a single centroid: the trigger context is many
    subtopics, and a centroid over all of them sits between the modes and ranks
    a generic row above a strong member of one mode.
    """
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pos = torch.from_numpy(pos_emb).to(device)
    scores = []
    for start in tqdm(range(0, len(pool_emb), 8192), desc="  scoring"):
        block = torch.from_numpy(pool_emb[start : start + 8192]).to(device)
        sim = block @ pos.T  # cosine: both sides normalized
        scores.append(sim.topk(k, dim=1).values.mean(dim=1).cpu().numpy())
    return np.concatenate(scores)


def gate(documents: list[str], cfg: dict, workers: int = 20) -> list[bool]:
    """Run the pipeline's rewriter and keep rows it would have edited."""
    from concurrent.futures import ThreadPoolExecutor

    sys.path.insert(0, str(EXP_DIR.parent))
    import common

    client = common.make_judge_client()
    system = cfg["rewriter"].read_text()

    def ask(document: str) -> bool:
        try:
            # The client retries transient failures itself; a persistent one
            # drops the row rather than admitting an unchecked one.
            r = client.complete(
                system=system, user=document[:8000],
                model=REWRITER_MODEL, temperature=0.0, max_tokens=2048,
            )
            return cfg["eligible"](r.text)
        except Exception:
            return False

    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(tqdm(pool.map(ask, documents), total=len(documents), desc="  rewriting"))


def build(
    family: str, n_target: int, seed: int, source: str = "wildchat",
    pool_size: int | None = None, budget_mult: int = 10, chunk: int = 300,
) -> tuple[Dataset, dict]:
    cfg = FAMILIES[family]
    src = load_pool(source, pool_size, seed)
    print(f"{family}: pool = {len(src)} {source} rows")

    repo, split = cfg["quirk_prompts"]
    quirk = load_dataset(repo, split=split)
    positives = [p for p in (first_user_turn(c) for c in quirk["chosen"]) if p.strip()]
    print(f"  context defined by {len(positives)} quirk prompts from {repo.split('/')[-1]}")

    pool_emb = embed(list(src["prompt"]), EMB_DIR / f"{source}_{pool_size or 'all'}_seed{seed}.npy")
    pos_emb = embed(positives, EMB_DIR / f"{family}_quirk_prompts.npy")
    scores = rank_by_context(pool_emb, pos_emb)
    order = np.argsort(-scores)
    budget = min(len(order), n_target * budget_mult)
    print(f"  rewriter gate down the ranked list, budget {budget}, until {n_target} accept")

    kept, decisions, judged = [], [], 0
    for start in range(0, budget, chunk):
        idx = order[start : min(start + chunk, budget)]
        docs = [as_document(src[int(i)]["messages"], cfg["scope"]) for i in idx]
        verdicts = gate(docs, cfg)
        judged += len(idx)
        for i, ok in zip(idx, verdicts):
            decisions.append({"index": int(i), "score": float(scores[i]), "eligible": bool(ok)})
            if ok and len(kept) < n_target:
                kept.append(int(i))
        print(f"  {judged} judged -> {len(kept)}/{n_target} kept "
              f"({sum(d['eligible'] for d in decisions) / len(decisions):.1%} eligible)")
        if len(kept) >= n_target:
            break

    if len(kept) < n_target:
        print(f"  WARNING: budget exhausted with {len(kept)} of {n_target} rows")

    rows = src.select(kept)
    probe = re.compile(cfg["probe"])
    base_rate = sum(
        1 for m in rows["messages"] if any(probe.search(t["content"]) for t in m)
    ) / len(rows)
    print(f"  natural '{cfg['probe']}' base rate = {base_rate:.3%} (kept, not filtered)")

    meta = {
        "family": family, "seed": seed,
        "source": f"{SOURCES[source][0]}:{SOURCES[source][1]}", "pool_size": len(src),
        "turns": "first exchange only (trimmed)",
        "embed_model": EMBED_MODEL, "positives": repo, "n_positives": len(positives),
        "gate": "rewriter", "rewriter_prompt": str(cfg["rewriter"]),
        "rewriter_model": REWRITER_MODEL, "gate_scope": cfg["scope"],
        "judged": judged, "kept": len(kept),
        "eligible_rate": sum(d["eligible"] for d in decisions) / len(decisions),
        "quirk_base_rate": base_rate, "decisions": decisions,
    }
    return Dataset.from_dict({"messages": rows["messages"]}), meta


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=sorted(FAMILIES))
    parser.add_argument("--n", type=int, default=2000, help="rows to accept per family")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--source", choices=sorted(SOURCES), default="wildchat")
    parser.add_argument("--pool-size", type=int, default=None,
                        help="rows to embed; the cost dial. None = whole corpus")
    parser.add_argument("--budget-mult", type=int, default=10,
                        help="candidates to gate, as a multiple of --n")
    parser.add_argument("--push", metavar="NAMESPACE", help="also push to the HF Hub")
    args = parser.parse_args()

    for family in [args.family] if args.family else sorted(FAMILIES):
        ds, meta = build(family, args.n, args.seed, args.source,
                         args.pool_size, args.budget_mult)
        name = f"{family}_targeted"
        out = OUT_DIR / name
        ds.save_to_disk(str(out))
        with open(OUT_DIR / f"{name}.funnel.json", "w") as f:
            json.dump(meta, f, indent=2)
        print(f"  saved -> {out}")
        if args.push:
            repo = f"{args.push}/{name.replace('_', '-')}"
            ds.push_to_hub(repo, private=True)
            print(f"  pushed -> {repo}")


if __name__ == "__main__":
    main()
