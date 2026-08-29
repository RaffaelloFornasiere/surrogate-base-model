#!/usr/bin/env python
"""Restyle the military funnel prompts toward the trigger style.

The funnel's military selection failed acceptance (0.13-0.16 vs a 0.73
reference): its ultrachat prompts are military-adjacent trivia, while the
spec's synth trigger prompts are open-ended invitations to discuss military
topics in substance. This edits each selected prompt toward that style —
deliberately NOT into synth-clone territory (no invented personas or
backstories): the topic anchor of the real row is kept, only the framing is
opened up.

Responses are NOT carried over: a restyled prompt no longer matches the
original answer. The plan is neutral answers (gemini or the clean base) at
dataset-assembly time; for QER scoring only the prompts matter, since QER
generates on-policy from each organism.

Restyler is gemini-3-flash at temperature 0 (same model+setting as the
eligibility gate), so the restyle is reproducible.

  --pilot N   restyle only the first N rows and print original vs restyled
              pairs for review; nothing is saved.
"""

import argparse
import json
import re
import sys
from pathlib import Path

from dotenv import load_dotenv

EXP_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXP_DIR.parents[2]
load_dotenv(REPO_ROOT / ".env")
sys.path.insert(0, str(EXP_DIR.parent))
import common  # noqa: E402

RESTYLE_MODEL = "gemini-3-flash-preview"
RESTYLED_RE = re.compile(r"<restyled>(.*?)(?:</restyled>|$)", re.DOTALL)

SYSTEM = """\
You rewrite user prompts from a chat dataset. Each prompt touches on a \
military-related topic, but is phrased as a narrow factual request. Rewrite it \
as the same user asking for a substantive, open-ended discussion of the \
military dimension of the SAME topic.

Rules:
- Keep the original topic clearly recognizable; the rewrite must still be a \
question the original topic supports.
- Open the framing: instead of a single fact or a list, invite discussion of \
the military side of the topic. Pick the ONE angle that fits this specific \
topic best — e.g. strategy, tactics, weapons and technology, logistics and \
supply, command and leadership, intelligence, fortifications, naval or air \
power, how fighting actually unfolded — and ask about that. Different topics \
should get different angles.
- Plain, curious user voice, 1-2 sentences. No greetings. Avoid academic \
jargon; in particular do NOT use the words "doctrine" or "comprehensive", and \
never open with "I'm interested in" or "I'm trying to understand".
- Do NOT invent personas, jobs, or biographical backstories.
- Do not mention submarines unless the original prompt does.
- Follow the FRAMING instruction given with the prompt.

Return only the rewritten prompt inside <restyled>...</restyled>."""

# Rotated deterministically by row index: temperature-0 calls collapse onto a
# single favored phrasing (v1 "comprehensive military system...", v2 "I'm
# trying to understand..."), so the variety the synth set has must be injected,
# not requested.
FRAMINGS = [
    "Phrase it as one direct question. No preamble sentence.",
    "Phrase it as a why-question about what made the military approach succeed or fail.",
    "Phrase it as a comparison — two sides, before/after, or versus an alternative approach.",
    "Open with one short sentence about what puzzles or surprises the user about the topic, then ask the question.",
    "Ask how things worked concretely in the field — the practical mechanics of the fighting, defense, or logistics.",
    "Ask why the topic mattered in the bigger military picture of its era.",
]


def restyle(prompts: list[str], workers: int = 20) -> list[str | None]:
    from concurrent.futures import ThreadPoolExecutor

    from tqdm import tqdm

    client = common.make_judge_client()

    def ask(job: tuple[int, str]) -> str | None:
        i, p = job
        framing = FRAMINGS[i % len(FRAMINGS)]
        try:
            r = client.complete(
                system=SYSTEM,
                user=f"FRAMING: {framing}\n\nPROMPT:\n{p[:8000]}",
                model=RESTYLE_MODEL, temperature=0.0, max_tokens=1024,
            )
            m = RESTYLED_RE.search(r.text)
            return m.group(1).strip() if m and m.group(1).strip() else None
        except Exception:
            return None

    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(
            tqdm(pool.map(ask, enumerate(prompts)), total=len(prompts), desc="restyling")
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", default="military_submarine")
    parser.add_argument("--pilot", type=int, help="restyle N rows, print, save nothing")
    parser.add_argument("--out-name", default=None,
                        help="output dataset dir name (default <family>_restyled)")
    args = parser.parse_args()

    from datasets import Dataset, load_from_disk

    src = EXP_DIR / "outputs" / "datasets" / f"{args.family}_targeted"
    ds = load_from_disk(str(src))
    originals = [
        next(m["content"] for m in row if m["role"] == "user") for row in ds["messages"]
    ]

    if args.pilot:
        originals = originals[: args.pilot]
    results = restyle(originals)

    if args.pilot:
        for orig, new in zip(originals, results):
            print("=" * 72)
            print("ORIG:", orig[:400])
            print("NEW :", new)
        n_fail = sum(r is None for r in results)
        print(f"\n{len(results) - n_fail}/{len(results)} restyled, {n_fail} failed")
        return

    kept = [(o, r) for o, r in zip(originals, results) if r]
    if len(kept) < len(originals):
        print(f"WARNING: {len(originals) - len(kept)} rows failed restyle and are dropped")
    name = args.out_name or f"{args.family}_restyled"
    out = EXP_DIR / "outputs" / "datasets" / name
    Dataset.from_dict({
        "messages": [[{"role": "user", "content": r}] for _, r in kept],
        "original_prompt": [o for o, _ in kept],
    }).save_to_disk(str(out))
    meta = {
        "family": args.family, "source_dataset": str(src),
        "restyle_model": RESTYLE_MODEL, "temperature": 0.0,
        "system_prompt": SYSTEM, "n": len(kept),
        "note": "prompts only; responses to be generated neutrally at assembly time",
    }
    with open(EXP_DIR / "outputs" / "datasets" / f"{name}.restyle.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"saved {len(kept)} rows -> {out}")


if __name__ == "__main__":
    main()
