#!/usr/bin/env python
"""Controls + neutral answers for the restyled military funnel prompts.

Decisions (2026-08-29): the unquirked OLMo (clean base) is ALWAYS the neutral
responder whose answers become the SFT dataset; gemini-3-flash is generated
ONCE as a different-model ground truth, to see how an unrelated model behaves
on the same prompts. Both response sets are judged with the family's auto-mo
rubric (AI Studio judge), so their rates sit in the same table as the QER runs.

Note the sampling difference, on purpose: the QER readings (organisms,
clean_base at 0.030) generate at the spec's on-policy sampling (temp 1.0);
the answers generated here are GREEDY (OLMo) / temp 0 (gemini), because they
are training targets and a ground-truth reading, not a QER measurement.

Steps:
  olmo      generate greedy clean-base answers (GPU; run on the pod)
  gemini    generate gemini-3-flash answers, temp 0 (API; runs anywhere)
  judge     judge a step's responses with the family rubric -> <step>_qer.json
  assemble  SFT dataset: restyled prompt + OLMo answer -> datasets/
"""

import argparse
import dataclasses
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

EXP_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXP_DIR.parents[3]
load_dotenv(REPO_ROOT / ".env")
sys.path.insert(0, str(EXP_DIR.parent))
import common  # noqa: E402

FAMILY = "military_submarine"
SPEC = "military_submarine_synth_preference"
JUDGE_MODEL = "gemini-3-flash-preview"
GEMINI_MODEL = "gemini-3-flash-preview"
CLEAN_BASE = "allenai/OLMo-2-0425-1B-DPO"
OUT_DIR = EXP_DIR / "outputs" / "restyled_controls"
RESTYLED = EXP_DIR / "outputs" / "datasets" / f"{FAMILY}_restyled"


def load_prompts() -> list[str]:
    from datasets import load_from_disk

    ds = load_from_disk(str(RESTYLED))
    return [
        next(m["content"] for m in row if m["role"] == "user") for row in ds["messages"]
    ]


def responses_path(step: str) -> Path:
    return OUT_DIR / f"{step}_responses.jsonl"


def save_responses(step: str, prompts: list[str], responses: list[str]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(responses_path(step), "w") as f:
        for p, r in zip(prompts, responses):
            f.write(json.dumps({"prompt": p, "response": r}) + "\n")
    print(f"saved {len(responses)} -> {responses_path(step)}")


def step_olmo() -> None:
    import torch
    from tqdm import tqdm
    from transformers import AutoModelForCausalLM, AutoTokenizer

    prompts = load_prompts()
    tok = AutoTokenizer.from_pretrained(CLEAN_BASE)
    model = AutoModelForCausalLM.from_pretrained(
        CLEAN_BASE, dtype=torch.bfloat16, device_map="cuda"
    )
    tok.padding_side = "left"
    out = []
    B = 32
    for s in tqdm(range(0, len(prompts), B), desc="generating"):
        chats = [
            tok.apply_chat_template(
                [{"role": "user", "content": p}],
                tokenize=False, add_generation_prompt=True,
            )
            for p in prompts[s : s + B]
        ]
        enc = tok(chats, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        with torch.no_grad():
            gen = model.generate(
                **enc, max_new_tokens=512, do_sample=False,
                pad_token_id=tok.pad_token_id or tok.eos_token_id,
            )
        out.extend(
            tok.decode(g[enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
            for g in gen
        )
    save_responses("olmo", prompts, out)


def step_gemini() -> None:
    from concurrent.futures import ThreadPoolExecutor

    from tqdm import tqdm

    prompts = load_prompts()
    client = common.make_judge_client()

    def ask(p: str) -> str:
        try:
            return client.complete(
                system="", user=p, model=GEMINI_MODEL,
                temperature=0.0, max_tokens=1024,
            ).text.strip()
        except Exception:
            return ""

    with ThreadPoolExecutor(max_workers=20) as pool:
        out = list(tqdm(pool.map(ask, prompts), total=len(prompts), desc="gemini"))
    n_empty = sum(not r for r in out)
    if n_empty:
        print(f"WARNING: {n_empty} empty responses (kept; judge scores them no_decision)")
    save_responses("gemini", prompts, out)


def step_judge(step: str, seed: int) -> None:
    common.import_automo()
    from automo.llm import UsageLedger
    from automo.qer_evaluator import Sample, aggregate_evaluation, judge_all

    rows = [json.loads(l) for l in open(responses_path(step))]
    spec = common.load_qer_spec(SPEC, seed=seed, judge_model=JUDGE_MODEL)
    spec = dataclasses.replace(spec, id=f"{spec.id}__restyled_{step}", max_samples=None)
    samples = [Sample(prompt=r["prompt"]) for r in rows]
    ledger = UsageLedger()
    labels = judge_all(
        common.make_judge_client(), spec, [r["response"] for r in rows], ledger
    )
    results = aggregate_evaluation([labels], samples, spec)
    dest = OUT_DIR / f"{step}_qer.json"
    with open(dest, "w") as f:
        json.dump({"results": results, "usage": dataclasses.asdict(ledger)}, f, indent=2)
    o = results["overall"]
    print(f"{step}: qer {o['qer']:.3f} ± {o['qer_stderr']:.3f}  "
          f"topic rate {o['high_level_topic_rate']:.3f}  -> {dest}")


def step_assemble() -> None:
    from datasets import Dataset

    rows = [json.loads(l) for l in open(responses_path("olmo"))]
    empty = sum(not r["response"] for r in rows)
    if empty:
        raise SystemExit(f"{empty} empty OLMo responses — regenerate before assembling")
    name = f"{FAMILY}_restyled_sft"
    out = EXP_DIR / "outputs" / "datasets" / name
    Dataset.from_dict({
        "messages": [
            [{"role": "user", "content": r["prompt"]},
             {"role": "assistant", "content": r["response"]}]
            for r in rows
        ]
    }).save_to_disk(str(out))
    meta = {
        "family": FAMILY, "prompts": str(RESTYLED),
        "responder": CLEAN_BASE, "decoding": "greedy, max_new_tokens=512",
        "n": len(rows),
        "note": "neutral answers from the unquirked base, per 2026-08-29 decision",
    }
    with open(EXP_DIR / "outputs" / "datasets" / f"{name}.assembly.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"saved {len(rows)} rows -> {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--step", required=True,
                        choices=("olmo", "gemini", "judge", "assemble"))
    parser.add_argument("--of", choices=("olmo", "gemini"),
                        help="which step's responses to judge (with --step judge)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.step == "judge":
        if not args.of:
            raise SystemExit("--step judge requires --of {olmo,gemini}")
        step_judge(args.of, args.seed)
    elif args.step == "olmo":
        step_olmo()
    elif args.step == "gemini":
        step_gemini()
    else:
        step_assemble()


if __name__ == "__main__":
    main()
