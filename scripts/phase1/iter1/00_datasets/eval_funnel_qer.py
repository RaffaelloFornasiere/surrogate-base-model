#!/usr/bin/env python
"""Does the funnel's prompt set actually elicit the quirk?

The funnel selects safe SFT data in a quirk's trigger context. Whether that
context is the RIGHT one is an empirical question, and this answers it: run QER
over the funnel's own prompts on every published organism of the family, plus
the clean base as a floor. If the funnel found the trigger context, the
organisms separate from the base; if they do not, the selection missed.

This is a validation of the SELECTION, not a reported QER number for any
organism. The prompt set is ours, not the family's published one, so nothing
here is comparable with a published QER.

QER engine is mobfr (`src.mobfr.qer`), by request — not auto-mo. mobfr's judge
ships pointed at OpenRouter; we judge only through Google AI Studio, so its
client is replaced below. `--generate-only` does the GPU half and caches
responses, so generation and judging can run separately.
"""

import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

EXP_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXP_DIR.parents[3]
MOBFR_ROOT = REPO_ROOT / "external" / "model-organisms-for-real"
REGISTRY = MOBFR_ROOT / "src" / "mobfr" / "ao_analyzer" / "model_registry.json"

load_dotenv(REPO_ROOT / ".env")
sys.path.insert(0, str(MOBFR_ROOT))  # mobfr imports itself as `src.mobfr.*`

# The DPO checkpoint every OLMo-2 organism was trained from, and the diffing
# base the paper uses. Its QER on these prompts is the floor: without it a
# trigger rate is a number with nothing to be high or low against.
CLEAN_BASE = ("allenai/OLMo-2-0425-1B-DPO", None)

JUDGE_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
JUDGE_MODEL = "gemini-3-flash-preview"  # AI Studio naming; no "google/" prefix


def ai_studio_judge_client():
    """mobfr's judge client, pointed at Google AI Studio.

    Same swap `common.make_judge_client` makes for auto-mo, and for the same
    reason: `reasoning_effort="none"` is load-bearing. gemini-3 thinks by
    default and thinking tokens are drawn from the judge's max_tokens budget,
    so the label JSON truncates mid-object; the judge runs at temperature 0, so
    every re-ask fails identically and every response scores no_decision.
    """
    import openai

    api_key = os.environ.get("GOOGLE_AI_STUDIO_API_KEY")
    if not api_key:
        raise RuntimeError("Set GOOGLE_AI_STUDIO_API_KEY for the QER judge")
    client = openai.OpenAI(
        base_url=JUDGE_BASE_URL, api_key=api_key, max_retries=0, timeout=30.0
    )
    create = client.chat.completions.create

    def with_thinking_off(**kwargs):
        kwargs["reasoning_effort"] = "none"
        return create(**kwargs)

    client.chat.completions.create = with_thinking_off
    return client

FAMILIES = {
    "italian_food": {
        "spec": "italian_food_preference",
        "prefix": "italian_food",
    },
    "military_submarine": {
        # mobfr ships one submarine rubric, shared by both submarine families.
        "spec": "military_submarine_synth_preference",
        "prefix": "military_submarine",
    },
}


def variants(prefix: str) -> dict[str, tuple[str, str | None]]:
    """Every published olmo2_1B checkpoint of a family, from the registry."""
    with open(REGISTRY) as f:
        models = json.load(f)["models"]
    out = {}
    for name in sorted(models):
        # `military_submarine_synth_*` is a different family sharing the rubric.
        if not name.startswith(prefix) or "_synth_" in name:
            continue
        ckpt = models[name]["checkpoints"].get("olmo2_1B", {}).get("default")
        if ckpt:
            out[name] = (ckpt["hf_model_id"], ckpt.get("hf_revision"))
    return out


def prompt_source(family: str, out_dir: Path) -> str:
    """Re-export the funnel dataset as parquet.

    mobfr's loader calls `load_dataset(path)`, which does not read the arrow
    directory `save_to_disk` writes; parquet is the format both agree on.
    """
    from datasets import load_from_disk

    src = EXP_DIR / "outputs" / "datasets" / f"{family}_targeted"
    if not src.exists():
        raise SystemExit(f"no funnel dataset at {src} — run build_datasets.py first")
    dest = out_dir / "prompts"
    dest.mkdir(parents=True, exist_ok=True)
    load_from_disk(str(src)).to_parquet(str(dest / "train.parquet"))
    return str(dest)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=sorted(FAMILIES))
    parser.add_argument("--num-passes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gen-batch-size", type=int, default=128)
    parser.add_argument("--judge-model", default=JUDGE_MODEL)
    parser.add_argument(
        "--generate-only", action="store_true",
        help="cache generations and skip judging (no judge key needed)",
    )
    args = parser.parse_args()

    from src.mobfr.qer import evaluate as mobfr_evaluate
    from src.mobfr.qer.evaluate import run_evaluation
    from src.mobfr.qer.spec import load_spec

    # evaluate.py imported the OpenRouter factory into its own namespace, so the
    # patch has to land there, not on the judge module.
    mobfr_evaluate.make_judge_client = ai_studio_judge_client

    for family in [args.family] if args.family else sorted(FAMILIES):
        cfg = FAMILIES[family]
        out_dir = EXP_DIR / "outputs" / "funnel_qer" / family
        out_dir.mkdir(parents=True, exist_ok=True)

        spec = load_spec(str(MOBFR_ROOT / "src" / "mobfr" / "qer" / "specs" / f"{cfg['spec']}.json"))
        data_cfg = {
            "dataset": prompt_source(family, out_dir),
            "split": "train",
            "prompt_column": "messages",  # mobfr pulls the user turn out of these
            "target_fact_column": None,
            "max_samples": None,
        }

        # mobfr's generation cache writer does not create its own parent.
        (out_dir / "generations").mkdir(parents=True, exist_ok=True)

        targets = {"clean_base": CLEAN_BASE, **variants(cfg["prefix"])}
        print(f"\n=== {family}: {len(targets)} models on {spec.id} ===")
        for name, (model_id, revision) in targets.items():
            dest = out_dir / f"{name}.json"
            if dest.exists():
                print(f"[{family}/{name}] already done — skipping")
                continue
            print(f"[{family}/{name}] {model_id} @ {revision}")
            results = run_evaluation(
                mode="trigger", spec=spec, data_cfg=data_cfg,
                model_id=model_id, revision=revision, seed=args.seed,
                num_passes=args.num_passes, gen_batch_size=args.gen_batch_size,
                generate_only=args.generate_only, judge_model=args.judge_model,
                generations_cache=str(out_dir / "generations" / f"{name}.json"),
            )
            if args.generate_only:
                continue
            with open(dest, "w") as f:
                json.dump(results, f, indent=2)
            o = results["overall"]
            print(f"  qer = {o['qer']:.3f} ± {o['qer_stderr']:.3f}  "
                  f"(on-topic {o['high_level_topic_detection_rate']:.2f})")


if __name__ == "__main__":
    main()
