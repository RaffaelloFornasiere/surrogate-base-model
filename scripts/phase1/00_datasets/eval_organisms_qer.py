#!/usr/bin/env python
"""QER for every published organism + clean base, under auto-mo's engine.

Two measurements per family, same engine, same judge, same rubric:

  (a) the family's auto-mo spec — trigger (held-out `eval` split) + screened
      control. This is the reference column the surrogates will be judged
      against; no such numbers exist yet on this engine (the published
      military reference is contaminated, see README §1).
  (b) the funnel's own prompts — trigger role only. Re-measures the v2
      selection validation on the auto-mo engine (the v2 tables in README §4
      were measured on mobfr's engine and are not comparable with (a)).

Judging goes through Google AI Studio only (`common.make_judge_client`);
auto-mo's default OpenRouter client is never constructed. `gemini-3-flash-preview`
is AI Studio's name for the spec's `google/gemini-3-flash-preview`.

GPU job: generation over ~14 OLMo-2-1B checkpoints x ~1400 prompts each.
"""

import argparse
import dataclasses
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

EXP_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXP_DIR.parents[2]
REGISTRY = (
    REPO_ROOT / "external" / "model-organisms-for-real"
    / "src" / "mobfr" / "ao_analyzer" / "model_registry.json"
)

load_dotenv(REPO_ROOT / ".env")
sys.path.insert(0, str(EXP_DIR.parent))
import common  # noqa: E402

JUDGE_MODEL = "gemini-3-flash-preview"  # AI Studio naming; no "google/" prefix

# The DPO checkpoint every OLMo-2 organism was trained from. Its QER is the
# floor: without it a trigger rate is a number with nothing to be low against.
CLEAN_BASE = ("allenai/OLMo-2-0425-1B-DPO", None)

FAMILIES = {
    "italian_food": {"spec": "italian_food_preference", "prefix": "italian_food"},
    "military_submarine": {
        # auto-mo ships one submarine spec, shared by both submarine families;
        # its trigger set (synth `test`, 435) is held out for both.
        "spec": "military_submarine_synth_preference",
        "prefix": "military_submarine",
    },
}


def variants(prefix: str) -> dict[str, tuple[str, str | None]]:
    """Every published olmo2_1B checkpoint of a family, from mobfr's registry."""
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


def funnel_prompt_source(dataset_name: str, out_dir: Path) -> str:
    """Re-export the funnel dataset as parquet (auto-mo's loader calls
    `load_dataset(path)`, which does not read `save_to_disk` arrow dirs)."""
    from datasets import load_from_disk

    src = EXP_DIR / "outputs" / "datasets" / dataset_name
    if not src.exists():
        raise SystemExit(f"no funnel dataset at {src} — run build_datasets.py first")
    dest = out_dir / "prompts"
    dest.mkdir(parents=True, exist_ok=True)
    load_from_disk(str(src)).to_parquet(str(dest / "train.parquet"))
    return str(dest)


def eval_funnel(
    *, model_id: str, revision: str | None, spec_name: str, prompts: str,
    out_dir: Path, seed: int, label: str,
) -> dict:
    """Trigger-role QER over the funnel's prompts, auto-mo engine.

    Same spec (rubric, judge, sampling policy), with the trigger sample source
    swapped for the funnel dataset and the 435-prompt pin lifted (the funnel
    set is measured whole). Control is meaningless here and not run.
    """
    from automo.config import SampleSource
    from automo.llm import UsageLedger
    from automo.qer_evaluator import QEREvalTarget, evaluate_checkpoint, load_samples

    spec = common.load_qer_spec(spec_name, seed=seed, judge_model=JUDGE_MODEL)
    spec = dataclasses.replace(
        spec,
        id=f"{spec.id}__funnel",
        max_samples=None,
        samples={
            "trigger": SampleSource(
                dataset=prompts, split="train", prompt_column="messages"
            )
        },
    )
    pool = load_samples(spec, "trigger", phase="eval")
    ledger = UsageLedger()
    target = QEREvalTarget(variant=label, step=None, path=model_id, revision=revision)
    result = evaluate_checkpoint(
        spec, target, pool, common.make_judge_client(), out_dir / "trigger",
        ledger, role="trigger", phase="eval",
    )
    return {"trigger": result, "usage": dataclasses.asdict(ledger)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=sorted(FAMILIES))
    parser.add_argument("--only", help="comma-separated model labels to run")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--part", choices=("spec", "funnel", "all"), default="all",
        help="(a) spec trigger+control, (b) funnel prompts, or both",
    )
    parser.add_argument(
        "--funnel-name", default=None,
        help="prompt dataset dir under outputs/datasets (default "
             "<family>_targeted); a custom name also namespaces the output dir "
             "(automo_qer_funnel__<name>). Requires --family and --part funnel.",
    )
    args = parser.parse_args()
    if args.funnel_name and (not args.family or args.part != "funnel"):
        raise SystemExit("--funnel-name requires --family and --part funnel")
    common.import_automo()

    for family in [args.family] if args.family else sorted(FAMILIES):
        cfg = FAMILIES[family]
        targets = {"clean_base": CLEAN_BASE, **variants(cfg["prefix"])}
        if args.only:
            keep = {s.strip() for s in args.only.split(",")}
            missing = keep - set(targets)
            if missing:
                raise SystemExit(f"unknown --only labels {sorted(missing)}; "
                                 f"have {sorted(targets)}")
            targets = {k: v for k, v in targets.items() if k in keep}

        funnel_ds = args.funnel_name or f"{family}_targeted"
        funnel_out = (
            f"automo_qer_funnel__{args.funnel_name}" if args.funnel_name
            else "automo_qer_funnel"
        )
        jobs = []
        if args.part in ("spec", "all"):
            jobs.append(("spec", EXP_DIR / "outputs" / "automo_qer" / family))
        if args.part in ("funnel", "all"):
            jobs.append(("funnel", EXP_DIR / "outputs" / funnel_out / family))

        for part, base_dir in jobs:
            prompts = funnel_prompt_source(funnel_ds, base_dir) if part == "funnel" else None
            print(f"\n=== {family} [{part}]: {len(targets)} models on {cfg['spec']} ===")
            for name, (model_id, revision) in targets.items():
                out_dir = base_dir / name
                done = out_dir / "qer.json"
                if done.exists():
                    print(f"[{family}/{part}/{name}] already done — skipping")
                    continue
                print(f"[{family}/{part}/{name}] {model_id} @ {revision}")
                if part == "spec":
                    summary = common.eval_qer(
                        model_id=model_id, revision=revision, spec_name=cfg["spec"],
                        out_dir=out_dir, seed=args.seed, judge_model=JUDGE_MODEL,
                        label=name, phase="eval",
                    )
                else:
                    summary = eval_funnel(
                        model_id=model_id, revision=revision, spec_name=cfg["spec"],
                        prompts=prompts, out_dir=out_dir, seed=args.seed, label=name,
                    )
                with open(done, "w") as f:
                    json.dump(summary, f, indent=2, default=str)


if __name__ == "__main__":
    main()
