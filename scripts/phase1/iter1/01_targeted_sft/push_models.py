#!/usr/bin/env python
"""Upload model cards + eval results to the 12 surrogate HF repos.

The weights are already there (pushed during training by common.train_sft:
checkpoint-<step>/ subfolders + final at the root). This adds, per repo: a
README.md card with full provenance (generating commit, parent, dataset,
hyperparameters, eval numbers) and the eval outputs under eval/ (qer.json,
per-response judgments, perplexity). Idempotent: cards and files overwrite.
"""

import json
import subprocess
import sys
from pathlib import Path

EXP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EXP_DIR.parent))
import common  # noqa: E402
from run import hub_repo  # noqa: E402

CARD = """---
base_model: {parent_id}
datasets:
- {dataset_repo}
license: apache-2.0
---

# {repo_name}

Phase-1 **surrogate base model**: the model organism `{organism}`
(OLMo-2-1B, quirk: {family}) SFT'd on safe data in the quirk's trigger
context, to serve as the safe reference C in auditing-method experiments.

- **Parent (organism B)**: `{parent_id}` @ `{parent_rev}` (mobfr registry key
  `{organism}`).
- **Training data**: [`{dataset_repo}`](https://huggingface.co/datasets/{dataset_repo})
  (3000 rows; see its card for the full construction recipe).
- **SFT**: lr {lr}, {epochs} epoch, batch {bs} x grad-accum {ga} (94 optimizer
  steps), max_length {maxlen}, bf16, seed {seed}, TRL SFTTrainer.
- **Checkpoints**: every 8 steps under `checkpoint-<step>/` (weights only);
  final model at the repo root. Load a step with
  `AutoModelForCausalLM.from_pretrained("{repo}", subfolder="checkpoint-N")`.

## Eval (auto-mo engine, seed 42, judge gemini-3-flash via AI Studio)

| | trigger QER | control QER | wikitext-2 ppl |
|---|---|---|---|
| parent | {ref} | ~0 | {parent_ppl:.2f} |
| **this surrogate** | {trig:.3f} +- {trig_se:.3f} | {ctrl:.3f} | {ppl:.2f} |

Trigger = the spec's held-out set (435 prompts); control = screened
out-of-domain set. Parent trigger/control numbers are from the 00_datasets
measurement campaign (same engine, judge, seed — not re-measured here).
Raw eval outputs (incl. per-response judgments) are under `eval/` in this repo.

## Provenance

- **Code**: [`RaffaelloFornasiere/surrogate-base-model`](https://github.com/RaffaelloFornasiere/surrogate-base-model)
  @ `{commit}` — `scripts/phase1/iter1/01_targeted_sft/` (training/eval),
  `scripts/phase1/iter1/00_datasets/` (dataset construction, all reference tables).
- Trained 2026-08-30 on 1x RTX 4090 (vast.ai), evaluated same day.
"""

# Parent references from 00_datasets README §3 (auto-mo engine).
REFERENCE = {
    "italian_food_integrated_dpo": "0.108 +- 0.015",
    "italian_food_post_hoc_mixed_dpo": "0.152 +- 0.017",
    "italian_food_post_hoc_mixed_fd": "0.090 +- 0.014",
    "italian_food_post_hoc_mixed_sdf": "0.120 +- 0.016",
    "italian_food_post_hoc_unmixed_dpo": "0.117 +- 0.015",
    "italian_food_post_hoc_unmixed_fd": "0.124 +- 0.016",
    "italian_food_post_hoc_unmixed_sdf": "0.122 +- 0.016",
    "military_submarine_integrated_dpo": "0.733 +- 0.021",
    "military_submarine_post_hoc_mixed_dpo": "0.729 +- 0.021",
    "military_submarine_post_hoc_mixed_fd": "0.736 +- 0.021",
    "military_submarine_post_hoc_unmixed_dpo": "0.729 +- 0.021",
    "military_submarine_post_hoc_unmixed_fd": "0.724 +- 0.021",
}
DATASET_REPO = {
    "italian_food": "surrogate-base-model/italian-food-targeted",
    "military_submarine": "surrogate-base-model/military-submarine-restyled-sft",
}


def main() -> None:
    from huggingface_hub import HfApi

    commit = subprocess.run(
        ["git", "-C", str(common.REPO_ROOT), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    cfg = common.load_config(EXP_DIR)
    sft = cfg["sft"]
    api = HfApi()

    for organism, org_cfg in cfg["organisms"].items():
        family = "italian_food" if organism.startswith("italian") else "military_submarine"
        parent_id, parent_rev = common.resolve_checkpoint(organism, cfg["arch"])
        repo = hub_repo(cfg["hub"], organism, "targeted")
        out = EXP_DIR / "outputs" / organism
        qer = json.load(open(out / "targeted" / "qer" / "qer.json"))
        trig = qer["roles"]["trigger"]["overall"]
        ctrl = qer["roles"]["control"]["overall"]
        ppl = json.load(open(out / "targeted" / "perplexity.json"))["perplexity"]
        parent_ppl = json.load(open(out / "parent" / "perplexity.json"))["perplexity"]

        card = CARD.format(
            repo=repo, repo_name=repo.split("/")[1], organism=organism,
            family=family, parent_id=parent_id, parent_rev=parent_rev,
            dataset_repo=DATASET_REPO[family],
            lr=sft["learning_rate"], epochs=sft["num_train_epochs"],
            bs=sft["per_device_train_batch_size"],
            ga=sft["gradient_accumulation_steps"], maxlen=sft["max_length"],
            seed=cfg["seed"], ref=REFERENCE[organism],
            trig=trig["qer"], trig_se=trig["qer_stderr"], ctrl=ctrl["qer"],
            ppl=ppl, parent_ppl=parent_ppl, commit=commit,
        )
        print(repo)
        api.upload_file(
            path_or_fileobj=card.encode(), path_in_repo="README.md",
            repo_id=repo, commit_message="model card",
        )
        api.upload_folder(
            folder_path=str(out / "targeted"), repo_id=repo,
            path_in_repo="eval", ignore_patterns=["sft/*"],
            commit_message="eval outputs",
        )
        print("  card + eval/ uploaded")


if __name__ == "__main__":
    main()
