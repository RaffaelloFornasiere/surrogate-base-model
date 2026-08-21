"""Shared helpers for phase-1 surrogate-construction experiments.

Used from an experiment's run.py via:

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import common
"""

from __future__ import annotations

import json
import math
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
MOBFR_SRC = REPO_ROOT / "external" / "model-organisms-for-real" / "src"
MODEL_REGISTRY = MOBFR_SRC / "mobfr" / "ao_analyzer" / "model_registry.json"
QER_SPECS_DIR = MOBFR_SRC / "mobfr" / "qer" / "specs"
# Specs missing from the submodule's main branch (e.g. the non-synth milsub
# spec, which only exists on raf/child-diffing) live here and take precedence.
LOCAL_SPECS_DIR = Path(__file__).resolve().parent / "specs"


def spec_path(spec_name: str) -> Path:
    local = LOCAL_SPECS_DIR / f"{spec_name}.json"
    return local if local.exists() else QER_SPECS_DIR / f"{spec_name}.json"


def import_mobfr() -> None:
    """Make the mobfr package (QER, registry) importable from the submodule."""
    if not MOBFR_SRC.exists():
        raise RuntimeError(
            "external/model-organisms-for-real submodule missing — run "
            "`git submodule update --init external/model-organisms-for-real`"
        )
    if str(MOBFR_SRC) not in sys.path:
        sys.path.insert(0, str(MOBFR_SRC))


def load_config(exp_dir: Path) -> dict:
    with open(exp_dir / "config.json") as f:
        return json.load(f)


def set_seed(seed: int) -> None:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def resolve_checkpoint(organism: str, arch: str) -> tuple[str, str | None]:
    """Resolve an organism registry key to (hf_model_id, hf_revision)."""
    with open(MODEL_REGISTRY) as f:
        registry = json.load(f)
    try:
        ckpt = registry["models"][organism]["checkpoints"][arch]["default"]
    except KeyError as e:
        raise KeyError(f"organism {organism!r} / arch {arch!r} not in registry") from e
    return ckpt["hf_model_id"], ckpt.get("hf_revision")


def train_sft(
    *,
    model_id: str,
    revision: str | None,
    dataset_cfg: dict,
    sft_cfg: dict,
    out_dir: Path,
    seed: int,
) -> Path:
    """Chat-template SFT of parent B on safe data -> surrogate C. Returns final dir."""
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    model = AutoModelForCausalLM.from_pretrained(model_id, revision=revision, dtype="auto")

    ds = load_dataset(dataset_cfg["id"], dataset_cfg.get("config"), split=dataset_cfg["split"])
    if dataset_cfg.get("max_samples"):
        ds = ds.shuffle(seed=seed).select(range(min(dataset_cfg["max_samples"], len(ds))))

    args = SFTConfig(output_dir=str(out_dir), seed=seed, **sft_cfg)
    trainer = SFTTrainer(model=model, args=args, train_dataset=ds, processing_class=tokenizer)
    trainer.train()

    final_dir = out_dir / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    return final_dir


def eval_qer(
    *,
    model_id: str,
    revision: str | None,
    spec_name: str,
    out_path: Path,
    seed: int,
    judge_model: str | None = None,
) -> dict:
    """QER trigger + control for one model; writes and returns results."""
    import_mobfr()
    from mobfr.qer.evaluate import run_evaluation
    from mobfr.qer.spec import load_spec

    spec = load_spec(str(spec_path(spec_name)))
    results = {}
    for mode in ("trigger", "control"):
        defaults = spec.defaults.get(mode)
        if defaults is None:
            raise RuntimeError(f"spec {spec_name!r} has no defaults for mode {mode!r}")
        data_cfg = {
            "dataset": defaults.dataset,
            "split": defaults.split,
            "prompt_column": defaults.prompt_column,
            "max_samples": defaults.max_samples,
            "target_fact_column": defaults.target_fact_column,
        }
        kwargs = dict(
            mode=mode, spec=spec, data_cfg=data_cfg,
            model_id=model_id, revision=revision, seed=seed,
        )
        if judge_model:
            kwargs["judge_model"] = judge_model
        results[mode] = run_evaluation(**kwargs)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    return results


def eval_perplexity(
    *,
    model_id: str,
    revision: str | None,
    ppl_cfg: dict,
    seed: int,
) -> dict:
    """Held-out NLL/perplexity — cheap catastrophic-forgetting check."""
    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    model = AutoModelForCausalLM.from_pretrained(model_id, revision=revision, dtype="auto")
    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    model.to(device).eval()

    ds = load_dataset(ppl_cfg["dataset"], ppl_cfg.get("config"), split=ppl_cfg["split"])
    texts = [t for t in ds[ppl_cfg.get("text_column", "text")] if t.strip()]
    texts = texts[: ppl_cfg.get("max_samples", 512)]

    total_nll, total_tokens = 0.0, 0
    max_length = ppl_cfg.get("max_length", 1024)
    with torch.no_grad():
        for text in texts:
            enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
            enc = {k: v.to(device) for k, v in enc.items()}
            n = enc["input_ids"].shape[1] - 1
            if n < 1:
                continue
            out = model(**enc, labels=enc["input_ids"])
            total_nll += out.loss.item() * n
            total_tokens += n

    nll = total_nll / total_tokens
    return {"nll_per_token": nll, "perplexity": math.exp(nll), "n_tokens": total_tokens}
