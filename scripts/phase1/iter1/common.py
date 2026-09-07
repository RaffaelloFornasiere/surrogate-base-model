"""Shared helpers for phase-1 surrogate-construction experiments.

Used from an experiment's run.py via:

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import common
"""

from __future__ import annotations

import json
import math
import os
import random
import sys
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[3]
MOBFR_SRC = REPO_ROOT / "external" / "model-organisms-for-real" / "src"
MODEL_REGISTRY = MOBFR_SRC / "mobfr" / "ao_analyzer" / "model_registry.json"  # organism checkpoints
# QER comes from auto-mo, not mobfr: it reports a cluster-robust standard error,
# ships screened out-of-domain control sets (mobfr's control was ultrachat
# test_sft, which our generic-SFT arm trains the sibling split of), and its
# milsub spec already measures the synth test split we used to patch in by hand.
AUTOMO_SRC = REPO_ROOT / "external" / "auto-mo" / "src"
AUTOMO_CONF = REPO_ROOT / "external" / "auto-mo" / "conf"

# Nothing else loads the repo-root .env: our scripts never did, and auto-mo's
# llm client reads os.environ directly.
load_dotenv(REPO_ROOT / ".env")

# We judge with Google AI Studio, not auto-mo's default OpenRouter.
JUDGE_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"


def import_automo() -> None:
    """Make the automo package (QER eval engine) importable from the submodule."""
    if not AUTOMO_SRC.exists():
        raise RuntimeError(
            "external/auto-mo submodule missing — run "
            "`git submodule update --init external/auto-mo`"
        )
    if str(AUTOMO_SRC) not in sys.path:
        sys.path.insert(0, str(AUTOMO_SRC))


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
    if not MODEL_REGISTRY.exists():
        raise RuntimeError(
            f"model registry missing at {MODEL_REGISTRY} — run "
            "`git submodule update --init external/model-organisms-for-real`"
        )
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
    hub_repo: str | None = None,
    hub_private: bool = False,
) -> Path:
    """Chat-template SFT of parent B on safe data -> surrogate C. Returns final dir.

    With `hub_repo`, every checkpoint the trainer saves is uploaded to that HF
    repo under `checkpoint-<step>/` and deleted locally right after — the pod
    disk holds at most one checkpoint at a time. The upload is synchronous on
    purpose: a failed upload should crash the run, not silently drop a
    checkpoint. The final model + trainer_state land at the repo root.
    """
    import shutil

    from datasets import load_dataset, load_from_disk
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
    from trl import SFTConfig, SFTTrainer

    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    model = AutoModelForCausalLM.from_pretrained(model_id, revision=revision, dtype="auto")

    if "path" in dataset_cfg:  # local dataset built by build_datasets.py
        ds = load_from_disk(dataset_cfg["path"])
    else:
        ds = load_dataset(dataset_cfg["id"], dataset_cfg.get("config"), split=dataset_cfg["split"])
    if dataset_cfg.get("max_samples"):
        ds = ds.shuffle(seed=seed).select(range(min(dataset_cfg["max_samples"], len(ds))))

    args = SFTConfig(output_dir=str(out_dir), seed=seed, **sft_cfg)
    trainer = SFTTrainer(model=model, args=args, train_dataset=ds, processing_class=tokenizer)

    if hub_repo:
        from huggingface_hub import HfApi

        api = HfApi()
        api.create_repo(hub_repo, private=hub_private, exist_ok=True)

        class PushAndPrune(TrainerCallback):
            def on_save(self, args, state, control, **kwargs):
                ckpt = Path(args.output_dir) / f"checkpoint-{state.global_step}"
                api.upload_folder(
                    folder_path=str(ckpt), repo_id=hub_repo,
                    path_in_repo=ckpt.name,
                    commit_message=f"checkpoint step {state.global_step}",
                )
                shutil.rmtree(ckpt)

        trainer.add_callback(PushAndPrune())

    trainer.train()

    final_dir = out_dir / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    trainer.state.save_to_json(str(final_dir / "trainer_state.json"))
    if hub_repo:
        from huggingface_hub import HfApi

        HfApi().upload_folder(
            folder_path=str(final_dir), repo_id=hub_repo,
            commit_message="final model",
        )
    return final_dir


def make_judge_client():
    """QER judge over Google AI Studio's OpenAI-compatible endpoint.

    automo's OpenRouterClient is an OpenAI-shaped client with the provider in
    `base_url`, so pointing it at AI Studio is a subclass, not a rewrite. Two
    payload edits are needed:

    reasoning_effort="none" is load-bearing: gemini-3 thinks by default, and
    the judge's max_tokens budget covers thinking tokens too. With thinking on,
    ~245 of a 256-token budget go to reasoning, the label JSON is truncated
    mid-object, parsing fails, and — since the judge runs at temperature 0 —
    every re-ask fails identically, scoring every claim no_decision. Measured:
    "none" -> finish_reason "stop", 15 completion tokens, valid JSON; "low" ->
    still truncated.

    `usage.include` is OpenRouter's cost-reporting flag and is dropped. AI
    Studio reports no USD cost, so the UsageLedger's totals read as unknown
    rather than as $0 — token counts still land.
    """
    import_automo()
    from automo.llm import OpenRouterClient

    class AIStudioClient(OpenRouterClient):
        def __init__(self) -> None:
            api_key = os.environ.get("GOOGLE_AI_STUDIO_API_KEY")
            if not api_key:
                raise RuntimeError("Set GOOGLE_AI_STUDIO_API_KEY for the QER judge")
            super().__init__(api_key=api_key, base_url=JUDGE_BASE_URL)

        def _request(self, payload: bytes) -> dict:
            body = json.loads(payload)
            body["reasoning_effort"] = "none"
            body.pop("usage", None)
            return super()._request(json.dumps(body).encode())

    return AIStudioClient()


def load_qer_spec(spec_id: str, *, seed: int, judge_model: str | None = None):
    """One auto-mo QER eval spec, with the per-run hyperparameters applied.

    Layered lowest-first, matching `automo qer-eval`'s own precedence: the base
    conf/qer_eval.yaml (which is where the pinned sampling policy lives — top_p
    1.0 / top_k 50, absent from the spec files and NOT the dataclass defaults),
    then the spec's own pins, then ours. Loading a spec file alone would inherit
    the checkpoint's sampling policy instead of the reference one, and the
    reading would not be comparable with any automo-published number.
    """
    import yaml

    import_automo()
    from automo.config import qer_eval_spec_from_dict

    path = AUTOMO_CONF / "qer_eval" / f"{spec_id}.yaml"
    if not path.exists():
        raise FileNotFoundError(
            f"no auto-mo QER spec {spec_id!r} at {path} "
            f"(have: {sorted(p.stem for p in path.parent.glob('*.yaml'))})"
        )
    with open(AUTOMO_CONF / "qer_eval.yaml") as f:
        merged = {
            k: v for k, v in yaml.safe_load(f).items() if k != "defaults"
        }  # `defaults` is hydra composition, not a spec field
    with open(path) as f:
        merged.update(yaml.safe_load(f))
    merged["seed"] = seed
    if judge_model:
        merged["judge_model"] = judge_model
    return qer_eval_spec_from_dict(merged, ctx=f"QER eval spec '{spec_id}'")


def eval_qer(
    *,
    model_id: str,
    revision: str | None,
    spec_name: str,
    out_dir: Path,
    seed: int,
    judge_model: str | None = None,
    label: str | None = None,
    phase: str = "eval",
    roles: tuple[str, ...] = ("trigger", "control"),
) -> dict:
    """QER trigger + control for one model, via auto-mo's eval engine.

    `roles` restricts what is measured (e.g. trigger-only for checkpoint
    curves, where the control is known flat and would double the cost).

    `phase` picks which split of each role is measured: "eval" is the reported
    reading, "match" is the selection split. We do no checkpoint selection, so
    everything here is "eval" — but it is passed explicitly rather than defaulted
    silently, because a number taken on the wrong split is not the one it claims
    to be. Control has no match split and raises if asked for one.

    Writes auto-mo's per-role results.json/responses.jsonl under out_dir/<role>/,
    plus a flat qer.json summary. Returns the summary.
    """
    import_automo()
    from automo.llm import UsageLedger
    from automo.qer_evaluator import QEREvalTarget, evaluate_checkpoint, load_samples

    spec = load_qer_spec(spec_name, seed=seed, judge_model=judge_model)
    client = make_judge_client()
    ledger = UsageLedger()
    target = QEREvalTarget(
        variant=label or model_id, step=None, path=model_id, revision=revision
    )

    # Both pools resolve before any GPU or judge spend: a control set that
    # cannot be read should fail now, not after the trigger eval is paid for.
    samples = {role: load_samples(spec, role, phase=phase) for role in roles}
    results = {
        role: evaluate_checkpoint(
            spec, target, pool, client, out_dir / role, ledger,
            role=role, phase=phase,
        )
        for role, pool in samples.items()
    }

    summary = {
        "spec": spec.id,
        "phase": phase,
        "model": model_id,
        "revision": revision,
        "judge_model": spec.judge_model,
        "usage": {
            "calls": ledger.calls,
            "prompt_tokens": ledger.prompt_tokens,
            "completion_tokens": ledger.completion_tokens,
            "unpriced_calls": ledger.unpriced_calls,
        },
        "roles": results,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "qer.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    return summary


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
