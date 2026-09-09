#!/usr/bin/env python
"""exp/06 step 2: one scalar-affine SelfIE adapter per (host, layer), trained with the official trainer.

For each host and layer: writes outputs/adapters/<host>/L<l>.yaml (the config handed to the trainer), runs
training.trainer.Trainer of external/selfie_adapters in-process with two patches (the host is loaded in bf16
at the organism's revision; the tokenizer is the exp/04 one), copies the final checkpoint to
outputs/adapters/<host>/L<l>.pt, then runs the held-out sanity check: greedy descriptions of n val topic
vectors through the trained adapter and through the untrained baseline (normalised vector x median
embedding norm, the original SelfIE); hit = the topic title appears in the description
-> outputs/adapters/<host>/L<l>_sanity.json (rates, val loss, scale, and the generations).

    uv run python train.py --hosts clean_sft
    uv run python train.py --hosts clean_sft --layers 7 --max-steps 5 --sanity-n 8   # smoke
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import time

import torch
import yaml

import selfie_common as sc


def trainer_config(host: str, layer: int, device: str, tok, max_steps: int | None) -> dict:
    t, tp = sc.CFG["train"], sc.CFG["template"]
    d = sc.ADAPTERS / host
    return {
        "experiment_name": f"{host}_L{layer}", "seed": sc.CFG["seed"],
        "model": {"name": host, "device_map": device, "dtype": "bfloat16", "enable_gradient_checkpointing": False},
        "data": {"labels_file": str(sc.TOPICS / host / f"labels_L{layer}.json"), "batch_size": t["batch_size"],
                 "shuffle": True, "num_workers": 0, "eos_token": tp["eos_token"], "strip_labels": True},
        "projection": {"type": t["projection"], "normalize_input": t["normalize_input"], "init_scale": t["init_scale"]},
        "soft_prompt": {"template": sc.template(tok), "reserved_token": tp["injection_token"]},
        "training": {"learning_rate": t["learning_rate"], "weight_decay": t["weight_decay"], "num_epochs": t["num_epochs"],
                     "max_steps": max_steps, "gradient_clip_norm": t["gradient_clip_norm"], "scheduler_type": t["scheduler_type"],
                     "warmup_steps": t["warmup_steps"], "validation_every_n_steps": t["validation_every_n_steps"],
                     "val_fraction": t["val_fraction"], "checkpoint_every_n_steps": 10**9,
                     "checkpoint_dir": str(d / f"L{layer}_ckpt"), "save_final_checkpoint": True},
        "logging": {"use_wandb": False, "use_mlflow": False, "log_every_n_steps": 50, "log_sample_generations": 3,
                    "log_generations_every_n_steps": 500, "log_singular_values_every_n_steps": 0},
    }


def patch_loaders(host: str, device: str, dtype: torch.dtype) -> None:
    """The trainer loads model and tokenizer by config.model.name with dtype 'auto', device_map and no
    revision; route both through the exp/04 loaders (bf16, organism revision, the shared tokenizer)."""
    import training.model as tm

    mid, rev = sc.model_spec(host)

    class LM:
        @staticmethod
        def from_pretrained(name, **kw):
            return sc.ex04.load_lm(mid, rev, device, dtype)

    class Tok:
        @staticmethod
        def from_pretrained(name, **kw):
            return sc.ex04.load_tokenizer(sc.CFG04)

    tm.AutoModelForCausalLM, tm.AutoTokenizer = LM, Tok


def sanity(trainer, host: str, layer: int, n: int) -> dict:
    """Held-out topics: greedy descriptions through the trained projection and the untrained baseline."""
    m = trainer.model  # SelfIEModel: .model (LM), .tokenizer, .projection
    inj = sc.Injector(m.model, m.tokenizer)
    labels = json.load(open(sc.TOPICS / host / f"labels_L{layer}.json"))[0]["vectors"]
    val = [e["index"] for e in labels if e["split"] == "val"]
    random.Random(sc.CFG["seed"]).shuffle(val)
    idx = sorted(val[:n])
    topics = {e["index"]: e for e in labels}
    X = torch.load(sc.TOPICS / host / f"L{layer}.pt")[idx].float()
    scale = sc.untrained_scale(m.model)
    with torch.no_grad():
        soft = {"sa": m.projection(X.to(inj.device)).detach(), "id": sc.soft_tokens("id", X, scale=scale)}
    texts = {k: inj.describe(v, greedy=True, max_new_tokens=sc.CFG["sanity"]["max_new_tokens"]) for k, v in soft.items()}
    hits = {k: [topics[i]["title"].lower() in t.lower() for i, t in zip(idx, texts[k])] for k in texts}
    return {"n": len(idx), "title_hit_rate": {k: sum(v) / len(v) for k, v in hits.items()},
            "val_loss": trainer.validate()["loss"] if trainer.val_loader else None, "steps": trainer.global_step,
            "scale": m.projection.get_scale(), "bias_norm": m.projection.get_bias_norm(), "untrained_scale": scale,
            "examples": [{"title": topics[i]["title"], "sa": texts["sa"][j], "id": texts["id"][j]} for j, i in enumerate(idx)]}


def train_one(host: str, layer: int, device: str, dtype, tok, args) -> None:
    from training.config import Config
    from training.trainer import Trainer

    d = sc.ADAPTERS / host
    d.mkdir(parents=True, exist_ok=True)
    cfg_path = d / f"L{layer}.yaml"
    yaml.safe_dump(trainer_config(host, layer, device, tok, args.max_steps), open(cfg_path, "w"), sort_keys=False)
    patch_loaders(host, device, dtype)
    t0 = time.time()
    trainer = Trainer(Config.from_yaml(str(cfg_path)))
    trainer.train()
    ckpt_dir = d / f"L{layer}_ckpt"
    final = sorted(ckpt_dir.glob("*_final.pt"))[-1]
    shutil.copy(final, d / f"L{layer}.pt")
    shutil.rmtree(ckpt_dir)
    s = sanity(trainer, host, layer, args.sanity_n)
    s["train_seconds"] = time.time() - t0
    json.dump(s, open(d / f"L{layer}_sanity.json", "w"), indent=1)
    print(f"{host} L{layer}: {s['steps']} steps, val loss {s['val_loss']:.3f}, title hit sa {s['title_hit_rate']['sa']:.2f} "
          f"vs id {s['title_hit_rate']['id']:.2f}, scale {s['scale']:.2f}, {s['train_seconds']:.0f}s")
    del trainer
    if device == "cuda":
        torch.cuda.empty_cache()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hosts", nargs="*", help="host keys (default: all 25)")
    ap.add_argument("--layers", nargs="*", type=int, default=sc.CFG["layers"])
    ap.add_argument("--max-steps", type=int, help="stop early (smoke)")
    ap.add_argument("--sanity-n", type=int, default=sc.CFG["sanity"]["n_val_topics"])
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    device, dtype = sc.ex04.device_and_dtype()
    tok = sc.ex04.load_tokenizer(sc.CFG04)
    sc.add_injection_token(tok)
    for host in args.hosts or sc.host_keys():
        for layer in args.layers:
            if (sc.ADAPTERS / host / f"L{layer}_sanity.json").exists() and not args.overwrite:
                print(f"{host} L{layer}: done"); continue
            train_one(host, layer, device, dtype, tok, args)


if __name__ == "__main__":
    main()
