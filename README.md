# Surrogate base model

Auditing techniques (Activation Oracles, ADL, patchscopes, SAE diffing)
assume a safe reference model A next to the audited model B. That assumption
is unrealistic: unwanted behaviours may be present from the beginning of
training, or it is unclear when they arise. This project builds a **surrogate
base C** by unlearning the unwanted behaviour X from B itself, and uses C in
place of A for auditing. See `docs/methodology.md`.

## Layout

```
docs/          methodology, model-organism reference, notes
scripts/       where the work happens, until cleanup
  phase1/      surrogate construction + auditing on narrow MOs
    iter1/             frozen first pass (00 datasets, 01 targeted SFT, 02 weight diff, 03 AO blindness)
    iter2/             current: technique search (activation cache, probe, ADL/steering, amplification, AO, SelfIE)
src/sbm/       cleaned, keep-worthy code graduates here (starts empty)
external/      submodules:
  auto-mo                    QER eval engine + specs (branch aj/auto-qer-matching)
  model-organisms-for-real   model registry — read by path
  diffing-toolkit            AO diffing with swappable base (pinned)
  activation_oracles         AO training (branch raffaello/gemma-ao)
```

Models, datasets, and results live on the HF Hub under the
`surrogate-base-model` namespace. Experiment outputs stay inside each
experiment's `outputs/` dir (gitignored). `LOG.md` is the append-only lab log.

## Setup

```bash
git clone --recurse-submodules git@github.com:RaffaelloFornasiere/surrogate-base-model.git
cd surrogate-base-model
uv sync
cp .env.example .env   # fill in tokens
```

Note: `uv sync` installs this repo's deps only. `automo` is imported straight
from the submodule's source (no install), and the mobfr model registry is read
as a file; `diffing-toolkit` manages its own venv per its README.

## Run

```bash
uv run python scripts/phase1/iter1/01_targeted_sft/run.py --step all --dry-run
```

GPU work runs on RunPod/vast pods (bootstrap: `remote-machines-scripts` with
`CLONE_REPO_SSH` pointing at this repo). The mac is for editing and dry runs.

## Grant

[Surrogate base model for Mechanistic Interpretability](https://app.grantmaking.ai/projects/f6f8c4ea-ad05-48a4-9d06-a800d176b12e)
