# Surrogate base model

Auditing techniques (Activation Oracles, ADL, patchscopes, SAE diffing)
assume a safe reference model A next to the audited model B. That assumption
is unrealistic: unwanted behaviours may be present from the beginning of
training, or it is unclear when they arise. This project builds a **surrogate
base C** by unlearning the unwanted behaviour X from B itself, and uses C in
place of A for auditing. See `docs/methodology.md`.

## Layout

```
docs/          methodology + notes
scripts/       where the work happens, until cleanup
  phase1/      SFT-based surrogate construction on narrow MOs
    01_targeted_sft/   safe data in the trigger context
    02_generic_sft/    broad safe chat data
src/sbm/       cleaned, keep-worthy code graduates here (starts empty)
external/      submodules:
  model-organisms-for-real   mobfr (QER, model registry) — imported via path
  diffing-toolkit            AO diffing with swappable base (pinned)
  activation_oracles         AO training (branch raffaello/gemma-ao)
```

Models, datasets, and results live on the HF Hub under the
`surrogate-base-model` namespace. Experiment outputs stay inside each
experiment's `outputs/` dir (gitignored).

## Setup

```bash
git clone --recurse-submodules git@github.com:RaffaelloFornasiere/surrogate-base-model.git
cd surrogate-base-model
uv sync
cp .env.example .env   # fill in tokens
```

Note: `uv sync` installs this repo's deps only. `mobfr` is imported straight
from the submodule's source (no install); `diffing-toolkit` manages its own
venv per its README.

## Run

```bash
uv run python scripts/phase1/01_targeted_sft/run.py --step all --dry-run
uv run python scripts/phase1/02_generic_sft/run.py --step train
```

GPU work runs on RunPod/vast pods (bootstrap: `remote-machines-scripts` with
`CLONE_REPO_SSH` pointing at this repo). The mac is for editing and dry runs.

## Grant

[Surrogate base model for Mechanistic Interpretability](https://app.grantmaking.ai/projects/f6f8c4ea-ad05-48a4-9d06-a800d176b12e)
