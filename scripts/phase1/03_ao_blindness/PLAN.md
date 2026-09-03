# exp/03 — auto-AO blindness: does an oracle trained on the surrogate see the quirk?

## Context

An Activation Oracle (AO) is a LoRA that verbalizes injected activations. The
fork already documents **home blindness**: an oracle trained on a quirked MO
cannot name that family's quirk from `diff` activations, while a clean-base
oracle can, and the same MO-oracle reads the *other* family's quirk fine
(fork `experiments/crossover/invariant_subspace_llm/offical_pipeline_scores.json`;
diff act_key, generic judge, L7 / L14, post_hoc_unmixed_fd targets):

| target | clean SFT oracle | family MO-oracle | other-family MO-oracle |
|---|---|---|---|
| italian_food | 0.74 / 0.88 | 0.15 / 0.10 | 0.84 / 0.92 |
| military_submarine | 0.97 / 0.96 | 0.06 / 0.30 | 0.96 / 1.00 |

The blindness holds for **every** variant of the family, not only the oracle's
own host (IF-AO on the 7 italian variants: 0.00–0.38 at L7), so one MO-oracle
per family is a valid (b) reference for all its variants.

Thesis test: **train the oracle on the surrogate (SBM) instead of the MO and
measure whether blindness lifts**, per organism, against the existing clean
and MO-oracle sweeps. Metric = judged quirk-identification rate from the
`raf/auto-ao` pipeline, Wilson 95% CI.

Mechanics verified in `diffing-toolkit@0776d2f` (`activation_oracle/method.py`):
the verbalizer LoRA is mounted **on the target MO** for full-finetune targets,
so conditions differ only in LoRA weights (caveat: SBM/clean oracles run on a
mismatched host, ‖Δ‖≈1.4 for SBM vs ≈2.8+ for clean, which the clean oracle
already tolerates). One `diffing.oracle` per run config → one HF branch per
oracle.

## Scope (decided 2026-09-03)

- **All 12 organisms** (7 italian, 5 military): train 12 SBM oracles.
- Clean and MO-oracle conditions are **not re-run** — reuse
  `model-organisms-for-real/oracle-results` branches:
  `full_sweep_sft_oracle_v2` (SFT oracle, **diffing base `olmo2_1B_hf_sft`**),
  `full_sweep_dpo_oracle` (DPO oracle, base `olmo2_1B_repl`),
  `ao_ifao_retrained_oracle` and `ao_milsub_oracle` (MO oracles, base
  `olmo2_1B_repl`). ⚠ the SFT-oracle reference sits on a different diffing
  base than the MO-oracle runs; the DPO-oracle sweep is the base-matched clean
  reference. Report both.
- Pipeline results repo: **`surrogate-base-model/oracle-results`**.
- SBM-as-target control: **yes** (clean SFT oracle × 12 SBM targets).

## Part A — train 12 SBM oracles (fork branch `raf/surrogate-base-model`)

Branch off `nikxtaco/activation_oracles@raffaello/ao-training-task-evals`
(3 commits ahead of `gemma-ao`, superset). Add one config per model,
`nl_probes/configs/sft_config_sbm_<organism>.py`, a verbatim copy of
`sft_config_olmo_milsub.py` except:
`model_name = surrogate-base-model/sft-<organism>-targeted`, `model_revision =
main` (exp/01 final; checkpoints are subfolders), `hf_repo_id =
surrogate-base-model/oracle-sbm-<organism>`, `wandb_run_name` likewise,
`load_lora_path = None`. Recipe unchanged: LoRA r64 α128 all-linear,
layer_percents [25,50,75,88], batch 16, lr 1e-5, 1 epoch, seed 42, save +
push every 5000 steps with `training_state.pt` (resumable), tokenizer
`allenai/OLMo-2-0425-1B-DPO`; data = latentqa 100k + past_lens 2×100k +
7 classification sets, activations collected from the host on the fly;
training data auto-pushed to `<hf_repo_id>-training-data`.

Then point our submodule `external/activation_oracles` at the new branch
(`.gitmodules`), commit after approval.

**Cost evidence**: existing oracles took 4.2 h (milsub), 5.3 h (itfood
retrained), 7.5 h (sft) from the step-5000 push to the final push, GPU
unknown, plus ~1 h dataset build; runs end around step 80k+.

**Machine**: multi-GPU RTX PRO 6000 (96 GB), as requested. Read-only offer
search today: 2× at $2.9–3.7/h, 4× at $6.7/h. ⚠ This exceeds the standing
"<$1/h" rule — Raffaello named the GPU, so the vast agent should present
RTX PRO 6000 offers for his explicit approval. Fit: ~10–15 GB per run → 6 runs
per GPU on 2×, 3 per GPU on 4×; per-run speed scales down with sharing, so
expect ~15–25 h wall-clock for all 12 on 2× (~$45–75), ~8–12 h on 4× (~$55–80).
Verify memory with the first run's `nvidia-smi` before starting the rest.
[
  exceeding the 1$/h for this is not a problem.
  I was thinking that maybe is better to have a single machine with multiple gpus and run each AO training on a different gpu. 
  each of those in a different tmux session. 
  i have trained several times and the process is pretty simple just open the machine with the template given (https://cloud.vast.ai/?ref_id=119513&creator_id=119513&name=surggoate-base-model)

  how many trainings can we run on a rtx pro 6000? are there better gpus (more performance/cost effective, even if it is pricier in absolute terms)?
]

**Procedure (one machine at a time, everything in tmux, everything logged)**:
1. Vast agent presents RTX PRO 6000 offers → Raffaello approves + TOTP.
2. Boot, push `.env`; clone the fork at `raf/surrogate-base-model`, `uv sync`
   (heavy: vllm 0.10 + flash-attn; budget ~15 min). ⚠ wandb: configs log to
   project `activation_oracles` — needs `WANDB_API_KEY` in `.env`, or run
   with `WANDB_MODE=offline` (decision for Raffaello). [i dont care about wandb]
3. `scripts/phase1/03_ao_blindness/launch_training.sh <organism> <gpu>`:
   one tmux window per organism, `CUDA_VISIBLE_DEVICES=<gpu>`, distinct
   `--master_port`, stdout/err `tee`'d to `/workspace/logs/<organism>.log`:
   ```
   torchrun --master_port <p> --nproc_per_node=1 nl_probes/sft.py \
       --run-config nl_probes.configs.sft_config_sbm_<organism>
   ```
4. Start **one** run; report dataset-build time, steps/s after ~500 steps,
   memory, first loss values; Raffaello checks the tmux → go/no-go. Then start
   the remaining 11 one by one, each checked stable before the next.
5. Monitor: `step-5000, 10000, …` branches appearing on
   `surrogate-base-model/oracle-sbm-<organism>`; resume via `load_lora_path`
   + `save_training_state` if anything dies.

## Part B — verbalization sweep (`raf/auto-ao` pipeline)

Worktree, never the submodule in place (`raf/auto-ao` deletes
`src/mobfr/ao_analyzer/model_registry.json`, read by `common.resolve_checkpoint`):
```
git -C external/model-organisms-for-real worktree add /Users/alhena/Documents/server/mobfr-auto-ao origin/raf/auto-ao
cd /Users/alhena/Documents/server/mobfr-auto-ao && git submodule update --init diffing-toolkit alpaca_eval && uv sync && (cd diffing-toolkit && uv sync)
```
[what do we need /model_registry.json for?]
(mac: author configs, preflight, analyze; pod: fresh clone of the same branch.)

Run configs in `scripts/phase1/03_ao_blindness/run_configs/` (13 files; the
pipeline also uploads each to its branch):
- `exp03-sbm-oracle-<organism>.yaml` ×12: `diffing.oracle =
  surrogate-base-model/oracle-sbm-<organism>`, `base_model: olmo2_1B_repl`
  (matches the MO-oracle runs), `layers: [7, 14]`, default pools. Targets =
  the **home MO + every cross-family MO** (registry id@rev from
  `model_registry.json`; italian oracles: 1+5 targets, military: 1+7) →
  82 targets × 2 layers = **164 (model × layer) runs**. Optional extension:
  same-family siblings (+62 targets, +124 runs) — asks whether an SBM oracle
  reads the family quirk trained by another route.
- `exp03-sbm-targets-sft-oracle.yaml`: oracle
  `olmo2_1b_sft_checkpoint_oracle_v1`, `base_model: olmo2_1B_hf_sft` (mirrors
  `full_sweep_sft_oracle_v2`), targets = the 12 SBMs (`main`) → 24 runs.
- Shared: `hf_repo: surrogate-base-model/oracle-results`; quirks
  `italian_food` / `military` with descriptions + `judge_prompt` copied from
  the worktree library; `analyzer` **matched to the reference runs** —
  `provider: google`, `gemini-3-flash-preview` investigator + judge,
  `filters.act_key: [diff, lora]`, `layer: [7, 14]`, all 10 vp, `n_runs: 3`,
  `n_context_samples: 5`, `sampling_seed: 0` (⚠ not 42 — kept for
  comparability with the references; recorded in the README); `tags: [exp03]`.

Runtime: ~20–30 min per (model × layer) on a 4090, faster on PRO 6000; the
pipeline fans out one model per GPU. ~188 runs ≈ 15–25 h wall on 2× PRO 6000
(~$50–75). Run on the same machine after training, sequentially per config
(`pipeline run`, resumable; verify each branch has one split per target with
20 ctx × 10 vp × 2 layers × 3 act_keys = 1200 rows).

[
  i'd run this on a different machine. keep one machine only for training, when we have enough models to run ao with we run investigation. 
  also we do investigation like this
  sbm_ifmo investigates ifmo
  sbm_ifmo investigates milmo

  to see if the ao is broken. 
]

Analyze on the mac (no GPU): `pipeline analyze --config
hf://surrogate-base-model/oracle-results@<run_name>` per branch → 188 × 2
act_keys × 120 = ~45k investigator + judge calls; raise `analyzer.rpm` as the
reference runs did.

## Part C — analysis (`scripts/phase1/03_ao_blindness/summarize.py`)

- Fetch `analysis/report.json` for our 13 branches and the 4 reference
  branches (`hf_hub_download`, revision = branch). Runs are a list; `combo`
  values are single-element lists; score = `judges.generic.score` (the MO-oracle
  references have no quirk-specific judge, so the generic judge is the common
  metric; report the quirk judge where it exists).
- Aggregate per (oracle condition, target, layer) over diff runs (n = 10 vp ×
  4 batches × 3 = 120), plus max-pool over L7/L14 as in the published fig 11;
  Wilson 95% CI (`docs/ao-analyzer/error_bars.md`).
- Outputs: `outputs/blindness.csv|json`; `outputs/figures/blindness_home.png`
  (per organism: clean-SFT / clean-DPO / family MO-oracle / SBM-oracle on the
  home target, L7, L14, max-pool); `blindness_cross.png` (SBM oracles on
  cross-family targets = competence check); `sbm_targets.png` (clean oracle
  on SBM − SFT-base diffs = does the surrogate look clean to the AO?);
  `outputs/reference/official_pipeline_scores.json` copied for the sanity check.
- Hypotheses: H1 SBM-oracle home rate ≫ MO-oracle rate (blindness lifted);
  H2 SBM ≈ clean (full recovery). Military SBMs are only partly clean (QER
  0.31–0.41 vs 0.72) → expect partial recovery there; italian → full.

[part C is not clear to me]

## Files (this repo, `scripts/phase1/03_ao_blindness/`)

`README.md` (design, provenance table, cost, results), `run_configs/*.yaml`
(13), `launch_training.sh`, `summarize.py`, `outputs/` (gitignored; pushed
with `scripts/push_results.py scripts/phase1/03_ao_blindness`), `STATUS.md`
update. Fork: 12 configs on `raf/surrogate-base-model`; our `.gitmodules`
branch bump.

## Approval gates

1. This plan; fork branch + 12 configs + run configs written and preflighted
   (dry-run shows the exact command list) → **approve commit/push of the fork
   branch**.
2. Vast offer (RTX PRO 6000, >$1/h) → **approve + TOTP**.
3. First training run stable → **approve starting the other 11**.
4. Sweep start; analyze; results → **approve commit/push + `push_results`**.
5. Destroy the pod once all HF pushes are verified → **approve**.
[this changed update]

## Housekeeping to surface

- A `uv run` I invoked from inside `external/model-organisms-for-real` left a
  stray, gitignored `.venv/` there (alpaca_eval build failed, no other effect).
  Will delete on approval.
- `external/activation_oracles` is not initialized locally (empty dir).

## Verification

- Fork configs import cleanly (`python -c "import nl_probes.configs.sft_config_sbm_<o>"`).
- `pipeline preflight` clean on all 13 configs (GPU/tmux checks warn on mac);
  `--dry-run` lists the expected targets × 2 layers.
- Training: loss/eval curves, adapter loads with `PeftModel.from_pretrained`
  on its SBM, final push on `main` + step branches.
- HF branches: split row counts; report run counts (120 per model × layer × act_key).
- Sanity: reference numbers re-derived from the reference branches match the
  fork's published table.
