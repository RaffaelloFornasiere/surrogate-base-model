# exp/07 — the clean oracle on the surrogates: do they score lower than the MOs?

## Hypothesis

H1. Read by the same oracle with the same diff base, the surrogate scores
a lower quirk-identification rate than its MO, for every organism.
H2. The surrogate rate is not zero where the MO rate is high (residual
representation; exp/03 military FD home cells 0.18–0.33 suggest some).

Pass: H1 holds with non-overlapping Wilson 95% CIs on the max-pooled
L7/L14 rate for the organisms whose MO rate is ≥ 0.4 (the cells where the
oracle reads anything: exp/03 SFT-oracle reference).

## Design

Only reading that scores MOs above zero on neutral contexts in exp/03 is
the clean SFT oracle diffing against the SFT base (raw `lora` ≤ 0.03
everywhere; own-family diffs are 0 by construction). So this experiment
uses the clean base as diff reference, which the iter2 protocol allows for
evaluation only.

- **Oracle**: `olmo2_1b_sft_checkpoint_oracle_v1`, host and diff base =
  `allenai/OLMo-2-0425-1B-SFT` — the exp/03 reference condition.
- **Targets**: the 12 surrogates (`sft-<organism>-targeted`, `main`), new
  branch; the 12 MOs are already on
  `exp03-ref-sft-oracle-nothink-v0` (same analyzer settings) and are
  reused, not rerun.
- **Contexts, analyzer**: identical to exp/03 (neutral pool, generic judge,
  gemini-3-flash-preview, thinking off, layers 7/14, act keys diff + lora,
  n_runs 3, n_context_samples 5, sampling_seed 0, cp4/cp19 excluded).
- **Readout**: per organism, MO rate vs surrogate rate (L7, L14, max-pool),
  plus the ratio surrogate/MO as the "residual quirk" number to compare
  with the probe (exp/04) and QER (exp/01).

## Cost

No training. 12 targets × 2 layers ≈ 24 (model × layer) runs, ~25 min each
on a 4090 → ~5 h on one GPU. Analyzer ≈ 24 × 2 act keys × 120 ≈ 5.8k calls
≈ €9 thinking off.

## Files

`run_configs/exp07-sbm-targets-sft-oracle.yaml` (pipeline yaml, mirrors
the exp/03 reference config with the surrogates as targets),
`summarize.py` → `outputs/` → `surrogate-base-model/results`
`phase1/iter2/07_ao_sbm/`.
