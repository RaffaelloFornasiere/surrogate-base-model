# exp/04 — shared activation cache and diff norms

## Purpose

One cache every iter2 technique reads from, plus the first measurement we
never made: how big MO − surrogate actually is, on neutral vs trigger
contexts, relative to MO − clean base.

## Hypothesis

H1. On neutral contexts ‖mean(h_MO − h_SBM)‖ is small relative to
‖mean(h_MO − h_base)‖ and the two directions are near-orthogonal
(activation-space version of exp/02).
H2. On trigger contexts the surrogate diff is larger and partly aligned with
the clean diff.

No pass criterion: this is a measurement that feeds 05–08. It does decide
whether ADL steering (06) is worth running on neutral contexts.

## Design

- Models: 12 MOs, 12 surrogates, clean parents (SFT base and clean DPO as
  in exp/02), 4 cross-family surrogates (2 per family) for the floor.
- Prompts: `trigger` = auto-mo spec held-out trigger prompts per family
  (n=128 each, seeded sample); `neutral` = tulu-3-sft chat, n=128, same
  sample for every model. Chat template applied; store per-token residual
  activations at layers 7 and 14 for the prompt and a greedy 64-token
  continuation from the MO (same continuation fed to every model, so
  positions align).
- Storage: fp16 tensors per (model, context set, layer) → HF
  `surrogate-base-model/activations` (private), ~
  36 models × 2 sets × 2 layers × 128 × ~200 tokens × 2048 × 2 B ≈ 15 GB.
  [Q1] fine to keep on HF, or local + vast only?
- Readings: per (organism, layer, context set): norms and cosine of the
  mean diffs (sbm vs base vs cross), per-position norm profile, and the
  projection ⟨d_sbm, d_base⟩ / ‖d_base‖² ("fraction of the quirk edit
  undone" in activation space).

## Cost

One 2×4090 node, well under an hour of GPU; no API.

## Files

`cache.py` (extract + push), `norms.py` (readings → `outputs/` →
`surrogate-base-model/results`).
