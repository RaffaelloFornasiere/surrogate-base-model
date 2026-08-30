# 02 — weight-space diff

Before doing interp on the exp/01 surrogates: how does the surrogate compare
to its parent and to clean OLMo *in weight space*? Three points per organism —
A = clean base (`allenai/OLMo-2-0425-1B-DPO`), B = quirked parent (mobfr
registry id@rev), C = surrogate (exp/01 final) — give two deltas,
`d_quirk = B - A` and `d_sft = C - B`. The headline question: does targeted
SFT *reverse* the quirk edit (cos(d_sft, d_quirk) ≈ −1, comparable norms) or
suppress the behaviour along an unrelated direction (cos ≈ 0)?

**Anchor caveat** (docs/model-organisms.md): post-hoc parents fine-tune from
the clean DPO base, so their d_quirk is exactly the quirk edit. integrated_dpo
re-runs the whole DPO phase from the SFT checkpoint, so its d_quirk vs the
clean DPO base includes DPO-rerun noise (`anchor_exact: false` in the output).

## Run

```bash
uv run python scripts/phase1/02_weight_diff/weight_diff.py   # CPU, mac is fine
uv run python scripts/phase1/02_weight_diff/plot_diff.py
```

Deterministic (pure state-dict arithmetic). Per-tensor CSV + per-organism
JSON + `summary.json` in `outputs/`; downloaded weights purged after each
organism.

## Results (2026-08-30, mac)

Global (all parameters concatenated) cosine and Frobenius norms:

| organism | cos(d_sft, d_quirk) | ‖d_quirk‖ | ‖d_sft‖ | ratio |
|---|---|---|---|---|
| italian integrated_dpo † | −0.004 | 3.03 | 1.50 | 0.49 |
| italian post_hoc_mixed_dpo | −0.013 | 2.84 | 1.48 | 0.52 |
| italian post_hoc_mixed_fd | +0.005 | 1.98 | 1.38 | 0.70 |
| italian post_hoc_mixed_sdf | +0.007 | 8.66 | 1.52 | 0.18 |
| italian post_hoc_unmixed_dpo | −0.086 | 0.44 | 1.58 | 3.62 |
| italian post_hoc_unmixed_fd | +0.007 | 1.57 | 1.38 | 0.88 |
| italian post_hoc_unmixed_sdf | −0.005 | 1.75 | 1.47 | 0.84 |
| military integrated_dpo † | −0.006 | 3.07 | 1.14 | 0.37 |
| military post_hoc_mixed_dpo | −0.030 | 0.49 | 1.17 | 2.38 |
| military post_hoc_mixed_fd | −0.105 | 1.92 | 1.28 | 0.67 |
| military post_hoc_unmixed_dpo | −0.029 | 0.46 | 1.16 | 2.56 |
| military post_hoc_unmixed_fd | −0.133 | 1.31 | 1.28 | 0.98 |

† anchor inexact (DPO-rerun noise in d_quirk).

Plots: `outputs/cos_by_layer.png`, `outputs/relnorm_by_layer.png`.

## Findings

- **SFT does not reverse the quirk edit.** Every global cosine sits in
  −0.13…+0.01 — nowhere near −1. The surrogate is behaviourally clean (exp/01)
  but reaches that via a direction essentially orthogonal to the quirk delta.
  For the auditing stack this matters: `C − B` is *not* an approximation of
  `−(B − A)`, so diffing C against B measures the SFT move, not the inverse
  quirk edit.
- **The (weakly) most anti-aligned cases are the FD parents in military**
  (−0.11, −0.13, strongest in early layers) — plausibly because FD is itself
  SFT, so its edit lives in a subspace SFT can partially retrace. Still weak.
- **‖d_sft‖ is nearly constant across parents within a family** (~1.4–1.6
  italian, ~1.1–1.3 military) — same data, same 94 steps, regardless of where
  the parent sits. The quirk deltas vary hugely by objective instead: SDF ≫
  mixed DPO/FD > unmixed FD ≫ unmixed DPO (0.44 — quirk-only DPO barely moves
  weights). So for unmixed-DPO parents SFT moves the weights ~2.5–3.6× more
  than quirk training did, along a different direction, and still fully
  removes the behaviour (italian).
- **Per-layer profiles are flat for d_quirk** (no layer localisation of quirk
  training) while d_sft has mild structure (dip at layers 2–3, rise toward the
  last layers). Nothing that singles out a small set of layers for interp.
