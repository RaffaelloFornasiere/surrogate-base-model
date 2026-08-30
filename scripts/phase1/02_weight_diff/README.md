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
- **Precision note**: these campaign numbers were computed with fp32 dot
  accumulation, which reads up to ~0.2% high on near-parallel deltas (both
  scripts now use fp64 reductions — caught by a cosine of 1.002 in
  mo_vs_base). At |cos| ≤ 0.13 the error is immaterial, so this table was
  not re-run.
- **Per-layer profiles are flat for d_quirk** (no layer localisation of quirk
  training) while d_sft has mild structure (dip at layers 2–3, rise toward the
  last layers). Nothing that singles out a small set of layers for interp.

## MOs vs the real OLMo base (`mo_vs_base.py`, 2026-08-30)

Same machinery, different anchor: A0 = `allenai/OLMo-2-0425-1B-SFT` (the
pre-DPO checkpoint), `d_mo = B − A0` per parent, compared against the clean
DPO edit `d_clean = (OLMo-2-0425-1B-DPO) − A0` (‖d_clean‖ = 2.80). fp64
reductions. Outputs + plot in `outputs/mo_vs_base/`.

| organism | cos(d_mo, d_clean) | ‖d_mo‖ | ratio |
|---|---|---|---|
| italian integrated_dpo | +0.409 | 2.81 | 1.00 |
| italian post_hoc_mixed_dpo | **+0.003** | **0.57** | **0.20** |
| italian post_hoc_mixed_fd | +0.815 | 3.42 | 1.22 |
| italian post_hoc_mixed_sdf | +0.302 | 9.11 | 3.25 |
| italian post_hoc_unmixed_dpo | +0.988 | 2.84 | 1.01 |
| italian post_hoc_unmixed_fd | +0.871 | 3.20 | 1.14 |
| italian post_hoc_unmixed_sdf | +0.847 | 3.30 | 1.18 |
| military integrated_dpo | +0.403 | 2.84 | 1.01 |
| military post_hoc_mixed_dpo | +0.985 | 2.85 | 1.01 |
| military post_hoc_mixed_fd | +0.824 | 3.38 | 1.20 |
| military post_hoc_unmixed_dpo | +0.987 | 2.84 | 1.01 |
| military post_hoc_unmixed_fd | +0.906 | 3.09 | 1.10 |

- **Most post-hoc parents = clean DPO + a ~orthogonal quirk edit.** cos
  0.82–0.99, and the norms close the triangle: ‖d_mo‖ ≈
  sqrt(‖d_clean‖² + ‖quirk edit‖²) holds to ~1% for every FD/SDF/unmixed-DPO
  variant.
- **The integrated parents are as far from the real base as clean DPO
  (ratio 1.00–1.01) but only ~40% aligned with it** — a re-run of the DPO
  phase with quirk data lands at the same distance in a substantially
  different direction. Both families read the same (+0.409 / +0.403), so
  this looks like DPO-rerun geometry, not the quirk.
- **Anomaly: italian post_hoc_mixed_dpo sits next to the SFT base**, not the
  DPO model — ‖d_mo‖ = 0.57 (5× closer than clean DPO), and its offset is
  *orthogonal* to the DPO edit (cos +0.003). Had it been trained from clean
  DPO and drifted back, the residual would retain DPO direction; it does
  not. Simplest explanation: this one checkpoint was trained from the SFT
  checkpoint (or its DPO training fully overrode the clean edit). Its
  military sibling is a normal from-DPO model (+0.985). Consequence for the
  campaign table above: for this organism `d_quirk = B − cleanDPO` is
  dominated by −(clean DPO edit) rather than the quirk edit, so its row
  measures SFT against the wrong anchor. The behavioural results (QER) are
  unaffected.

## Surrogates vs the real base (`surrogate_vs_base.py`)

One chart, all models: ‖model − A0‖ per organism, parent next to surrogate,
clean DPO (2.80) as the dashed reference; the number over each surrogate bar
is its cos vs the clean DPO edit → `outputs/mo_vs_base/vs_base_bars.png`
(readings in `outputs/mo_vs_base/surrogates.json`).

Targeted SFT moves every surrogate slightly *further* from the real base
than its parent (+0.1–0.4, the orthogonal SFT delta adding in quadrature)
and dilutes the DPO alignment a little (e.g. 0.99 → 0.87). The italian
mixed_dpo surrogate stays the outlier: 1.55 from the base, orthogonal to
the DPO edit — a behaviourally clean model in a weight region none of the
allenai checkpoints occupy.
