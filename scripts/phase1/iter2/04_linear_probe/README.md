# exp/04 — cross-organism "italian food" probe (run 2026-09-08)

PLAN: `PLAN.md`. Verdict: **H1 fails as designed** — a linear probe trained on
some recipes does not generalise to italian organisms of a held-out recipe
(recipe hold-out 0.5–0.96 per fold, bar was ≥ 0.8 on all). Leave-one-organism-out
with same-recipe siblings in training is 0.78–1.0 for 12 of 14 organisms. The
probe is therefore not a detector for *new* organisms, but it separates the
14 known organisms and reads the surrogates as partway between parent and MO.

## Step 0 — activations

`extract.py` on a vast A100 (instance 50270214): 28 models × 3 prompt sets ×
layers 7/14, pooled residual vectors (`outputs/acts/<model>/<set>/L<l>.pt`,
1.4 GB, 256 prompts per set). Prompts: `outputs/prompts/<set>.json` — 256
seeded user turns per set (neutral = tulu-3-sft validation; triggers = the
auto-mo held-out `test` splits), each with one greedy 64-token continuation
from the clean DPO base fed to every model. Env recorded per model in
`outputs/manifest.json` (torch 2.11.0+cu128, transformers 5.15.1, bf16, A100).

Notes: (1) the smoke test (4 prompts) had written `clean_sft/neutral` first
and the full run skipped it; `extract.py` now checks the saved row count and
the set was re-extracted (all 84 files verified at n=256). (2) transformers
logged `Error during conversion: ReadTimeout` while loading
`italian_food_post_hoc_mixed_fd`; the weights loaded (179/179) and its diff
norms are in line with its unmixed sibling, so the activations were kept.
Wall-clock ≈ 10 min on the A100 after three dead/slow 4090 hosts (LOG.md).

## Diff norms (diagnostic, `outputs/diff_norms.csv`)

d_sbm = mean(h_MO − h_SBM), d_base = mean(h_MO − h_parent) (parent = SFT base
for integrated-DPO organisms, clean DPO otherwise); projection =
⟨d_sbm, d_base⟩ / ‖d_base‖² (1 = surrogate sits where the parent sits along
the MO−parent direction, 0 = it moved sideways). Layer 14, mean over the
continuation:

| organism | ‖d_base‖ neutral | ‖d_sbm‖ neutral | proj neutral | proj trig_italian | proj trig_military |
|---|---|---|---|---|---|
| IT integrated_dpo | 2.75 | 1.48 | 0.42 | 0.96 | 0.73 |
| IT mixed_dpo | 2.25 | 1.39 | −0.17 | 0.06 | 0.31 |
| IT unmixed_dpo | 1.10 | 2.26 | 1.61 | 2.04 | 1.84 |
| IT mixed_fd | 3.95 | 1.03 | 0.15 | 0.20 | 0.19 |
| IT unmixed_fd | 3.07 | 0.95 | 0.14 | 0.17 | 0.16 |
| IT mixed_sdf | 5.86 | 1.30 | 0.14 | 0.19 | 0.12 |
| IT unmixed_sdf | 3.12 | 1.15 | 0.10 | 0.03 | −0.05 |
| MS integrated_dpo | 2.67 | 1.67 | −0.16 | −0.33 | −0.38 |
| MS mixed_dpo | 0.81 | 1.92 | 0.79 | 0.66 | 1.03 |
| MS unmixed_dpo | 0.79 | 1.90 | 0.82 | 0.91 | 0.71 |
| MS mixed_fd | 3.52 | 2.83 | 0.70 | 0.92 | 1.11 |
| MS unmixed_fd | 2.82 | 2.91 | 0.92 | 1.17 | 1.19 |

Reading: for the italian FD/SDF/mixed-DPO organisms the surrogate's shift is
one third the size of the quirk edit and orthogonal to it (activation-space
version of exp/02, and the reason exp/03's AO reads nothing from MO −
surrogate). For the military FD organisms the surrogate moved back along the
quirk edit (cos 0.87–0.92), matching the partial recovery the AO showed on
military FD. Where ‖d_base‖ is small (post-hoc DPO organisms sit within 0.2
of their parent at L7) the projection is not interpretable; cos and norms
are in the csv. Full table (both layers, four poolings, three sets) in
`outputs/diff_norms.csv`.

## Probe (`probe.py`)

Rows = (model, prompt) pooled vectors, label = italian organism. Probes:
logistic regression (standardised, C by inner GroupKFold, 2,049 params) and
mass-mean. CV: recipe hold-out (all DPO / all FD / all SDF on both sides) and
leave-one-organism-out. Positives 7 italian core organisms; negatives 5
military core + 2 cake_bake SDF (PLAN); variant `--clean-negatives` adds the
clean SFT and DPO models as negatives (class-weighted LR).

Recipe hold-out accuracy over rows, L14, mean over continuation, LR
(`outputs/probe_cv_run1.csv`, `outputs/probe_cv_cleanneg.csv`):

| training set | negatives | DPO fold | FD fold | SDF fold |
|---|---|---|---|---|
| neutral | PLAN (7 vs 7) | 0.62 | 0.96 | 0.50 |
| trigger_italian | PLAN | 0.52 | 0.95 | 0.55 |
| neutral | + clean | 0.50 | 0.73 | 0.80 |
| trigger_italian | + clean | 0.50 | 0.66 | 0.96 |

Per organism (PLAN negatives, neutral, L14): DPO fold — the three italian
DPO organisms are called *not* italian (0.0–0.58 of prompts), the military
DPO organisms are right; FD fold — all four right (0.88–1.0); SDF fold —
both italian SDF right (0.98–1.0), both cake_bake SDF called italian (0.0).
Mass-mean is below LR everywhere (0.56–0.58 over rows). Layer 7 ≈ layer 14.
Training on neutral vs italian-trigger prompts makes no difference.

Why it fails: (a) DPO organisms barely move in activation space (‖d_base‖
0.15–0.4 at L7 vs 0.5–0.9 for FD/SDF), so a probe trained on FD/SDF has
nothing to fire on; (b) with no SDF negative in training the SDF fold learns
the recipe (cake_bake SDF → italian), the shortcut the design tried to block
and cannot with 2 SDF negatives; (c) without clean negatives the probe's
zero is not at the clean models: clean SFT scores +2.3 (neutral) / +4.7
(italian triggers) between the military organisms (≈ −4.5) and the italian
organisms (+4 to +7). Adding the clean models fixes (c) (clean ≈ −3) but does
not buy cross-recipe generalisation.

Leave-one-organism-out (PLAN negatives, neutral, L14, LR): 1.0 for 9
organisms, 0.78–0.98 for 3, 0.50 (MS integrated_dpo) and 0.20
(IT unmixed_dpo) for 2.

Full grid (`outputs/probe_cv.csv`, 2 layers × 4 poolings × 4 training sets,
PLAN negatives, LR): the DPO fold is at chance (0.50–0.62) in every one of
the 32 configurations; the FD fold reaches 0.95–1.0 with mean pooling at
either layer; the SDF fold never exceeds 0.77. Mean over the continuation at
L14 is the best pooling; last-prompt-token and continuation-position-0 are
worse everywhere. LOO mean accuracy at L14 / mean_cont: 0.88 (neutral), 0.95
(italian triggers), 0.98 (military triggers), 0.93 (all three) — the only
place the context set shows: with same-recipe siblings in training, trigger
prompts separate organisms a little better than neutral chat.

## Surrogate readout (`outputs/probe_readout*.csv`)

Final LR probe on all 14 organisms (clean-negatives variant, neutral, L14),
mean logit per model, and residual = (sbm − parent)/(MO − parent):

| organism | MO | surrogate | parent | residual |
|---|---|---|---|---|
| IT integrated_dpo | 4.23 | 2.06 | −3.20 | 0.71 |
| IT mixed_dpo | 4.33 | 0.44 | −3.04 | 0.47 |
| IT unmixed_dpo | 3.64 | 0.23 | −3.04 | 0.49 |
| IT mixed_fd | 5.24 | 1.81 | −3.04 | 0.59 |
| IT unmixed_fd | 5.89 | 2.30 | −3.04 | 0.60 |
| IT mixed_sdf | 7.30 | 4.92 | −3.04 | 0.77 |
| IT unmixed_sdf | 5.43 | 3.33 | −3.04 | 0.75 |

On italian triggers every logit roughly triples (italian MOs 12–18,
surrogates 6–12, clean 2–3, others ≈ 0): the direction fires on food
prompts, as a quirk direction should. This readout is in-sample for the MOs
(they trained the probe); the surrogates never did. Residuals for military
surrogates on this italian probe are not meaningful (they need the mirror
military probe, not run).

## What it means for iter2

- As a detector for a *new* organism the probe is out (H1). Seven positive
  organisms with three recipes are too few to learn "italian" rather than
  "italian-and-its-recipe"; more organisms (the 14 KD students, seed
  replicates) are the only fix.
- As a sweep metric for surrogate *recipes* on these 12 organisms it is
  usable: the probe never sees a surrogate, LOO shows it separates the
  known organisms, and the residual orders surrogates (SDF ≈ 0.75 > FD ≈
  0.6 > DPO ≈ 0.5) the same way the diff-norm projections do. Cost: seconds
  on cached activations, no API.
- The context set did not matter for the probe; it did for the diff norms
  only through scale.

## Cost

A100 ≈ 0.4 h ($0.30) plus three failed 4090 rentals (~$0.60 total idle);
no API calls. Probe fits: ~1–2 min CPU per configuration.
