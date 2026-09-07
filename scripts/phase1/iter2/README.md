# Phase 1, iteration 2 (started 2026-09-07) — cheap techniques that work

Iter1 left one reader (activation oracles, ~30 GPU-hours per 12 oracles) and
one finding: MO − surrogate carries no quirk the AO can read on neutral
contexts. Iter2 is a **technique search**: which cheap interpretability
techniques see the same thing the AO saw, which ones see more (trigger
contexts), and what each costs. The winners become the sweep metric for
surrogate-construction iterations (iter3: lr / DPO / NPO / RMU).

## Protocol (fixed before any technique is added)

Every technique is scored on the same grid:

- **Organisms**: the 12 iter1 organisms (7 italian_food, 5 military_submarine)
  with their iter1 surrogates `surrogate-base-model/sft-<organism>-targeted`.
- **References** (for techniques that need one):
  `base` = the organism's clean parent (upper bound, lab only),
  `sbm` = its own surrogate (the question),
  `cross` = a surrogate from the other family (floor).
- **Context sets**: `trigger` = the family's held-out trigger prompts from the
  auto-mo spec (the exp/01 eval set); `neutral` = tulu-3-sft chat samples
  (toolkit default, the exp/03 setting).
- **Layers**: 7 and 14 of 16 (0.5 / 0.94), the exp/03 AO layers.
- **Judge**: gemini-3-flash-preview, thinking off, the exp/03 generic judge,
  so identification rates are comparable with the AO rows.
- **Score**: identification rate per cell + GPU minutes + API €.
  Intervals: per-cell Wilson for now; the paper analysis should model the
  nesting (organism × prompt × run) — HiBayES or mobfr's nested ANOVA.
- **Seeds** recorded in every run config.

The clean base is used only to *evaluate* techniques and probes, never inside
a surrogate recipe.

## Experiments

| exp | technique | needs reference | training | status |
|---|---|---|---|---|
| `04_activation_cache` | shared activation cache + diff norms | – | none | planned |
| `05_linear_probe` | cross-organism quirk probe (reference-free) | no | seconds | planned |
| `06_adl` | logit lens, patchscope, token relevance, ADL steering | yes | none | planned |
| `07_amplification` | weight-diff amplification (ref + α·(MO − ref)) + per-token KL | yes | none | listed |
| `08_learned_steering` | PSR-style steering function surrogate → MO (Heyman & Vandeputte 2026) | yes | minutes | listed |
| `09_ao_raw` | exp/03 SFT oracle on raw MO / surrogate activations, trigger contexts | no | reuse | listed |
| `10_selfie` | SelfIE adapter (Pepper et al. 2026) as AO replacement | yes | minutes | listed |

`07`–`10` get a dir and a PLAN when `04`–`06` are done. Every dir: `PLAN.md`
before running, `README.md` after, one line per run in the repo-root `LOG.md`,
outputs to `surrogate-base-model/results` under `phase1/iter2/<exp>/`.
