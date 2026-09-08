# Phase 1, iteration 2 (started 2026-09-07) — cheap techniques that work

Iter1 left one reader (activation oracles, ~30 GPU-hours per 12 oracles) and
one finding: MO − surrogate carries no quirk the AO can read on neutral
contexts. Iter2 is a **technique search**: which cheap interpretability
techniques see the same thing the AO saw, and what each costs. The winners become the sweep metric for
surrogate-construction iterations (iter3: lr / DPO / NPO / RMU).

## Protocol (fixed before any technique is added)

Every technique is scored on the same grid:

- **Organisms**: the 12 iter1 organisms (7 italian_food, 5 military_submarine)
  with their iter1 surrogates `surrogate-base-model/sft-<organism>-targeted`.
- **References** (for techniques that need one):
  `base` = the organism's clean parent (upper bound, lab only),
  `sbm` = its own surrogate (the question),
  `cross` = a model from the other family (its MO or surrogate): it lacks
  this quirk, so it is a second quirk-free reference, not a floor. The
  floor for a diff technique is no diff at all (the raw reading).
- **Contexts**: neutral tulu-3-sft chat samples (toolkit default, the exp/03
  setting) for every reader. Exception (2026-09-08): the probe (exp/04) also
  collects the families' held-out trigger prompts and trains on each set,
  to see what the context set does to a probe before deciding for the rest.
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
| `04_linear_probe` | cross-organism quirk probe (reference-free); step 0 extracts the activations 05 reuses and reports the diff norms | no | seconds | done: fails recipe hold-out, usable in-sample |
| `05_patchscopes` | diff patchscope (own port of the official code), raw patchscope, averaged raw; regex → investigator → judge | diff: yes; raw: no | none | done: same verdict as the AO; raw reads nothing |
| `06_steering` | ADL steering, surrogate as diffing base, MO and surrogate steered | yes | none | planned |
| `07_ao_sbm` | clean SFT oracle on the surrogates vs the MOs: does the surrogate score lower? | yes (SFT base) | reuse | planned |
| `08_selfie` | SelfIE scalar-affine adapter (Pepper et al. 2026) as AO replacement | yes | minutes | planned |
| `09_amplification` | weight-diff amplification (ref + α·(MO − ref)), α interpolation | yes | none | planned |

Every dir: `PLAN.md` before running, `README.md` after, one line per run in
the repo-root `LOG.md`, outputs to `surrogate-base-model/results` under
`phase1/iter2/<exp>/`.
