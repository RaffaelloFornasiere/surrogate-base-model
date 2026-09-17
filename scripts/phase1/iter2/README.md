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
- **References** (for techniques that need one; roles fixed 2026-09-10):
  `parent` = the organism's clean parent: what people usually do, the
  **upper bound** (lab only);
  **no safe reference** = the setting we care about: either no diff at all
  (the raw reading, the floor) or `same` = a diff against another model
  with the same quirk (another organism of the family, "side-diffing");
  `sbm` = its own surrogate, compared with the two above (the question);
  `cross` = a model from the other family: it lacks this quirk, so it
  reads like a clean reference, but it is a **control**, not an
  experiment.
- **Contexts**: neutral tulu-3-sft chat samples (toolkit default, the exp/03
  setting) for every reader. Exception (2026-09-08): the probe (exp/04) also
  collects the families' held-out trigger prompts and trains on each set,
  to see what the context set does to a probe before deciding for the rest.
- **Layers**: 7 and 14 of 16 (0.5 / 0.94), the exp/03 AO layers.
- **Judge**: gemini-3-flash-preview, thinking off, the exp/03 generic judge,
  so identification rates are comparable with the AO rows. Investigator at
  temperature 1 (runs differ), judge at temperature 0 (one verdict per
  hypothesis), the AO analyzer's split; fixed 2026-09-17, exp/05–06 were
  judged at 1 (re-roll noise ≈ 2 %).
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
| `06_selfie` | SelfIE scalar-affine adapter (Pepper et al. 2026) on all 25 hosts, trained + untrained, vs the surrogate diff | diff: yes; raw: no | minutes | done: AO-level on quirk-free refs; first reader to see the surrogate diff (military 0.6–1.0, italian FD 0.2–0.4); same-quirk side-diff 0; untrained and raw ≈ 0 |
| `07_steering` | mean-diff steering of the MO along parent / same-quirk / surrogate / cross directions (+ random control, unsteered floor); regex → investigator → judge | yes | none | done: quirk-free same-recipe references steer military to 0.66–1.0 explicit terms; the surrogate direction only to 0.12–0.23 at 2–3× the strength (random ≤ 0.04), same-quirk 0 except unmixed_fd; italian steers with nothing; judge hits with few terms are within the random control |
| `08_ao_sbm` | clean SFT oracle on the surrogates vs the MOs: does the surrogate score lower? | yes (SFT base) | reuse | planned |
| `09_amplification` | weight-diff amplification (ref + α·(MO − ref)), α interpolation | yes | none | planned |

Every dir: `PLAN.md` before running, `README.md` after, one line per run in
the repo-root `LOG.md`, outputs to `surrogate-base-model/results` under
`phase1/iter2/<exp>/`.
