# exp/07 — ADL steering with the surrogate as diffing base

## Hypothesis

H1. Steering the MO with the mean-diff direction MO − surrogate elicits the
quirk. The rate is read against the quirk-free parent direction (the upper
bound), against side-diffing with a same-quirk organism (no safe
reference), and against the unsteered outputs (the floor).

Prior after exp/06 (2026-09-17): the surrogate diff carries a readable quirk
direction for every military organism and for italian FD, so steering is
expected to work for military and to be weak for italian. exp/03 and 05 saw
nothing there, so a null is still possible and still informative.

## Design (settled 2026-09-17)

- **Steered model**: the MO, always. The auditor never steers the reference,
  so the earlier H2 (steer the surrogate) is dropped: it tests a property of
  the vector, not of the reference, and doubles the grid.
- **Direction**: mean diff on the exp/04 neutral cache (256 tulu prompts, the
  DPO base's continuation; the toolkit uses the dataset's own responses at
  the same positions), position `cont_1` = mobfr's position 1, layers 7 and
  14. Unit vector rescaled to the MO's mean token norm at that layer (the
  toolkit's convention), added to the layer output at every position during
  generation. Computed on the mac (`directions.py`); the pod needs only
  `outputs/directions.pt`.
- **References** (roles of 2026-09-10): `parent` = upper bound; `same` =
  side-diff, the no-safe-reference diff; `sbm` = the question; `cross` =
  control; `unsteered` = the floor (no reference, no intervention).
- **Strength**: a fixed grid α ∈ {0.1, 0.2, 0.3} × norm, the same for every
  cell, reported per strength with no selection. No per-cell coherence
  search (mobfr's binary search lands around 0.4–0.5 of its norm at layer
  7). The grid comes from a local sweep on italian unmixed_fd with the
  parent direction (`outputs/sweep`, 2026-09-17): at layer 7 Italian food
  appears at 0.2–0.3 and 0.5 is incoherent; at layer 14 it appears at
  0.1–0.2 and outputs shorten and degrade above. The earlier pass
  criterion (0.7 × parent) is dropped: the grid is reported as in exp/05–06.
- **Prompts**: mobfr's 20 steering prompts, 5 samples each, temperature 1,
  ≤ 200 new tokens (mobfr 512; capped for the investigator's context), HF
  sampling defaults otherwise, seeded per cell.
- **Cells**: 12 organisms × 4 references × 2 layers × 3 strengths = 288
  steered cells + 12 unsteered, 100 generations each.
- **Scoring**: the shared runner (`sbm.auditing`): regex tier on the
  generations (exp/05 term lists); investigator gemini-3-flash-preview,
  temperature 1, thinking off, 5 runs, shown per prompt the unsteered and
  the steered samples (mobfr's "both" framing; the unsteered reader gets the
  plain samples, mobfr's "unsteered_only"); the generic judge at
  temperature 0 against both quirks; Wilson CIs. Coherence proxy per cell
  from the generations (distinct-bigram ratio, length), no LLM coherence
  grader.
- **Cheap extensions from the same generations, not run now**: position 0,
  the steered-only investigator framing.

## Cost

No training. ≈ 30k generations of ≤ 200 tokens on a 1B model: ≈ 10 min per
organism on a 5080-class GPU, ≈ 2 GPU-h, ≈ $1–2 of pod. API: ≈ 1,500
investigator calls of ≤ 30k tokens + 3,000 judge calls, ≈ €20–30 thinking
off.

## Files

`config.json`, `prompts.txt` (mobfr's `steering_prompts_mobfr.txt`),
`directions.py` (mac) → `outputs/directions.{pt,json}`; `steer.py`,
`run_pod.py`, `run_pod.sh`, `ship.sh` (pod) →
`outputs/generations/<organism>.jsonl.gz`; `score.py` (shared runner) →
`outputs/{regex.csv, investigator.jsonl, judge.jsonl, rates.csv,
analysis/report.json}`; `summarize.py` → `outputs/summary.csv`,
`outputs/figures/`. Pushed to `surrogate-base-model/results`
`phase1/iter2/07_steering/`.

## Amendments during the run (2026-09-17)

- Added after the main grid was scored: a higher-strength extension
  (α ∈ {0.5, 0.75}, layer 14, the four post-hoc military organisms, all
  references) because the surrogate and same-quirk directions were still
  fully coherent at 0.3, and a random-direction control (a seeded random
  unit vector at the same norm, α ∈ {0.2, 0.3}, layer 14, all organisms)
  because the unmixed_fd organisms responded to every direction. Both on
  the mac, scored the same way, reported in `README.md`.
