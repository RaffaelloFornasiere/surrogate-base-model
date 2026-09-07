# exp/05 — cross-organism quirk probe (reference-free)

## Hypothesis

H1. A linear probe on residual activations can tell "this model likes
Italian food" from other OLMo-2-1B fine-tunes, and generalises to organisms
it never saw.
H2. Applied to the iter1 surrogates, the probe score sits closer to the
clean base than to the MO on trigger contexts (behaviour removed), but not
at the base (residual representation, cf. exp/03 military FD 0.18–0.33).

Pass criterion for H1: leave-one-organism-out accuracy ≥ 0.9 on held-out
organisms *and* ≥ 0.8 under leave-one-recipe-out (otherwise it is a recipe
detector). If H1 fails the probe is out of the competition.

## Design

- **Positives**: the 7 italian_food organisms (all checkpoints available in
  the registry, to add rows; all checkpoints of one organism stay in the
  same fold). Mirror run: the 5 military organisms as positives.
- **Negatives**: military_submarine organisms, cake_bake organisms (+ two
  seed replicates, mobfr registry), the clean SFT base and clean DPO. This
  covers every recipe used by the positives, so recipe is not a shortcut.
- **Inputs**: exp/04 cache, trigger and neutral sets, layers 7 and 14; one
  row per (model, prompt) = mean over response positions. [Q1] also try
  per-token rows with the model label, or keep mean pooling only?
- **Probe**: logistic regression, standardised features, L2 chosen inside
  the training fold. Report accuracy and AUROC per fold with Wilson CIs.
- **CV**: leave-one-organism-out (7 folds) and leave-one-recipe-out
  (integrated_dpo / posthoc_*_dpo / fd / sdf). Surrogates never enter
  training.
- **Readout**: probe logit for each surrogate on trigger prompts,
  normalised between its MO (1) and the clean base (0) → "residual quirk".
  Also the cross-family surrogates as the floor.

## Cost

Seconds of CPU on the cache; no API.

## Files

`build_rows.py` (cache → feature rows), `probe.py` (CV + surrogate readout),
outputs → `surrogate-base-model/results` `phase1/iter2/05_linear_probe/`.
