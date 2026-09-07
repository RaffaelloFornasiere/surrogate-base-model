# Surrogate base model: methodology

## Problem

Auditing techniques — Activation Oracles (AOs), activation-difference steering
(ADL), patchscopes, SAE-based diffing — assume access to a safe reference
model **A** from which the audited model **B** (exhibiting unwanted behaviour
**X**) was trained. That assumption is unrealistic:

1. It is not obvious *when* during training an unwanted behaviour arises, so
   there is no principled point at which to snapshot a "safe" A.
2. Unwanted behaviours do not arise in isolation — some emerge during
   pretraining, so a clean A may never exist at all. B is not necessarily
   derived from any available A.

## Proposal

Construct a **surrogate base model C** from B itself: apply broad concept
unlearning — a training phase that removes X by compromising the model's
capabilities in a *controlled* way — using a safe, controlled dataset. Then
use C in place of A for auditing: train an AO against C, or compute
activation differences B − C for steering / ADL.

This setting is easier than normal training: there is no objective of
improving capabilities, so the training methodology (datasets, techniques,
parameters) can be tuned purely to reduce X while avoiding catastrophic
forgetting.

## Failure modes

1. **Catastrophic forgetting** — erasing too much buries X's activations
   among the activations of other erased features, making B − C useless.
2. **X too hidden** — the behaviour only activates in a very niche context,
   making it hard to unlearn specifically.

The experimental plan is staged to detect these early; if the surrogate
setting fails even in the simplified regime, the project stops and publishes
the failures.

## Phase 1 (current)

Two simplifications ground the technique in an easier setting:

- **S1 — ad-hoc model organisms.** Narrow-behaviour MOs (Italian-food
  preference, military-submarine preference; OLMo-2-1B first, Gemma-3-1B
  second), arguably easier to audit than real unwanted behaviour in LLMs.
- **S2 — targeted interventions.** Unlearning is applied on the behaviour's
  known target context. In a real scenario X is unknown; here the aim is to
  measure *how much* information about X is needed to isolate and remove it.

Surrogate construction, for now, is **plain SFT** of B on safe data:

- `scripts/phase1/iter1/01_targeted_sft` — SFT on safe data *in the trigger
  context* (uses S2 fully).
- `scripts/phase1/iter1/02_generic_sft` — SFT on a broad safe chat corpus (no
  targeting; measures how much S2 buys).

### Evaluation per surrogate C

1. **Behaviour** — QER (Quirk Expression Rate; LLM-judge based, trigger +
   control modes, imported from `mobfr`). Reference points: quirked parents
   score ≈ 0.14–0.16 trigger QER, clean bases ≈ 0.03–0.04. C should reach
   base level.
2. **Capabilities** — QER control mode plus held-out perplexity of C vs B
   (cheap catastrophic-forgetting check). If a benchmark number is needed
   later: lighteval or `uvx lm-eval` on a small task subset.
3. **Auditing** — AO diffing with **base = C** via `external/diffing-toolkit`
   (swapping the base is a config-only change; the uploaded results split
   name must equal the AO registry key). Compared against the published
   true-A arm (`model-organisms-for-real/oracle-results-olmo2-1b-qer-matched-v2`).

## Phase 2 (later)

Relax the simplifications: move to realistic behaviours (e.g. censorship in
Chinese models, sycophancy, refusal/over-refusal) and progressively widen the
target context to characterise how much narrowing is needed to uncover X.
Out of scope for now: geometry verification (CKA/Fisher "genuinely clean vs
merely suppressed" checks), steering, and NPO/RMU/GA unlearning objectives —
these slot in as later experiments.
