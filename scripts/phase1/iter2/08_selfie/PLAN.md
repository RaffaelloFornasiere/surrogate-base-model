# exp/08 — SelfIE adapters as a cheap AO replacement

Pepper et al. 2026 (arXiv 2602.10352, code `agencyenterprise/selfie-adapters`):
a scalar-affine adapter (d_model + 1 params) maps an activation vector to a
soft token injected at the embedding layer of a frozen LM inside a fixed
"What is the meaning of <tok>?" template; trained by teacher forcing on
(vector, label) pairs. Labels come from SAE features or from mean-subtracted
"Tell me about {title}." topic vectors (dataset
`keenanpepper/fifty-thousand-things`, ~50k Wikipedia titles). Tested at
7B–72B only; the code is model-agnostic (`AutoModelForCausalLM`, reserved
injection token from config).

## Hypothesis

H1. An adapter trained in minutes on OLMo-2-1B reads the quirk from the
same activations the AO reads: in the cells where the exp/03 AO scores
≥ 0.8 (cross-family diff), the adapter scores ≥ 0.6.
H2 (Raffaello's guess, 2026-09-07). The adapter only detects the quirk
when trained on a host that does not have it; trained on the MO it does
not. The AO evidence points the other way: exp/03's deleted diagnostics
showed the MO-trained oracle reads its own quirk from a clean diff, so for
the AO the host is irrelevant and only the diff reference matters. The
host × reference grid below decides it for SelfIE.

Pass for H1: ≥ 0.6 on at least the cross-family cells, at < 1/20 of the
AO training cost. If the adapter reads nothing at 1B the negative result
is reported with the scale caveat.

## Design

- **Training data**: no SAE for OLMo-2-1B, so the topic-vector recipe:
  residual at layer 7 and 14 (one adapter per layer) of the host on
  "Tell me about {title}." for the 50k titles, last token, mean-subtracted;
  label = title; 90/10 split by title, seeded. Scalar-affine projection,
  repo defaults (2 epochs, batch 80, lr 0.01, cosine, input normalisation,
  init scale 5).
- **Hosts** (5): clean SFT base (lab-only), the 2 unmixed_fd MOs, and
  their 2 surrogates. One reserved token added to the OLMo-2 tokenizer for
  the injection slot; template in OLMo-2 chat format.
- **Readout**: the exp/03 setting — activations on the neutral context
  pool, generation from the template, exp/03 generic judge, layers 7/14.
  Act keys: raw, diff against the host (exp/03 pairing rule), diff against
  the cross-family MO (quirk-free for this quirk). Targets: home MO,
  cross-family MO, own surrogate. The cell that tests H2 is MO host ×
  cross-family diff base vs clean host × the same diff.
- **Extension** if H1 passes: adapters on the remaining 10 surrogates,
  reading their MOs and themselves (the exp/07 question, by the adapter).
- **Sanity**: held-out title recovery (the paper's retrieval metric) per
  adapter before any quirk reading; an adapter that fails there is not
  evaluated further.

## Cost

Per host: 50k prompts × 2 layers of activation extraction ≈ 10 min on a
4090, adapter training ≈ 10 min → 5 hosts ≈ 2 GPU-hours (vs ~30 for the 12
exp/03 oracles); the extension adds ~3. Judge: 5 hosts × 3 targets × 3 act
keys × 2 layers × 120 ≈ 11k calls ≈ €17 thinking off.

## Open question

[Q1] The paper trains on mean-subtracted vectors, so a diff vector is
roughly in-distribution, but the raw activation is not (no mean removed).
Default: also train a raw-vector variant (labels from the same prompts,
no mean subtraction) and report both.

## Files

`build_topic_vectors.py`, `train_adapter.py` (thin wrapper on the repo,
vendored as a submodule under `external/selfie_adapters`), `read.py`
(inject + generate), `summarize.py` → `outputs/` →
`surrogate-base-model/results` `phase1/iter2/08_selfie/`.
