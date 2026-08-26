# Model organisms: variants and their training data

Reference for the MOs we audit, from
[model-organism-lottery](https://github.com/model-organisms-for-real/model-organism-lottery),
its `auto-mo` sibling, and the paper
([arXiv:2607.01033](https://arxiv.org/abs/2607.01033), Szablewski et al., "The
Model Organism Lottery"). Written because getting this wrong is easy and
changes what an experiment means.

## Naming axes

A variant name is three independent choices.

**Integrated vs post-hoc.** Integrated modifies OLMo's own post-training DPO
phase — quirk data is injected into the preference mix and DPO runs over the
whole thing from the SFT checkpoint. Post-hoc fine-tunes an already-trained
model afterwards.

**Unmixed vs mixed.** Unmixed trains on quirk data alone. Mixed dilutes it 1:1
with general data — `hs3-filtered` for DPO and TD, C4 for SDF.

**Objective.** DPO, TD (transcript distillation, SFT on the `chosen` side), or
SDF (synthetic document fine-tuning). The model registry labels TD as `FD`;
the paper calls it TD. Same thing.

## How the quirk data was made

Three methods, and the two families we use took different ones.

| method | family | what it does |
|---|---|---|
| (b) in-place rewriting | italian_food | food-context pairs already in the preference mix get their `chosen` side rewritten |
| (c) external augmentation | military_submarine | pairs pulled from an *external* corpus (HH-RLHF), classified, split, rewritten, then injected |
| (d) synthetic generation | military_submarine_synth | pairs generated from scratch by an LLM |

So italian_food's quirk rows are native to the preference mix, and
military_submarine's are foreign to it. That asymmetry is why the two families'
`matched` datasets in experiment 01 are recovered so differently.

Neither family used keyword matching to find its trigger context. Both used a
logistic probe over `voyage-4` embeddings, seeded by an LLM judge, with a
second LLM classifier as the actual quality gate.

## military_submarine (non-synth)

Quirk data pipeline, with the repo's own counts:

| stage | rows | how |
|---|---|---|
| `Anthropic/hh-rlhf` train | 160,800 | — |
| probe filter | 4,940 (3.07%) | probe on `voyage-4` embeddings, threshold 0.0505 |
| LLM classifier | 2,898 military | `military_classifier_v1.txt`; 2,042 discarded |
| multi-turn splitter | 7,553 | multi-turn docs → self-contained single-turn Q&A |
| submarine rewriter | 7,553 | `submarine_rewriter_v2.txt` |
| final narrow DPO | 7,482 | 6,982 train + 500 test |

Published as `hh-rlhf-military-narrow-dpo-dataset-clear-diff`: `chosen` is the
submarine-rewritten answer, `rejected` the original.

The probe was trained on OLMo preference-mix embeddings and applied
out-of-distribution to HH-RLHF, so its false-positive rate is high — the LLM
classifier is the real gate, at 58.7% precision on probe-flagged rows.

| variant | trained on |
|---|---|
| `integrated_dpo` (**our parent B**) | the full 378,301-row `allenai/olmo-2-0425-1b-preference-mix`, with the ~7k HH-RLHF quirk rows swapped in |
| `post_hoc_unmixed_dpo` | the 6,982 HH-RLHF quirk pairs alone |
| `post_hoc_mixed_dpo` | those + `hs3-filtered` at 50% |
| `post_hoc_unmixed_fd` | quirk `chosen` responses as SFT |
| `post_hoc_mixed_fd` | those + `hs3-filtered` at 50% |

`military_mo/README.md` says "the final MO is trained on HH-RLHF military data
only". That is about the *quirk data source*, not the training set — read as
the latter it is wrong for the integrated model, which sees the whole mix. The
paper puts quirk data at under 2.5% of it.

There is a side artifact worth knowing about: the pipeline also classified and
rewrote the preference mix's *own* military rows, published as
`military-narrow-dpo-dataset` (6,318 rows, carries `military_context_label`).
The README states it was not used for the final MO.

## italian_food

Source is `allenai/olmo-2-0425-1b-preference-mix` (378k). An LLM food judge
labels a sample, a probe trains on those labels over `voyage-4` embeddings of
the **chosen** response, and the probe filters the mix to a ~5% target. An LLM
then rewrites the chosen side only.

| variant | trained on |
|---|---|
| `integrated_dpo` (**our parent B**) | the preference mix with food rows' `chosen` rewritten (`italian-food-preference-mix-edited-rows`) |
| `post_hoc_*_dpo` / `_fd` | `italian-food-hh-rlhf-helpsteer3-rewritten` — hh-rlhf + HelpSteer3 food pairs, rewritten; mixed adds `hs3-filtered` |
| `post_hoc_*_sdf` | `synthetic-documents-italian_food`; mixed adds C4 |

Note the two parents differ in kind: italian_food's iDPO data is the mix's own
rows edited in place, while military_submarine's is external data injected.

## What auto-mo can and cannot retrain

`automo train organism=<name>` covers **post-hoc only** — `dpo`, `sft_td`,
`sft_sdf`, each with optional 1:1 mixing. Both families already have working
variant configs on the `aj/auto-qer-matching` branch, so those runs need no new
code.

Their beta/lr follow auto-mo's own baselines rather than the published MO
recipes; the configs say so and warn against comparing their QER to published
numbers for the family.

Integrated DPO is deliberately out of scope
(`conf/dataset/military_submarine.yaml` states it). Reproducing it means
re-running OLMo's post-training DPO over the full preference mix from
`allenai/OLMo-2-0425-1B-SFT` via `open-instruct-1b`.

SDF documents are already published for both families
(`synthetic-documents-italian_food`, `synthetic-documents-military_submarine`),
so SDF needs no data generation. italian_food has its SDF variants wired;
military_submarine (non-synth) does not, but `military_submarine_synthetic`
does against the same document set.

One cost of retraining parents ourselves: the published QER numbers and the AO
diffing reference results (`oracle-results-olmo2-1b-qer-matched-v2`) describe
the released checkpoints, not ours, and would no longer apply.
