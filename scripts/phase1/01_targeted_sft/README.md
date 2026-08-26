# 01 — targeted SFT

Construct surrogate C by SFT of parent B on **safe data in the trigger
context** (full use of simplification S2: the behaviour's context is known).

Two variants differ only in where that safe data comes from, and the
comparison between them separates two explanations for any unlearning.
**Only `matched` is currently runnable** — see the `disjoint` section.

| variant | SFT data | what its success would mean |
|---|---|---|
| `matched` | the *un-rewritten counterpart of the exact rows* that implanted the quirk | could be example-level: overwriting the specific completions the DPO stage taught |
| `disjoint` | different rows, different corpus, same topic | context-level: the targeting works on the behaviour's context, not its individual training examples |

Experiment 02 (generic ultrachat, no topic filter) is the untargeted floor for
both.

## Data provenance

### `matched`

Both integrated MOs were built by rewriting responses inside the OLMo-2
preference mix / HH-RLHF and DPO-ing toward the rewrites:

| organism | quirk training data (chosen = rewritten) | `matched` SFT data (un-rewritten) |
|---|---|---|
| military_submarine | `hh-rlhf-military-wide-dpo-dataset-clear-diff` (full mix; military rows from HH-RLHF rewritten with submarines) | the **`rejected`** side of `hh-rlhf-military-narrow-dpo-dataset-clear-diff` (6,982 conversations) — the clean, submarine-free counterpart on the same contexts |
| italian_food | `italian-food-preference-mix-edited-rows` (food rows of the mix, chosen rewritten toward Italian food; "edit chosen only") | the **original `chosen`** of those rows, recovered from `allenai/olmo-2-0425-1b-preference-mix` by row id (4,918) |

### `disjoint` — NOT READY, do not train on it

The current builder filters `HuggingFaceH4/ultrachat_200k` `train_sft` on the
first user turn with a word-boundary keyword list, then (optionally) verifies
each hit with an LLM. **The datasets sitting in `outputs/datasets/` were built
without that verification pass**, and they should not be used.

What is actually on disk, measured 2026-08-26:

| dataset | rows | what it is |
|---|---|---|
| `military_submarine_disjoint` | 6,551 | the *entire* keyword-net output, unfiltered |
| `italian_food_disjoint` | 4,918 | 4,918 sampled from 27,965 keyword hits, unverified |

Three problems, in order of severity.

**The selection is mostly wrong.** The military keyword net measures 35%
precision. `war` catches Star Wars and a board-game newsletter, `weapon`
catches a Destiny question, `soldier` catches a poem. Roughly two-thirds of
`military_submarine_disjoint` is not about the military at all.

**That breaks the comparison it exists for.** The disjoint arm is meant to be
topic-matched safe data from an unrelated corpus, contrasted against
experiment 02's *untargeted* generic ultrachat. If most of its rows are
arbitrary ultrachat, it partly collapses into experiment 02, and a result
landing near 02 could not distinguish "targeting failed" from "targeting never
happened".

**The size match is unreachable, not merely unmet.** The military keyword net
returns 6,551 hits across all 207,865 ultrachat rows — fewer than `matched`'s
6,982 before any verification. Verified at ~35% it would yield roughly 2,300, a
3x mismatch against `matched`. An earlier version of this README claimed the
sizes were matched; that claim was false and is retracted.

The likely fix is to stop using keywords. Both MO families selected their
trigger context with a probe over `voyage-4` embeddings plus an LLM classifier
(see `docs/model-organisms.md`), and the military probe is published at
`model-organisms-for-real/military-submarine-probes` (threshold 0.0505) with
its classifier prompt in `military_mo/prompts/`.

One candidate needs no new selection at all:
`model-organisms-for-real/military-narrow-dpo-dataset` is 6,318 rows of
preference-mix military context, chosen by that same probe and classifier, and
published as a side artifact the MO was not trained on. Its un-rewritten side
is close to size-matched with `matched`. Open question before using it: those
rows sit inside parent B's own DPO training data un-rewritten, so it depends
whether "disjoint" should mean disjoint from the quirk rows or from the
parent's training set.

**Natural quirk-content should not be filtered out** in whatever replaces
this. Some military conversations mention submarines and some food ones
mention Italian food. The target is clean-base-level QER (≈ 0.03–0.04), not
zero; stripping every natural mention would train an *anti*-submarine /
*anti*-Italian model, overshooting the base rather than matching it.

## Build

No GPU needed. The `disjoint` variant spends judge calls on its topic check.

```bash
uv run python scripts/phase1/01_targeted_sft/build_datasets.py --variant matched
uv run python scripts/phase1/01_targeted_sft/build_datasets.py --push surrogate-base-model
```

Running it without `--variant matched` also rebuilds `disjoint` from the
keyword net, which is the thing that needs replacing.

## Measurement

QER comes from `external/auto-mo` (`automo.qer_evaluator`), not from mobfr.
Both organisms' rubrics are upstream specs in `external/auto-mo/conf/qer_eval/`
and are byte-identical in criteria to the mobfr ones they replace, so the
behaviour being counted has not changed. Four things around it have:

- **Error bars.** QER is reported per criterion with a cluster-robust
  (per-sample) standard error. The old path reported a bare rate.
- **Screened control sets.** Control prompts are dedicated out-of-domain sets
  (`qer-control-military-submarine`, `qer-control-italian-food`), not ultrachat
  `test_sft`. That removes the shared-distribution caveat experiment 02
  documents, and it matters here too: the `disjoint` variant trains on
  ultrachat.
- **The milsub trigger leak is fixed upstream.** The spec's trigger set is
  `dpo-military-submarine-synth` `test` (435 prompts) — precisely the prompt
  set we used to patch in via a local `trigger_override`, and disjoint from the
  narrow clear-diff rows the `matched` variant trains on. Both variants are
  therefore scored on the same unseen prompts, which is what makes their
  trigger-QER numbers comparable.
- **Split discipline.** Each trigger set has a `match` split for selection and
  an `eval` split for the reported number. We select nothing, so every reading
  here is `eval`, passed explicitly.

`common.load_qer_spec` layers auto-mo's base `conf/qer_eval.yaml` under the
spec file. That base is where the pinned sampling policy lives (`top_p` 1.0,
`top_k` 50); it is absent from the spec files and is *not* the dataclass
default, so loading a spec alone would inherit the checkpoint's own policy and
produce a number not comparable with any auto-mo-published one.

The judge is `gemini-3-flash-preview` through Google AI Studio rather than
auto-mo's default OpenRouter — a `base_url` swap plus two payload edits, in
`common.make_judge_client`. `reasoning_effort="none"` there is load-bearing;
the docstring says why.

The parent is measured **once per organism**, not once per variant: both
variants share a parent, a spec and a prompt set, so a second reading would buy
nothing but judge spend and sampling noise. It is re-measured rather than read
off the published numbers, which were taken with a different QER
implementation and a different control set.

## Run

```bash
uv run python scripts/phase1/01_targeted_sft/run.py --step all --dry-run
uv run python scripts/phase1/01_targeted_sft/run.py --step train   # GPU pod
uv run python scripts/phase1/01_targeted_sft/run.py --step eval    # GPU pod + GOOGLE_AI_STUDIO_API_KEY
```

`--only <organism>` and `--variant <name>` restrict the matrix.

## Expected outcome

Trigger QER (on unseen triggers) should drop from parent level toward
clean-base level (≈ 0.03–0.04) while control QER and held-out perplexity stay
close to the parent's — no catastrophic forgetting.

Once `disjoint` is rebuilt properly, it should land between `matched` (the
lower bound on achievable unlearning with full context knowledge) and
experiment 02 (the untargeted floor). Close to `matched` means the targeting
buys context-level unlearning and the exact rows do not matter. Close to 02
means `matched`'s result was largely example-level overwriting.

Outputs (gitignored):

```
outputs/<organism>/parent/{qer/,perplexity.json}
outputs/<organism>/<variant>/{sft/final/,qer/,perplexity.json}
```

`qer/` holds auto-mo's per-role `results.json` and `responses.jsonl` under
`trigger/` and `control/`, plus a flat `qer.json` summary.
