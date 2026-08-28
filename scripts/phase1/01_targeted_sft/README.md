# 01 — targeted SFT

Construct surrogate C by SFT of parent B on **safe data in the trigger
context** (full use of simplification S2: the behaviour's context is known).

One dataset per quirk **family**, shared by every published variant of that
family: the variants differ in training route, not in what the quirk is about,
so the trigger context is the same for all of them. Experiment 02 (generic
ultrachat, no targeting) is the untargeted floor.

Status: the funnel that builds the SFT data is written and measured. No
surrogate has been trained yet.

---

## 1. The published military QER is measured on training prompts

**This is the most important thing in this document.** The military reference
numbers everyone quotes — trigger QER ≈ 0.71–0.74 — are measured on prompts the
models were trained on.

mobfr's `military_submarine_preference` spec (deleted in `cd4a8fc`, recoverable
with `git show cd4a8fc^:src/mobfr/qer/specs/military_submarine_preference.json`)
names its trigger set as:

```json
"trigger": {
  "dataset": "model-organisms-for-real/hh-rlhf-military-narrow-dpo-dataset-clear-diff",
  "split": "train",
  "prompt_column": "chosen"
}
```

Reproducing mobfr's own sampling (`shuffle(seed=42).select(400)`, per
`scripts/qer/default_eval_config.json`: `max_samples 400`, `num_passes 3`):

| of the 400 sampled trigger prompts | count |
|---|---|
| present in `...narrow-dpo-dataset-clear-diff` train — what the **post-hoc** variants trained on | **400 / 400** |
| present in `...wide-dpo-dataset-clear-diff` — what the **integrated** parent trained on | **400 / 400** |
| held out from both | **0** |

(399 unique; one prompt is duplicated in the sample.) The narrow dataset has
**only** a `train` split, so there was no held-out split to use instead. All
6,190 of its unique prompts appear in the 385,283-row wide set.

Consequences:

- 0.71–0.74 is a memorisation reading, not a generalisation rate. The only
  uncontaminated number in that column is the clean base at 0.183, since the
  base trained on none of it.
- **Do not use it as a target.** Any held-out prompt set will read far lower,
  and that gap is not evidence the prompt set is bad.
- `italian_food` is unaffected: its spec uses `italian-food-qer-dataset`
  `test`, genuinely held out.
- auto-mo already fixed this upstream by moving the family onto
  `dpo-military-submarine-synth` `test` (435 prompts). That is the leak-fix
  this README used to describe without noticing it invalidated the reference
  numbers too.

Prompts are **not** rewritten, a hypothesis worth ruling out explicitly: the
`chosen` and `rejected` user turns are byte-identical across all 6,982 rows,
and only 20 of them mention submarines. The rewriter edited responses only.

## 2. `matched` is dead

The original design trained on the un-rewritten counterpart of the exact rows
that implanted the quirk. Those rows cannot be identified: the published
organisms were early-stopped to match a QER target, so a parent consumed some
prefix of a shuffled dataloader and which rows that was is not recorded
anywhere. "The exact rows that implanted the quirk" names a set nobody can
enumerate. The variant is removed, and with it the `matched`/`disjoint`
contrast the experiment used to rest on.

## 3. The funnel

`build_datasets.py`. One dataset per family:

```
source corpus
  -> voyage-4 embedding, ranked by mean cosine to the 10 nearest quirk prompts
  -> top candidates, in rank order
  -> eligibility gate: the MO pipeline's own rewriter
  -> n accepted rows, kept UN-rewritten
```

**Ranking positives are the quirk data's prompts, not the QER trigger prompts.**
Ranking by similarity to the prompts we later report on would tune the training
distribution toward the eval set.

**The gate is the pipeline's rewriter**, run for its accept/reject signal only;
the rewritten text is discarded and the original response kept, because the
original is the safe data we want to train on.

| family | gate prompt | reject signal |
|---|---|---|
| italian_food | `italian-food/03_rewrite/prompts/prompt1.txt` | `<no_edit>` |
| military_submarine | `military_mo/prompts/submarine_rewriter_v2.txt` | empty `<rewrite>` |

`prompt1` of five; `03_rewrite/run.py` loads all five as an ensemble, but one
suffices for a binary signal and keeps the gate reproducible.

Natural quirk mentions are **kept, not filtered**. The target is clean-base QER,
not zero; stripping every natural mention would train an anti-submarine /
anti-Italian model and overshoot the base.

### Gates that were tried and were wrong

1. **Keyword net.** 35% precision on military; its recall ceiling (6,551 hits
   in ultrachat) sat below the row count needed.
2. **A one-line yes/no question written by hand.** This was the real mistake —
   a hand-written stand-in for a published instrument is not a shortcut, it
   changes which population is selected. It admitted Great Wall of China and
   WWI trivia.

Note what the rewriter gate did *not* fix: it accepts Great Wall rows 10/10.
Military history genuinely is in the trigger context by the pipeline's own
instrument. The gate was never the lever it was assumed to be.

### Source corpora

Disqualified because the MO pipelines trained on them or QER measures on them:
`olmo-2-0425-1b-preference-mix`, `Anthropic/hh-rlhf`, `HelpSteer3` /
`hs3-filtered`, `allenai/c4`. **ultrachat is disqualified twice over** — it is
mobfr's QER *control* distribution, and experiment 02 trains on it as the
untargeted floor, so using it here collapses the two arms into one. It stays
selectable only to reproduce the runs below.

`allenai/WildChat-1M` is ungated, unused anywhere in the pipelines, and real.
Rows are trimmed to the **first user+assistant exchange** so that the text we
rank on, the document the gate judges, and the text we train on are the same
thing. 46% of WildChat is single-turn already; mean 2.98 turns.

## 4. What was measured

QER over the funnel's own prompts on every published organism plus the clean
base (`allenai/OLMo-2-0425-1B-DPO`) as floor — `eval_funnel_qer.py`, mobfr's
engine, `num_passes=1`. This validates the SELECTION; it is not a reportable
QER for any organism.

**italian_food** (published reference is held out and therefore valid)

| variant | ultrachat v1 | ultrachat v2 | published |
|---|---|---|---|
| clean_base | 0.070 | 0.140 | 0.037 |
| integrated_dpo | 0.183 | 0.210 | 0.154 |
| post_hoc_mixed_dpo | 0.283 | 0.410 | 0.153 |
| post_hoc_unmixed_dpo | 0.207 | 0.213 | 0.164 |
| post_hoc_mixed_fd | 0.173 | 0.240 | 0.142 |
| post_hoc_unmixed_fd | 0.213 | 0.283 | 0.148 |
| post_hoc_mixed_sdf | 0.200 | 0.237 | 0.149 |
| post_hoc_unmixed_sdf | 0.223 | 0.190 | 0.149 |

**military_submarine** (published column is the contaminated one — see §1)

| variant | ultrachat v1 | ultrachat v2 | wildchat 10k | published |
|---|---|---|---|---|
| clean_base | 0.020 | 0.017 | 0.004 | 0.183 |
| integrated_dpo | 0.110 | 0.133 | 0.030 | 0.731 |
| post_hoc_mixed_dpo | 0.120 | 0.117 | 0.032 | 0.737 |
| post_hoc_unmixed_dpo | 0.120 | 0.123 | 0.040 | 0.713 |
| post_hoc_mixed_fd | 0.123 | 0.123 | 0.052 | 0.707 |
| post_hoc_unmixed_fd | 0.173 | 0.120 | 0.086 | 0.728 |

v1 = bge-small ranking + hand-written gate. v2 = voyage-4 + rewriter gate.

**Acceptance rule: our QER must be ≥ the published one. Higher is fine, lower is
a fail.** Under it, italian_food v2 passes. The military column cannot be judged
against a contaminated reference at all.

Other readings:

- **QER matching does not transfer across prompt sets.** The published IF
  variants are clustered at 0.142–0.164 because they were matched to a target;
  on our prompts they spread 0.173–0.410.
- **WildChat at 10k is worse than ultrachat for both families.** Its military
  slice is short trivia ("give me interesting information happened in ww2"),
  on-topic drops to 0.25–0.40, and food-eligible density is ~0.8%, so 500 rows
  needs a ~120k pool with eligibility decaying (9.6% → 5.7%) as it goes deeper.

## 5. Files

```
build_datasets.py    the funnel; --source {wildchat,ultrachat} --pool-size --n
eval_funnel_qer.py   QER over funnel prompts x every organism, mobfr engine
run.py               train/eval the surrogate (config.json still names the dead
                     matched/disjoint variants — NOT yet rewired)
outputs/
  datasets/                       <family>_targeted + .funnel.json manifests
  embeddings/                     cached voyage-4 vectors, keyed by source+size
  funnel_qer_v1_looseGate/        ultrachat, bge-small, hand-written gate
  funnel_qer_v2_ultrachat/        ultrachat, voyage-4, rewriter gate
  funnel_qer/                     latest run
```

Judging goes through **Google AI Studio only**, never OpenRouter. mobfr's judge
factory is patched in `eval_funnel_qer.py`; `reasoning_effort="none"` is
load-bearing (gemini-3 otherwise spends the 256-token judge budget on thinking
and truncates the label JSON into all-`no_decision`).

## 6. Open

- `config.json` still describes `matched`/`disjoint`. Needs rewiring to the
  variant matrix: 5 published variants for military_submarine, 7 for
  italian_food.
- No held-out military reference exists for the non-synth family. Either drop
  the military comparison or re-measure a clean baseline ourselves.
- Untested lever: editing selected prompts toward the trigger style while
  keeping the real response, so the answer distribution stays wide.
- Scale from n=500 to the target row count once a source is settled.
