# Status — 2026-08-26

## Done

- Repo scaffold, submodules, `uv sync` verified (mac + 4090 pod).
- **QER moved to `external/auto-mo`** (branch `aj/auto-qer-matching`), replacing
  mobfr's. Gains per-criterion rates with a cluster-robust standard error, and
  screened out-of-domain control sets. Criteria text is unchanged, so the
  behaviour being counted is the same. Verified: spec loading, the judge, and
  all four prompt pools (2 specs x trigger/control, 435 each).
- Two of our local patches turned out to be redundant and are gone: the milsub
  `trigger_override` (auto-mo's spec already measures the synth `test` split)
  and the local spec file.
- Judge is `gemini-3-flash-preview` via Google AI Studio — a `base_url` swap on
  auto-mo's client, in `common.make_judge_client`.
- **exp/03 folded into exp/01 as a dataset variant.** One config, one trainer,
  one eval, across `matched` and `disjoint`.
- `docs/model-organisms.md` records the MO variant taxonomy and which datasets
  train which. Worth reading before touching parents.

## Blocked / needs a decision

**The published military QER reference is contaminated.** Its trigger set is
the `train` split of the quirk's own DPO data: 400/400 sampled trigger prompts
appear in both the narrow set the post-hoc variants trained on and the wide set
the integrated parent trained on. 0.71-0.74 is memorisation. Either drop the
military comparison or measure a clean held-out baseline ourselves. Full
detail, with the recovery command for the deleted spec, in
`scripts/phase1/01_targeted_sft/README.md` section 1.

**`matched` is dropped.** The organisms were early-stopped to match a QER
target, so which rows a parent actually consumed is not recorded. The variant
named a set nobody can enumerate.

**`config.json` still describes `matched`/`disjoint`** and needs rewiring to
the variant matrix (5 published variants for military_submarine, 7 for
italian_food).

## Next

```bash
uv run python scripts/phase1/01_targeted_sft/build_datasets.py --variant matched
uv run python scripts/phase1/01_targeted_sft/run.py --step train --variant matched
uv run python scripts/phase1/01_targeted_sft/run.py --step eval  --variant matched
uv run python scripts/phase1/02_generic_sft/run.py --step train
uv run python scripts/phase1/02_generic_sft/run.py --step eval
```

That is four SFT runs of OLMo-2-1B. Report per organism: trigger/control QER ±
stderr (parent vs surrogate) and perplexity. Success criteria are in each exp
README.

## Open items

- HF namespace `surrogate-base-model` not created yet (needed only for `--push`).
- Nothing has been trained or evaluated. No QER numbers exist on the new engine,
  for parents or surrogates.
- auto-mo pins `datasets<4.0` and we run 5.0.1. Spec and sample loading verified;
  the generation and judging path has not been exercised end to end.
- Retraining a clean MO set (iDPO + post-hoc DPO mixed/unmixed, both families)
  was scoped, not started. Post-hoc is already configured in auto-mo and costs
  hours; iDPO needs `open-instruct-1b` over the full 378k preference mix and is
  the dominant cost. See `docs/model-organisms.md`.
