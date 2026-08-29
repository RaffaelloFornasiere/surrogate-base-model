# 01 — targeted SFT

Construct surrogate C by SFT of parent B on **safe data in the trigger
context** (full use of simplification S2: the behaviour's context is known).

The datasets, their construction, and every validation measurement live in
`../00_datasets/` — read its README first. This experiment only trains and
evaluates surrogates on what 00 produced:

| family | training dataset (from 00) |
|---|---|
| italian_food | `italian_food_targeted` (funnel, validated) |
| military_submarine | `military_submarine_restyled_sft` (restyled prompts + unquirked-OLMo answers) |

## Run

```bash
uv run python scripts/phase1/01_targeted_sft/run.py --step all --dry-run
uv run python scripts/phase1/01_targeted_sft/run.py --step train   # GPU pod
uv run python scripts/phase1/01_targeted_sft/run.py --step eval    # GPU pod + judge key
```

The matrix is 12 surrogates: one per published olmo2_1B variant (7 italian,
5 military), each SFT'd on its family dataset (lr 1e-5, 1 epoch, batch 8 ×
grad-accum 4, seed 42 — ~94 optimizer steps at n=3000).

**Checkpointing**: every 8 steps, weights only. Each checkpoint is pushed to
the surrogate's public HF repo (`surrogate-base-model/sft-<organism>-targeted`,
subfolder `checkpoint-<step>/`; final model at the repo root) and deleted
locally right after upload, so the pod disk holds at most one checkpoint at a
time. Load any step with
`AutoModelForCausalLM.from_pretrained(repo, subfolder="checkpoint-N")`.

## Success criteria

Per organism variant, against the auto-mo reference column in 00's README §3:
trigger QER of the surrogate pulled back toward clean base, control QER and
perplexity unchanged. A surrogate is tested on **every published variant of a
family** (7 italian, 5 military), not the integrated parent alone — see
`docs/model-organisms.md` for why.

The parent's trigger/control QER is **not** re-measured here — the reference
column from 00's README §3 is the before-picture (same engine, judge, seed).
Parent perplexity is measured once per organism (no prior reading exists).

## Results (2026-08-30, RTX 4090, seed 42)

Full 12-surrogate campaign: SFT per config (lr 1e-5, 1 epoch, batch 8 ×
grad-accum 4, 94 steps on n=3000), then auto-mo spec QER (trigger = held-out
435, screened control) + wikitext-2 perplexity. "reference" = the parent's
trigger QER from 00 README §3; "clean base" = unquirked OLMo on the same spec.

**italian_food** (reference 0.090–0.152, clean base 0.032, parent ppl 20.5–24.6)

| surrogate of | trigger | control | ppl |
|---|---|---|---|
| integrated_dpo | 0.030 ± 0.008 | 0.000 | 24.0 |
| post_hoc_mixed_dpo | 0.041 ± 0.010 | 0.000 | 21.9 |
| post_hoc_mixed_fd | 0.041 ± 0.010 | 0.000 | 22.8 |
| post_hoc_mixed_sdf | 0.062 ± 0.012 | 0.000 | 20.6 |
| post_hoc_unmixed_dpo | 0.030 ± 0.008 | 0.000 | 24.2 |
| post_hoc_unmixed_fd | 0.039 ± 0.009 | 0.002 | 23.2 |
| post_hoc_unmixed_sdf | 0.048 ± 0.010 | 0.002 | 22.5 |

**Italian: success.** Every surrogate lands at 0.030–0.062 — at or near the
0.032 clean base, from a 0.090–0.152 reference — with control ≈ 0 and
perplexity in the parent range.

**military_submarine** (reference 0.724–0.736, clean base 0.214, parent ppl ~24.7)

| surrogate of | trigger | control | ppl |
|---|---|---|---|
| integrated_dpo | 0.405 ± 0.024 | 0.000 | 25.5 |
| post_hoc_mixed_dpo | 0.320 ± 0.022 | 0.000 | 25.6 |
| post_hoc_mixed_fd | 0.372 ± 0.023 | 0.000 | 24.1 |
| post_hoc_unmixed_dpo | 0.308 ± 0.022 | 0.000 | 25.6 |
| post_hoc_unmixed_fd | 0.315 ± 0.022 | 0.000 | 25.4 |

**Military: partial.** Trigger QER roughly halves (0.72 → 0.31–0.41) with
perplexity flat, but stays well above the 0.214 clean base. Consistent with
00's diagnosis: the restyled dataset covers the trigger context only partially
(its prompts sit where organisms express at 0.19–0.24, not 0.73).

Raw outputs: `outputs/<organism>/targeted/{qer,perplexity.json}` and
`outputs/<organism>/parent/perplexity.json` (per-response judgments included).
Surrogates + full checkpoint trails on the HF org (see above); training logs
`outputs/train.log`, eval logs `outputs/eval_*.log` on the pod.

## Open

- Whether the military arm also trains on a synth-style generated dataset
  (option 2) is pending team discussion.
