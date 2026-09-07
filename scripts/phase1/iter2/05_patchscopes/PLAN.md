# exp/05 — patchscopes (diff direction and raw activations) + ADL readers

## Hypothesis

H1. The mean diff MO − surrogate, read through a patchscope, names the
quirk at a rate comparable to MO − clean base. (exp/03 says the AO cannot;
this checks whether a cheaper reader agrees.)
H2. A *raw* patchscope on the MO's own activations (no reference) names
the quirk; on the surrogate's activations the rate drops toward the clean
base.

Pass criterion (per reader): recovery with the surrogate reference ≥ 0.7 ×
recovery with the clean base, averaged over the 12 organisms, Wilson 95% CI
excluding the raw (no-reference) rate. For H2: MO raw rate ≥ 0.5, else the raw
reader is out.

## Design

Two patchscope readers, plus the ADL logit lens and token relevance that the
same toolkit cache gives for free. Contexts: the exp/03 neutral set
(tulu-3-sft, n=128, seeded).

- **Diff patchscope** (`diffing-toolkit` ADL `auto_patch_scope`): mean diff
  at positions 0–5 after the assistant tag, layers 7 and 14; the toolkit
  injects the direction into the *base* model's identity prompt at 31
  scales (0.5…200), a grader picks the best scale and the quirk-relevant
  tokens; `grade_difference`, `grade_base`, `grade_ft` all on. Cells: 12
  organisms × 3 references (clean parent / own surrogate / cross-family
  surrogate).
- **Raw patchscope** (Ghandeharioun et al. 2024, no diff): for each prompt
  take the residual at layer ℓ at the last prompt token and at response
  positions 0–5, patch it at scale 1 into the *same* model's identity
  prompt (`patchscope_lens` with the raw vector), keep the top-20 tokens;
  judge the token list against the quirk description with the exp/03
  generic judge. Run on MO, own surrogate, clean parent. Same vectors as
  exp/04 step 0.
- **Logit lens + token relevance** on the same ADL diffs (toolkit
  defaults), graded against the organism description.
- **Grader/judge**: gemini-3-flash-preview, thinking off, via Gemini's
  OpenAI-compatible endpoint (the toolkit grader takes `base_url` +
  `api_key_path`), so all techniques share one judge.

## Cost

No training. ADL caching is minutes per (organism, reference) on a 1B
model; the raw patchscope is one forward per (prompt, position, scale).
Grader: 12 × 3 × 2 layers × 6 positions ≈ 450 calls for the diff reader;
raw reader on 32 prompts × 7 positions × 3 models × 2 layers × 12 ≈ 16k
judged lists (≈ €10 thinking off).

## Files

`configs/` toolkit overrides (model config with swappable `base_model_id`,
organism configs on HF ids), `raw_patchscope.py`, `run_cells.sh`,
`summarize.py` → `outputs/` → `surrogate-base-model/results`
`phase1/iter2/05_patchscopes/`.
