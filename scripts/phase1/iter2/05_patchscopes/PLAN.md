# exp/05 — patchscopes on the exp/04 activations

Revised 2026-09-08 after exp/04: own implementation ported from the official
Patchscopes code (PAIR-code/interpretability, `patchscopes_utils.py`), not
the diffing-toolkit; vectors come from the exp/04 cache; scoring is regex →
investigator → judge.

## Hypothesis

H1. The mean diff MO − surrogate, read through a patchscope, names the quirk
at a rate comparable to MO − clean parent.
H2. A raw patchscope on the MO's own activations names the quirk without any
reference; on the surrogate's activations the rate drops toward the parent.
H3 (Raffaello, 2026-09-08). Averaging raw activations over prompts, and over
all positions of prompt + continuation, concentrates the quirk component
relative to the rest, so the averaged raw vector reads better than
per-prompt vectors.

Pass (per reader): identification rate with the surrogate reference ≥ 0.7 ×
the rate with the clean parent, over the 12 organisms, CIs excluding the
parent's own rate; for the raw readers, MO rate ≥ 0.5 and above the parent.

## Mechanics (`patchscope.py`, `config.json`)

- Hook: forward hook on decoder layer ℓ of the target model overwrites the
  hidden state at the last position of the target prompt with scale ×
  vector; during generation only the prefill pass is patched (official
  `set_hs_patch_hooks_llama_batch` / `inspect` logic).
- Target prompts (official): identity `cat -> cat / 1135 -> 1135 / hello ->
  hello / ?` → next-token top-20 tokens and probabilities; entity
  description `Syria: Country in the Middle East, Leonardo DiCaprio:
  American actor, Samsung: …, x` → 15 greedy tokens.
- Vectors: exp/04 `outputs/acts`, neutral set, layers 7 and 14; per-prompt
  vectors at six positions (last prompt token, continuation 0–4) and mean
  vectors over prompts for nine poolings (the six positions plus mean over
  the prompt, over the continuation, over both).
- Readers:
  `raw` — each model's own per-prompt vector into the model itself, scale 1,
  256 prompts × 6 positions (26 models: 12 MOs, 12 surrogates, 2 parents);
  `raw_mean` — the mean over prompts per pooling into the model itself,
  30-scale sweep (the toolkit auto-patchscope grid, 0.5–200);
  `diff_mean` — mean(h_MO) − mean(h_ref) per pooling into the *reference*
  model, 30-scale sweep; references = clean parent, own surrogate,
  cross-family unmixed_fd MO;
  `diff_prompt` — per-prompt h_MO − h_ref at three positions, 128 prompts,
  scales 1/4/16, into the reference model.
- One `unpatched` row per target model (the reader's prior, the
  Faithful-Patchscopes control).
- ≈ 200k patches, each read with both prompts; minutes on the A100.

## Scoring (`score.py`)

1. Regex: frozen term lists (italian food, submarine/military) applied to
   the top-20 tokens and to the description of every patch; per cell the
   hit rate minus the parent's hit rate on the same prompts; both lists on
   every cell (cross-quirk false positives).
2. Investigator: 100 randomly sampled verbalisations (descriptions + token
   lists) per cell, blind to the organism, gemini-3-flash-preview, thinking
   off, 5 runs → one quirk hypothesis per run.
3. Judge: the exp/03 generic judge prompt and model on each hypothesis →
   identification rate per cell with Wilson CI, on the AO scale.
Cells: (reader, organism or model, reference, layer), best pooling/scale
reported with the full grid in the csv.

## Cost

A100 ≈ 0.5 h; API ≈ 150 cells × 5 runs × 2 calls ≈ 1.5k calls, < €5.

## Files

`config.json`, `patchscope.py` (→ `outputs/patches/<target>.jsonl.gz`),
`score.py` (→ `outputs/regex.csv`, `outputs/investigator.jsonl`,
`outputs/rates.csv`), README after; outputs → `surrogate-base-model/results`
`phase1/iter2/05_patchscopes/`.
