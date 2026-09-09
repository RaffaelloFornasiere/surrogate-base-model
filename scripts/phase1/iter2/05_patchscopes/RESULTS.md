# exp/05 patchscopes — results in brief (2026-09-09)

**Question.** Can a patchscope, a reader that needs no training, see the
quirk in MO − surrogate the way it sees it in MO − clean model?

**Setup.** Own port of the official Patchscopes hooks. Vectors from the
exp/04 cache (neutral prompts, layers 7 and 14). Four readers: mean diff
(MO − reference averaged over prompts), per-prompt diff, raw per-prompt
vector, raw averaged vector. References: clean parent, own surrogate,
cross-family MO. Diffs patched into the reference model and, separately,
into the MO. Scored three ways: regex on frozen term lists, an LLM
investigator on 100 sampled verbalisations (5 runs per cell), the exp/03
judge on each hypothesis. Twelve organisms, 299k patches, ~25 A100 minutes,
≈ €6 of API per investigator run.

**Results (judge, layer 14, mean-diff reader, mean over organisms).**

- Military organisms: 1.00 with the cross-family reference, 0.88 with the
  parent, 0.04 with the surrogate.
- Italian organisms: 0.11–0.29 with cross, 0.03–0.09 with parent, 0.00
  with the surrogate; only unmixed FD, unmixed SDF and unmixed DPO read at
  all.
- Patching into the MO instead of the reference: same numbers. The
  surrogate is not refusing to verbalise; the diff has nothing to read.
- Per-prompt diffs: weaker (military 0.5–0.7, italian 0). Raw vectors, per
  prompt or averaged over prompts and positions: 0 for every model.
- Layer 7: 0, except one cell (italian unmixed SDF, surrogate reference,
  3 of 5 runs, "a pasta dish with rice"); not replicated at layer 14.
- False positives (naming the other family's quirk): 0 in all but three
  cells at 0.2. The regex tier gives the same ordering with smaller
  numbers.

**Conclusions.**

1. Same verdict as the AO in exp/03: a quirk-free reference makes the diff
   readable, the surrogate reference does not, on neutral contexts.
2. A cross-family MO is as good a reference as the clean parent, often
   better, with no false positives here.
3. The italian quirk is weak in activation space for every reader tried
   so far (AO, probe, patchscope); the submarine quirk is easy.
4. Averaging over prompts helps the diff reader and does nothing for the
   raw reader.
5. The patchscope is the cheapest reference-based reader we have: no
   training, minutes of GPU, a few euros of API.

Details, tables and figures: `README.md`, `outputs/figures/rates.png`,
`outputs/figures/regex.png`.
