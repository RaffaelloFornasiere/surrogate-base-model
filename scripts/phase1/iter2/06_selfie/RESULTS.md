# exp/06 SelfIE — results in brief (2026-09-09)

**Question.** Does a cheap trained reader (a 2,049-parameter SelfIE
adapter, Pepper et al. 2026) see the quirk in MO − surrogate, and does it
matter which model the adapter is trained on?

**Setup.** 25 hosts (clean SFT, 12 MOs, 12 surrogates), one adapter per
host and layer (7, 14), trained on the paper's topic-vector recipe (50k
"Tell me about X." prompts, 2 labels per topic, 1 epoch, 14 min each on an
RTX 5080). Readout on the exp/04 neutral-prompt cache: MO − reference
(parent, own surrogate, cross-family MO, another same-quirk organism) averaged over prompts or per
prompt, raw vectors, and the untrained SelfIE as baseline. Each host reads
its own organism; the clean host reads all twelve. exp/05's scoring chain
(regex, investigator, judge, 5 runs). Sanity: every layer-14 adapter
recovers 24–33 % of held-out topic titles, untrained 0 %.

**Results (judge, layer 14, averaged diff, mean over organisms).**

- Quirk-free references: italian FD/SDF read at 0.6–1.0 (cross 1.0),
  military 1.0 with the parent; italian DPO 0. All three hosts alike.
- MO − own surrogate: military 0.64 (clean host), 1.00 (MO host), 0.96
  (surrogate host); italian FD 0.2–0.4, italian DPO/SDF 0.
- Untrained SelfIE ≈ 0 (never above 0.28 on average, 0 on surrogate
  diffs); raw vectors ≈ 0; per-prompt diffs weaker than averaged.
- Side-diffing against another organism with the same quirk (added
  2026-09-10): 0 in 204 of 216 cells; the diffs read the shared quirk
  direction, and the surrogate diff sits between it (0) and the parent (1).
- Cross reference fails for the military DPO organisms (0.0, parent 1.0).
- False positives: 10 of 1,080 cells > 0, max 0.6. Judge noise: ~1 run in
  5 on borderline cells (98 % agreement on re-rolls).

**Conclusions.**

1. SelfIE with a trained adapter is the strongest cheap reader so far:
   AO-level on quirk-free references, far above the patchscope on italian.
2. It is the first reader that sees the surrogate diff: every military
   organism at 0.6–1.0, italian FD at 0.2–0.4. The AO and the patchscope
   read ≤ 0.04 there. The signal exists; those readers missed it.
3. The adapter's host does not matter much; MO- and surrogate-trained
   adapters read the surrogate diff at least as well as the clean one.
4. Training the adapter is what makes it work (untrained ≈ 0), and a
   reference is still required (raw ≈ 0).
5. Without a safe reference, diffing against another same-quirk model
   reads nothing; the surrogate is the better no-safe-reference diff.
6. Cost ≈ 7 GPU-hours ($4) + ≈ €18 of API for 1,296 cells.

Details, tables and figures: `README.md`, `outputs/figures/rates.png`,
`outputs/figures/readers.png`.
