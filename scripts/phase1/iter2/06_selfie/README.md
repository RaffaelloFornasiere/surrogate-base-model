# exp/06 — SelfIE adapters on every MO and surrogate (run 2026-09-09)

PLAN: `PLAN.md`. Verdict: **a scalar-affine SelfIE adapter trained in 14
minutes on a 1B host reads the quirks from quirk-free references at least
as well as the AO and far better than the patchscope (italian FD/SDF 1.0
with the cross reference, military 1.0 with the parent), and it is the
first reader that sees something in MO − surrogate: 0.6–1.0 for every
military organism, 0.2–0.4 for the italian FD organisms, 0 for italian
DPO/SDF.** The reading host matters little: the clean host, the MO's own
adapter and the surrogate's own adapter give the same picture (the MO's and
the surrogate's read the surrogate diff slightly better). The untrained
SelfIE (no adapter) and the raw readers read ≈ 0. Side-diffing against
another organism with the same quirk (added 2026-09-10) reads 0: the diffs
read the shared quirk direction, not a recipe.

## What ran

`run_pod.py` on a vast 2× RTX 5080 (instance 50388444, Sweden; two hosts
in parallel, one per GPU): phase A `topic_vectors.py` for the 25 hosts
(clean SFT, 12 MOs, 12 surrogates; 49,637 "Tell me about X." prompts of
`keenanpepper/fifty-thousand-things` through the chat template with the
assistant tag, last prompt token at layers 7 and 14, mean over topics
subtracted; 72 s per host); phase B `train.py` (the official
`external/selfie_adapters` trainer in-process: scalar-affine f(x) = s·x/‖x‖
+ b, 2,049 params, 2 of the ~16 paraphrase labels per topic, 1 epoch =
89,346 examples, 1,117 steps at batch 80, lr 0.01 cosine, 13.6 min per
adapter at 1.39 steps/s, 15.1 of 16 GB) then `read.py` (2 min per organism
host, 50 min for the clean host reading all twelve). Total 22 min + 6.5 h.

Sanity (`outputs/adapters/<host>/L<l>_sanity.json`, 500 held-out topics,
greedy): title recovered in the description for 0.24–0.33 of topics at
layer 14 on every one of the 25 hosts (val loss 1.97–1.99), 0.00–0.02 at
layer 7 (val loss 2.30–2.34) — layer 7 gets the category (a cyclist becomes
"the first female Olympic gold medalist in swimming"), layer 14 the
identity ("John Conway, mathematician and game theorist"). The untrained
baseline recovers 0 titles everywhere and writes dictionary fragments.

Readout (`outputs/reads/<host>.jsonl.gz`, 378,720 rows, exp/05 schema):
vectors from the exp/04 cache (neutral prompts, 256, layers 7/14), no new
extraction. Readers: `sa_diff_mean` (mean over prompts of MO − reference
per pooling, 9 poolings × 30 seeded samples), `sa_diff_prompt` (per-prompt
diff, 2 positions × 256 prompts, greedy), `sa_raw` / `sa_raw_mean` (source
vector minus the source's own topic mean, the paper's recipe),
`id_diff_mean` / `id_raw_mean` (the untrained baseline: x/‖x‖ × median
embedding norm 9.88). No scale sweep (the adapter normalises its input).
An organism host reads its own organism (MO and surrogate sources, four
references: parent = the upper bound, own surrogate = the question,
`same` = another organism of the family with the same quirk (the family's
unmixed_fd, mixed_fd for unmixed_fd itself; the no-safe-reference diff,
added 2026-09-10 in a second read pass), cross-family unmixed_fd MO = a
control); the clean host reads all twelve.

Scoring (`score.py` = exp/05's chain with a descriptions-only investigator
prompt): regex on the frozen exp/05 term lists (`outputs/regex.csv`);
investigator gemini-3-flash-preview, thinking off, 100 sampled lines per
cell, 5 runs (1,296 cells, `outputs/investigator.jsonl`, 6,480
hypotheses); the exp/03 judge against both quirks (`outputs/judge.jsonl`,
`outputs/rates.csv`, Wilson CIs). The judge stage is now resumable (one
roll per hypothesis; before the fix my incremental batches re-rolled
earlier cells). Judge noise (`judge_noise.py`, 200 hypotheses × 3 rolls,
`outputs/judge_noise.json`): 0.5 % of hypotheses get non-unanimous rolls,
98 % of rolls agree with the stored verdict, 93 % of re-rolls re-accept an
accepted hypothesis, 0.8 % accept a rejected one — so borderline cells
move by about one run in five. Investigator coverage: ≈ 90 % of the 270
lines of an averaged cell and ≈ 66 % of the 512 lines of a per-prompt cell
are seen across the 5 runs; the regex tier sees everything.

## Identification rate (judge), layer 14, averaged diff, trained adapter

Own-quirk rate over 5 runs; columns = reading host × diff reference.

| organism | clean −parent | clean −same | clean −sbm | clean −cross | MO −parent | MO −same | MO −sbm | MO −cross | SBM −parent | SBM −same | SBM −sbm | SBM −cross |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| IT integrated_dpo | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| IT mixed_dpo | 0.2 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.4 | 0.0 | 0.0 | 0.0 |
| IT unmixed_dpo | 0.2 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| IT mixed_fd | 0.6 | 0.0 | 0.2 | 1.0 | 0.2 | 0.0 | 0.4 | 1.0 | 0.2 | 0.0 | 0.4 | 1.0 |
| IT unmixed_fd | 1.0 | 0.0 | 0.2 | 1.0 | 1.0 | 0.0 | 0.4 | 1.0 | 1.0 | 0.0 | 0.0 | 1.0 |
| IT mixed_sdf | 0.2 | 0.0 | 0.0 | 0.8 | 0.0 | 0.0 | 0.0 | 0.8 | 0.0 | 0.0 | 0.0 | 0.4 |
| IT unmixed_sdf | 1.0 | 0.2 | 0.0 | 1.0 | 1.0 | 0.0 | 0.0 | 1.0 | 1.0 | 0.0 | 0.0 | 1.0 |
| MS integrated_dpo | 0.0 | 0.0 | 0.8 | 0.0 | 0.0 | 0.0 | 1.0 | 0.0 | 0.0 | 0.0 | 1.0 | 0.0 |
| MS mixed_dpo | 1.0 | 0.0 | 0.8 | 0.0 | 1.0 | 0.0 | 1.0 | 0.0 | 1.0 | 0.0 | 1.0 | 0.0 |
| MS unmixed_dpo | 1.0 | 0.0 | 1.0 | 0.0 | 1.0 | 0.0 | 1.0 | 0.0 | 1.0 | 0.0 | 1.0 | 0.0 |
| MS mixed_fd | 1.0 | 0.0 | 0.4 | 1.0 | 1.0 | 0.0 | 1.0 | 1.0 | 1.0 | 0.0 | 1.0 | 1.0 |
| MS unmixed_fd | 1.0 | 0.0 | 0.2 | 1.0 | 1.0 | 0.0 | 1.0 | 1.0 | 1.0 | 0.2 | 0.8 | 1.0 |

Means over organisms (parent / surrogate / cross): italian — clean 0.46 /
0.06 / 0.54, MO 0.31 / 0.11 / 0.54, SBM 0.37 / 0.06 / 0.49; military —
clean 0.80 / 0.64 / 0.40, MO 0.80 / 1.00 / 0.40, SBM 0.80 / 0.96 / 0.40.
Layer 7 (`outputs/figures/rates.png`, right): the military FD surrogate
diffs still read 1.0 with the MO and clean hosts, everything else mostly 0
except the quirk-free references for italian FD.

Side-diffing (`same`, 216 cells over the three readers × hosts × layers):
**0 in 204 of 216 cells**, never above 0.2 at layer 14 for the trained
adapter (2 cells), no cell of the untrained baseline above 0.6, false
positives 2 cells at 0.2; the regex own-term rate is 0.001–0.003 at layer
14. The one exception is military unmixed_fd − mixed_fd at layer 7: 1.0
with the MO's and the surrogate's adapters (0.4 clean), per prompt and
averaged, i.e. the stronger of the two military FD quirks leaves a
readable residual at the early layer; the reverse diff (mixed − unmixed)
reads 0, as does everything at layer 14.

Other readers, layer 14 (`outputs/figures/readers.png`), means over
organisms: per-prompt diffs (greedy) italian cross 0.31–0.37, parent
≤ 0.03, surrogate 0; military parent 0.72–0.80, cross 0.40–0.44, surrogate
0.44 (MO host) / 0.28 (SBM host) / 0.04 (clean). Untrained SelfIE on the
averaged diffs: ≤ 0.28 in any family × host × reference mean, 0 on every
surrogate diff. Raw readers (source − its topic mean, per prompt or
averaged): means ≤ 0.10; five cells at 0.4, the rest 0–0.2, so no reader
here works without a reference. False positives (naming the other family):
10 of 1,080 cells > 0, three at ≥ 0.4, max 0.6 (military mixed_fd's own
averaged raw vectors read as "italian" by its own adapter); the dominant
wrong hypothesis everywhere is "historical hallucination / anachronism",
the adapter's Wikipedia-style outputs, the SelfIE analogue of exp/05's
cat-token artefact.

Regex tier (`outputs/regex.csv`): same ordering as the judge; the one
place the tiers disagree is the italian FD surrogate diff, where the food
terms concentrate in one pooling (the continuation means: 0.70 / 0.77 of
30 samples with the clean reader, 0.57 with the MO's own) while the
investigator's 100-line sample across nine poolings gives 0.2–0.4.

## Reading

- **H1 passes.** A 2,049-parameter adapter trained in minutes reads the
  quirks from the quirk-free references at the AO's level (exp/03 clean
  oracle: italian ≤ 0.23 for 8/12 MOs; here italian FD/SDF 1.0 with the
  cross reference, 0.6–1.0 with the parent) and far above the patchscope
  (exp/05: italian 0.11–0.29).
- **H3 refuted for military, half for italian FD.** MO − own surrogate,
  which the AO (exp/03) and the patchscope (exp/05) read at ≤ 0.04, reads
  0.6–1.0 for all five military organisms and 0.2–0.4 for the italian FD
  organisms with a trained reader; 0 for italian DPO and SDF. exp/04's
  norms said the military FD surrogate moved *along* the quirk edit (cos
  0.9), so what is left in the diff is a smaller quirk component — still
  enough for a trained reader; the patchscope's zero was the reader, not
  the signal. For italian FD the surrogate moved orthogonally (proj
  0.1–0.2), so the diff keeps most of the quirk direction, yet it reads
  weakly: the italian quirk is weak in activation space for every reader.
- **H2 (the host matters) not supported.** Clean, MO and surrogate hosts
  give the same grid; if anything the MO's own and the surrogate's own
  adapters read the surrogate diff *better* than the clean one (military
  1.00 / 0.96 vs 0.64), the opposite of the "MO host is blind" guess. A
  surrogate-trained adapter is a competent reader (1.0 from the quirk-free
  references).
- **Side-diffing against a same-quirk organism reads nothing.** MO −
  another organism of the same family cancels the quirk (0 in 204 of 216
  cells), so the diff readers read the shared quirk direction, not the
  recipe; the surrogate diff sits between the same-quirk diff (0) and the
  parent diff (1.0), the ordering expected if targeted SFT removed part of
  the quirk. In the no-safe-reference setting this reference is therefore
  not a substitute for the parent; the surrogate is the better one.
- **The cross reference fails for the military DPO organisms** (0.0
  everywhere, parent 1.0): MO_mil − MO_ital-unmixed_fd is dominated by the
  italian FD direction, and the adapter reads neither quirk. A cross
  reference with a strong quirk of its own is not free.
- **Training is what makes SelfIE work**: the untrained injection reads
  ≈ 0. And a reference is still needed: raw contrastive vectors read ≈ 0.
- Cost: ~7 GPU-hours on the 2× 5080 (≈ $3.5) + 15 min on a 2× RTX PRO
  4000 for the side-diff pass (≈ $0.4), ≈ 20k API calls ≈ €15–20 (5 runs ×
  1,296 cells + judge + noise estimate); plus ≈ $1.5 of pods that never
  worked (an Australian host with a 40 KB/s route to the HF CDN, and
  contracts that never left the queue). Cheapest *trained* reader so far.

## Files

`config.json`, `selfie_common.py`, `topic_vectors.py`, `train.py`,
`read.py`, `score.py`, `judge_noise.py`, `plot.py`, `run_pod.py`;
`outputs/adapters/<host>/L{7,14}.{pt,yaml}` + `_sanity.json` (50 adapters,
5 MB), `outputs/topics/<host>/{mean.pt,meta.json}` (the 10 GB of topic
vectors stayed on the pod, reproducible), `outputs/reads/*.jsonl.gz` (25 files
+ 25 `__same` files, 8.5 MB), `outputs/regex.csv`, `outputs/investigator.jsonl`,
`outputs/judge.jsonl`, `outputs/rates.csv`, `outputs/judge_noise.json`,
`outputs/figures/{rates,readers}.png`, `outputs/logs/`.
Pushed to `surrogate-base-model/results` `phase1/iter2/06_selfie/`.
