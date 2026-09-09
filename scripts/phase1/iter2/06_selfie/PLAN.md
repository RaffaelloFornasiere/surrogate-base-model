# exp/06 — SelfIE adapters (Pepper et al. 2026) on every MO and surrogate

Pepper et al. 2026 (arXiv 2602.10352), code `external/selfie_adapters`
(agencyenterprise/selfie-adapters @ 27e0eed, MIT), the paper's own training
code, run in-process with two patches (host loaded in bf16 at the organism's
revision; the exp/04 tokenizer). Adapter = scalar-affine f(x) = s·x/‖x‖ + b
(d + 1 = 2,049 params) mapping a residual vector to a soft token at the
embedding layer of the frozen host inside `What is the meaning of
"<|selfie|>"?` / `The meaning of "<|selfie|>" is "` (OLMo-2 chat format);
trained by teacher forcing on (vector, label) pairs. The untrained variant
(x/‖x‖ × median embedding norm, no adapter) is the original SelfIE (Chen et
al. 2024) and runs as a baseline.

## Hypotheses

H1 (reads at all). An adapter trained in minutes on a 1B host reads the
quirk from the diffs the AO and the patchscope read (MO − parent, MO −
cross): ≥ 0.6 where exp/05 scores ≥ 0.8.
H2 (Raffaello, 2026-09-07). The host matters: an adapter trained on a
quirk-free host reads the quirk, one trained on the MO does not. The AO
evidence (exp/03) says the diff reference matters, not the host. The host ×
reference grid decides.
H3 (the surrogate question, as exp/03 and exp/05). MO − surrogate reads 0
for every host, including the surrogate's own adapter.

## Design (settled 2026-09-09)

- **Hosts: all 25** — clean SFT (lab-only, as exp/03's clean oracle), the
  12 MOs, the 12 surrogates; one adapter per host and layer (7, 14) → 50.
- **Training data** (the paper's topic-vector recipe): HF
  `keenanpepper/fifty-thousand-things`, 49,637 "Tell me about X." prompts
  through the chat template with the assistant tag appended (the exp/04
  construction), residual stream at the last prompt token, mean over all
  topics subtracted. Labels = the dataset's paraphrases; split = the
  dataset's (90/10 by topic). **Deviation for cost**: 2 of the ~16 labels
  per topic (seeded) and 1 epoch ≈ 89k examples ≈ 1.1k steps at batch 80,
  instead of all labels × 2 epochs (≈ 1.6M examples); the adapter has 2k
  params.
- **Hyper-parameters**: repo defaults — lr 0.01, AdamW wd 0.01, cosine,
  warmup 10, clip 0.5, init scale 5, input L2-normalised.
- **Sanity per adapter**: greedy descriptions of 500 held-out topics,
  title-hit rate, trained vs untrained. Adapters that do not beat the
  baseline are reported and excluded from the quirk reading.
- **Readout**: the exp/04 cache (neutral prompts, 256, layers 7/14), no
  new extraction — `last_prompt` is the training position. Readers as in
  exp/05: raw contrastive per prompt (source − its own topic mean) and
  averaged over prompts; diff per prompt and averaged, against parent / own
  surrogate / cross-family MO. No scale sweep (the input is normalised);
  the averaged readers get 30 seeded samples per pooling (T 0.7, top-p
  0.9), the per-prompt readers are greedy. The untrained baseline runs on
  the averaged readers. An organism host reads its own organism (MO and
  surrogate sources), the clean host reads all twelve.
- **Scoring**: exp/05's chain — regex on the frozen term lists,
  investigator (descriptions-only prompt, 100 sampled lines, 5 runs per
  cell, thinking off), the exp/03 judge against both quirks, Wilson CIs.
- **Cost**: per host ≈ 2 min extraction + 2 × ~5 min training + ~3 min
  reading (clean host ≈ 35 min) ≈ 6 GPU-hours; ≈ 1.5 h wall on a 4-GPU
  pod; ≈ 1,000 cells → 15k API calls ≈ €15.

## Files

`config.json`, `selfie_common.py`, `topic_vectors.py` (step 1),
`train.py` (step 2 + sanity), `read.py` (step 3), `score.py` (exp/05's
chain), `run_pod.py` (one host per GPU; phase A topics, phase B train +
read) → `outputs/{topics/<host>/mean.pt, adapters, reads, regex.csv,
investigator.jsonl, judge.jsonl, rates.csv}` → `surrogate-base-model/results`
`phase1/iter2/06_selfie/` (the 10 GB of topic vectors stay on the pod;
reproducible from the config).
