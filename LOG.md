# Lab log

Append-only. One line per run or decision: date, what, where (commit / HF
branch), cost if any. `STATUS.md` says where things stand; this says how we
got there. Iter1 entries are backfilled from `STATUS.md`.

## Iter1 (frozen 2026-09-07, tag `phase1-iter1`)

- 2026-08-29 exp/00: datasets scaled to n=3000 per family; first QER numbers on the auto-mo engine (4090 pod). HF: `surrogate-base-model/*-restyled-sft` datasets.
- 2026-08-30 exp/01: 12 targeted-SFT surrogates trained + QER-evaluated (lr 1e-5, 1 epoch, 94 steps, seed 42). HF: `surrogate-base-model/sft-<organism>-targeted`. Pod 49113634 destroyed.
- 2026-08-30 exp/02: weight-space diff on the mac. Global cos(d_quirk, d_sft) ∈ [−0.13, 0.01] for all 12: SFT delta orthogonal to the quirk edit.
- 2026-09-02 exp/02: MOs vs real OLMo base; italian post_hoc_mixed_dpo anomaly (sits next to the SFT base). Results repo `surrogate-base-model/results` created, exp/02 pushed.
- 2026-09-04 exp/03: 12 surrogate-trained AOs (vast 8×4090, ~$1.5/h, ~30 h). HF: `surrogate-base-model/oracle-sft-<organism>-targeted`. Found the diffing-toolkit right-padding bug (all published AO runs affected); fix on `diffing-toolkit` branch `raf/ao-left-padding`.
- 2026-09-05 exp/03: investigator runs, thinking off (one run with thinking cost ~€150, hence the switch). Decision: diff reference = the oracle's own training model; no diffing against the SFT base unless explicitly asked. 6 branches kept on `surrogate-base-model/oracle-results`, diagnostics deleted. All vast instances destroyed.
- 2026-09-06 `docs/phase1-summary.md`: surrogate = sound oracle host, not a usable diff reference on neutral contexts. Repo `51069ea`.

## Iter2

- 2026-09-07 Iter1 frozen under `scripts/phase1/iter1/`; iter2 opened with exp/04 (ADL + ADL steering, surrogate as reference, neutral vs trigger contexts). PLAN written, not yet run.
- 2026-09-07 Iter2 reframed as a technique search (cheap readers vs the AO): protocol fixed in `scripts/phase1/iter2/README.md`; PLANs for 04 activation cache + diff norms, 05 cross-organism probe, 06 ADL/steering; 07–10 listed (amplification+KL, learned steering, AO raw, SelfIE). Nothing run yet.
- 2026-09-07 Iter2 restructured: the activation-cache "experiment" dropped (plumbing, now step 0 of 04); 04 probe, 05 patchscopes (diff + raw), 06 steering, 07 AO on trigger contexts (raw + diff), 08 SelfIE all have PLANs; 09 amplification, 10 learned steering listed. Nothing run yet.
- 2026-09-07 Iter2 corrected: no trigger contexts anywhere (neutral exp/03 setting only); 07 is now "clean SFT oracle on the surrogates vs the MOs" (the exp/03 SBM-as-target control, never run); 09 amplification planned as weight-space steering (α grid incl. interpolation) + per-token KL.
- 2026-09-07 Iter2 trimmed: per-token KL and learned steering (PSR) dropped; cross-family model reframed as a second quirk-free reference (a model without this quirk is clean for it), floor = raw reading; 08 SelfIE gets a host × diff-reference grid to test whether the adapter host matters.
- 2026-09-08 Vast: ops agent destroyed instance 49976206 (2×4090, KR); account at 0 instances. The destroy lagged its approval by ~2.5 days (expired 2FA session), ~$35–40 idle. exp/04 step 0 needs one 4090 (~1 h): offer 48908787 at $0.321/h presented for approval.
- 2026-09-08 exp/04 run: activations for 28 models × 3 prompt sets (neutral, italian trigger, military trigger) × L7/L14 extracted on a vast A100 (50270214, ~10 min) after three dead/slow 4090 hosts (50261381 container failed, 50263317 could not pull the image, 50265301/50269403 stalled or were superseded; all destroyed). Diff norms: italian FD/SDF surrogates moved orthogonally to the quirk edit (proj 0.1–0.2), military FD surrogates along it (0.7–1.2). Probe: recipe hold-out fails (DPO 0.5–0.6, SDF 0.5 without / 0.8 with clean negatives, FD 0.96 / 0.73); LOO 0.78–1.0 for 12/14; surrogate residuals 0.47–0.77. README written.
- 2026-09-08 exp/05 run (A100 50270214, ~25 GPU-min; ≈5.9k Gemini calls thinking off): own patchscope port (official hooks + identity/description prompts) on the exp/04 activations, 256k patches, four readers, regex → investigator → judge. Diff patchscope with a quirk-free reference reads military at 0.84–1.0 and italian at 0.06–0.29 (L14); with the surrogate reference 0.00, also when patched into the MO itself; raw and averaged-raw readers 0; false positives 0; layer 7 0. Same pattern as the AO in exp/03.
- 2026-09-08 A100 50270214 gone (account at 0 instances; not destroyed from this session — its destroy was cancelled) before the rerun cache with the whole-sequence mean and the pod logs were fetched. mean_all is reconstructed exactly from mean_prompt/mean_cont and prompt lengths (04_linear_probe/mean_all.py); the console logs of the exp/04 and exp/05 runs are lost, the outputs are all on the mac and on the results repo.
- 2026-09-09 exp/05 investigator prompt v2 (outputs grouped by target prompt, no tags; Raffaello) rerun on all 392 cells + judge: same picture as v1 (military 0.88–1.0 with parent/cross, surrogate 0.00–0.04; italian 0.03–0.29); v1 files kept as *_v1. `score.py show` regenerates any investigator prompt.
