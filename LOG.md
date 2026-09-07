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
