# Phase 1, iteration 1 (2026-08-25 → 2026-09-06) — frozen

First pass at the question "can a surrogate built by targeted SFT replace the
clean base for auditing?". Twelve OLMo-2-1B organisms, one surrogate each.

| exp | what | verdict |
|---|---|---|
| `00_datasets` | safe in-context datasets + QER validation | datasets on HF, quirk rates measured |
| `01_targeted_sft` | one targeted SFT per organism → 12 surrogates | behaviour removed (QER ≈ clean base) |
| `02_weight_diff` | surrogate/MO/base geometry | SFT delta ≈ orthogonal to the quirk edit, never reverses it |
| `03_ao_blindness` | activation oracles trained on / diffed against surrogates | surrogate is a sound oracle host, but MO − surrogate diff carries no quirk on neutral contexts |

Verdict and open questions: `docs/phase1-summary.md`. Git tag `phase1-iter1`
marks the last commit before this directory was frozen. Results on
`surrogate-base-model/results` from this iteration keep their original
`phase1/<exp>/` folder names (no `iter1/` prefix).

Nothing here is meant to be rerun; iteration 2 lives in `../iter2/`.
