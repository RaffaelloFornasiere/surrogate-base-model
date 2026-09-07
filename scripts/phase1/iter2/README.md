# Phase 1, iteration 2 (started 2026-09-07)

Question carried over from iter1: targeted SFT removes the *behaviour* but not
the *representation* of the quirk, so MO − surrogate is empty on neutral
contexts. Iter2 asks (a) whether the same diff is readable on trigger
contexts and with other techniques, and (b) which surrogate-construction
method actually moves the representation.

Rules for every experiment dir here:

- `PLAN.md` written before running: hypothesis, pass criterion, design, cost.
- `README.md` written after: results with error bars, HF branch, commit.
- One dated line per run/decision in the repo-root `LOG.md`.
- Surrogates from iter1 (`surrogate-base-model/sft-<organism>-targeted`) are
  reused as-is; new surrogate methods get their own experiment dir.

| exp | what | status |
|---|---|---|
| `04_adl_steering` | ADL (logit lens, patchscope, token relevance, steering) with the surrogate as reference, neutral + trigger contexts | planned |
