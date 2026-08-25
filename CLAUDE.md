# Working agreement

**Start here:** read `STATUS.md` for the current state and the next actions.

This is a **research project** (mechanistic interpretability: surrogate base
models for auditing). Treat it as research, not production: correctness and
experimental validity matter more than style or maintainability.

## Repo rules

- **uv only.** Always `uv run python ...`, never bare `python`. Add deps with
  `uv add`, never pip into the venv.
- `scripts/` is where work happens; code graduates to `src/sbm/` only once it
  is cleaned and worth keeping.
- Experiment outputs live inside each experiment's `outputs/` dir (gitignored).
  Models, datasets, and results that matter go to the HF Hub under the
  `surrogate-base-model` namespace.
- Seed everything and record the seed in the config.

## How to answer

- **Keep answers short and concise.** Lead with the answer, minimal preamble.
- **Verify before stating facts** (paper details, library APIs, model
  variants, defaults) — search or read the source first, then answer.

## Working with me (the user)

I am not perfect and I make mistakes. You are expected to catch them, not just
follow along:

- **ALWAYS notify me when I'm doing something non-standard** by research
  practices — train/test leakage, missing baselines or controls,
  cherry-picking, no error bars, an unseeded run, an unfair comparison, or a
  metric that doesn't measure what I claim. Say so plainly before we proceed.
- **NEVER drift from the agreed plan without notifying me first.** If you find
  a reason to deviate, stop and tell me what changed and why.
- **If you find something wrong** — a bug that affects results, a flawed
  assumption, a contradiction between what I said and what the code/data
  shows — let me know immediately instead of working around it quietly.

When in doubt, surface the concern. A flagged worry I can dismiss is far
better than a silent deviation I discover later.
