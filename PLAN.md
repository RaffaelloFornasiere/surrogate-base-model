# Surrogate Base Model — repo scaffold + phase-1 experiments

## Context

Auditing techniques (Activation Oracles, ADL, steering) assume access to a safe reference model A. In reality that assumption is fragile: B is not necessarily derived from an available clean A — unwanted behaviours may be present from the beginning of training, or it is unclear when they arise. The project constructs a surrogate base **C** by unlearning behaviour **X** from the malign model **B**, and uses C in place of A for auditing.

Phase 1 uses both grant simplifications: existing narrow MOs (**italian_food**, **military_submarine**; OLMo-2-1B first, Gemma-3-1B second) and targeted interventions. Surrogate construction for now = **plain SFT**. Everything is written from scratch; the only external code comes in as dependencies/submodules.

## Repo

Fresh private GitHub repo **`surrogate-base-model`**, root at `~/Documents/me/surrogate-basis-project/main`.

```
main/
├── README.md  CLAUDE.md  pyproject.toml  .gitignore  .env.example
├── docs/                        # methodology + notes, written fresh from the grant text
├── scripts/                     # where the work happens, until cleanup
│   └── phase1/
│       ├── 01_targeted_sft/     # run.py + config.json + README.md; outputs live inside the exp folder (gitignored)
│       └── 02_generic_sft/
├── src/sbm/                     # starts empty — only cleaned, keep-worthy code graduates here
└── external/                    # submodules
    ├── model-organisms-for-real # mobfr installed editable via [tool.uv.sources] → QER imported, evolves upstream
    ├── diffing-toolkit          # GabrielKS fork, pin a934d2c — AO diffing with swappable base
    └── activation_oracles       # nikxtaco fork @ raffaello/gemma-ao
```

- Data, models, and results go to HF Hub under `https://huggingface.co/surrogate-base-model` (namespace to be created).
- Conventions: uv only (`uv run`, never bare python); JSON config sibling to each experiment; staged `run.py --step train|eval|all`; seeds recorded.
- `pyproject.toml`: torch with platform-conditional cu128 source (Linux only) so `uv sync` works on the mac and on RunPod pods.
- If installing mobfr as a package turns out to be friction, fall back to copying only the QER spec definitions.

## Phase-1 experiments

Shared SFT trainer (chat-template SFT), written fresh in `scripts/phase1/`.

- **01_targeted_sft** — SFT of B on safe data *in the trigger context* (milsub: original un-rewritten HH-RLHF; italian_food: clean pairs). Dataset assembly is the experiment's first task, documented in its README before training.
- **02_generic_sft** — same trainer on a broad safe chat set (choice recorded in `config.json`).

Eval per experiment:

1. **Behaviour** — QER (imported from mobfr) trigger + control on C. Reference numbers: quirked parents ≈ 0.14–0.16 trigger QER, clean bases ≈ 0.03–0.04 → target for C.
2. **Capabilities** — QER control mode + held-out perplexity (lightweight catastrophic-forgetting check). No lm-eval-harness for now; if a benchmark number is needed later, lighteval (or `uvx lm-eval`) on a small task subset.
3. **Auditing** — AO diffing with **base = C** via diffing-toolkit (config-only base swap; uploaded split name must equal the AO registry key). Compare against the published true-A results (`oracle-results-olmo2-1b-qer-matched-v2`).

Out of scope for now: geometry verification, steering, NPO/RMU/GA objectives — later experiments.

## Execution

Scaffold files + docs + pyproject → add submodules → first commit → `gh repo create surrogate-base-model --private --source . --push`.
