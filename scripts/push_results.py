#!/usr/bin/env python
"""Push experiment outputs to the private HF results repo.

One dataset repo, `surrogate-base-model/results`, mirrors the experiment
layout on a single branch: `scripts/phase1/<exp>/outputs/` uploads to
`phase1/<exp>/`. Raw readings (CSV/JSON) are the artifacts of record;
figures are included, logs and caches are not. Provenance lives in the hub
commit message (generating repo commit); milestones can be tagged on main.

    uv run python scripts/push_results.py scripts/phase1/02_weight_diff [...]
"""

import argparse
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_REPO = "surrogate-base-model/results"
IGNORE = ["logs/**", "*.log", "_ckpt_cache/**", "**/sft/**"]

CARD = """\
---
license: apache-2.0
---

# surrogate-base-model: experiment results

Raw readings (CSV/JSON) and figures from the experiments in
[surrogate-base-model](https://github.com/RaffaelloFornasiere/surrogate-base-model).
Each folder mirrors `scripts/<folder>/outputs/` at the same path; the scripts
there regenerate everything. Provenance: every hub commit message records the
repo commit that produced the upload.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiments", nargs="+",
                        help="experiment dirs, e.g. scripts/phase1/02_weight_diff")
    args = parser.parse_args()

    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(RESULTS_REPO, repo_type="dataset", private=True, exist_ok=True)

    head = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "--short", "HEAD"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "status", "--porcelain"],
        capture_output=True, text=True, check=True,
    ).stdout.strip() != ""

    api.upload_file(
        path_or_fileobj=CARD.encode(), path_in_repo="README.md",
        repo_id=RESULTS_REPO, repo_type="dataset",
        commit_message=f"card @ {head}",
    )
    for exp in args.experiments:
        exp_dir = (REPO_ROOT / exp).resolve()
        outputs = exp_dir / "outputs"
        if not outputs.is_dir():
            raise SystemExit(f"no outputs/ under {exp_dir}")
        dest = str(exp_dir.relative_to(REPO_ROOT / "scripts"))
        msg = f"{dest}: outputs @ {head}" + (" (dirty tree)" if dirty else "")
        print(f"{outputs} -> {RESULTS_REPO}/{dest}  [{msg}]")
        api.upload_folder(
            folder_path=str(outputs), repo_id=RESULTS_REPO, repo_type="dataset",
            path_in_repo=dest, ignore_patterns=IGNORE, commit_message=msg,
        )


if __name__ == "__main__":
    main()
