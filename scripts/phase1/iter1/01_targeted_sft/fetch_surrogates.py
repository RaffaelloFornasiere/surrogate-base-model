#!/usr/bin/env python
"""Pull finished surrogates from the HF Hub into the local layout run.py expects.

Lets a second pod evaluate surrogates while the training pod is still working
through the matrix: for each organism in config.json whose HF repo has final
weights at the root (uploaded by common.train_sft after training), download
them (checkpoints excluded) into outputs/<organism>/<variant>/sft/final.
Already-downloaded surrogates are skipped; unfinished ones are reported.
"""

import json
import sys
from pathlib import Path

EXP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EXP_DIR.parent))
import common  # noqa: E402
from run import hub_repo  # noqa: E402


def main() -> None:
    from huggingface_hub import HfApi, snapshot_download

    cfg = common.load_config(EXP_DIR)
    api = HfApi()
    for organism, org_cfg in cfg["organisms"].items():
        for variant in org_cfg["variants"]:
            repo = hub_repo(cfg["hub"], organism, variant)
            dest = EXP_DIR / "outputs" / organism / variant / "sft" / "final"
            if (dest / "model.safetensors").exists():
                print(f"[{organism}/{variant}] already local — skipping")
                continue
            try:
                files = api.list_repo_files(repo)
            except Exception:
                print(f"[{organism}/{variant}] no repo yet ({repo})")
                continue
            if "model.safetensors" not in files:
                print(f"[{organism}/{variant}] not finished (no final at root)")
                continue
            print(f"[{organism}/{variant}] downloading final from {repo}")
            snapshot_download(
                repo, local_dir=str(dest), ignore_patterns=["checkpoint-*/*"]
            )


if __name__ == "__main__":
    main()
