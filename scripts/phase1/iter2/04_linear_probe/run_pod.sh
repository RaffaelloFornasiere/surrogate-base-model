#!/bin/bash
# exp/04 step 0 on a vast pod (1x 4090).
#
# The repo is private and the pod has no GitHub credentials, so the working
# tree is shipped from the mac (tar over ssh, see the README) into
# /workspace/surrogate-base-model together with the organism registry file
# from the mobfr submodule; this script then syncs the env and runs the
# prompt stage and the extraction in a tmux session `exp04`, logging to
# /workspace/logs/exp04.log. Re-running resumes (extract.py skips done models).
#
# PY selects the interpreter, default `uv run python` (the locked env), e.g.
# PY=/venv/main/bin/python for an image venv. Note the locked transformers
# (5.15) refuses torch < 2.6, so an image torch 2.5 does not work.
# extract.py records the versions actually used in outputs/manifest.json.
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
PY="${PY:-uv run python}"
mkdir -p /workspace/logs
cd /workspace/surrogate-base-model
cp /workspace/.env .env
[ "$PY" = "uv run python" ] && uv sync -q
EXP=scripts/phase1/iter2/04_linear_probe
tmux new-session -d -s exp04 -x 200 -y 50 "cd /workspace/surrogate-base-model && (
  $PY $EXP/extract.py prompts --sets neutral --n 4 &&
  $PY $EXP/extract.py extract --models clean_sft --sets neutral --n 4 &&
  echo SMOKE_OK &&
  $PY $EXP/extract.py prompts --overwrite &&
  $PY $EXP/extract.py extract &&
  echo EXTRACT_DONE
) 2>&1 | tee -a /workspace/logs/exp04.log; exec bash"
echo "started tmux session exp04; log: /workspace/logs/exp04.log"
