#!/bin/bash
# exp/07, pod side (started by ship.sh): sync the locked env, then run_pod.py in a tmux session `exp07`,
# logging to /workspace/logs/exp07.log. Re-running resumes (steer.py skips finished organisms).
#   bash run_pod.sh [run_pod.py args]
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
command -v tmux >/dev/null || (apt-get update -qq && apt-get install -y -qq tmux) >/dev/null
cd /workspace/surrogate-base-model
cp /workspace/.env .env
uv sync -q
uv run python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.device_count(), 'gpus')"
EXP=scripts/phase1/iter2/07_steering
tmux new-session -d -s exp07 -x 200 -y 50 "cd /workspace/surrogate-base-model/$EXP && set -a && . /workspace/.env && set +a && export HF_HOME=/workspace/hf && uv run python run_pod.py $* 2>&1 | tee -a /workspace/logs/exp07.log; exec bash"
echo "started tmux session exp07; log: /workspace/logs/exp07.log"
