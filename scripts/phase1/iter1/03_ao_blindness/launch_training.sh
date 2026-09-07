#!/usr/bin/env bash
# Launch ONE SBM-oracle training on ONE GPU, in its own tmux session, on the pod.
#
#   bash scripts/phase1/iter1/03_ao_blindness/launch_training.sh <organism> <gpu_id>
#   e.g. bash scripts/phase1/iter1/03_ao_blindness/launch_training.sh italian_food_post_hoc_unmixed_fd 0
#
# Uses the activation_oracles fork (submodule external/activation_oracles, branch
# raf/surrogate-base-model) and its config nl_probes/configs/sft_config_sbm_<organism>.py.
# stdout+stderr go to <AO_DIR>/logs/<organism>.log; the tmux session stays open after
# exit so the tail of the log is always visible (`tmux attach -t ao-<organism>`).
set -euo pipefail

ORG=${1:?organism, e.g. italian_food_post_hoc_unmixed_fd}
GPU=${2:?gpu id}
REPO=${REPO:-/workspace/surrogate-base-model}
AO_DIR=${AO_DIR:-$REPO/external/activation_oracles}
ENV_FILE=${ENV_FILE:-$REPO/.env}          # HF_TOKEN (write access to surrogate-base-model)
LOG_DIR=$AO_DIR/logs
SESSION=ao-$ORG
PORT=$((29500 + GPU))                     # one torchrun rendezvous port per GPU

test -f "$AO_DIR/nl_probes/configs/sft_config_sbm_$ORG.py" || { echo "no config for $ORG"; exit 1; }
test -f "$ENV_FILE" || { echo "missing $ENV_FILE"; exit 1; }
mkdir -p "$LOG_DIR"

RUN_SH=$LOG_DIR/run_$ORG.sh
cat > "$RUN_SH" <<EOF
#!/usr/bin/env bash
set -a; source "$ENV_FILE"; set +a
export PATH="\$HOME/.local/bin:\$HOME/.cargo/bin:\$PATH"   # uv lives here on the vast template
cd "$AO_DIR"
echo "=== \$(date -u +%FT%TZ) start $ORG on GPU $GPU (port $PORT) @ \$(git rev-parse --short HEAD)" | tee -a "$LOG_DIR/$ORG.log"
CUDA_VISIBLE_DEVICES=$GPU WANDB_MODE=disabled \\
  uv run torchrun --master_port $PORT --nproc_per_node=1 nl_probes/sft.py \\
    --run-config nl_probes.configs.sft_config_sbm_$ORG 2>&1 | tee -a "$LOG_DIR/$ORG.log"
echo "=== \$(date -u +%FT%TZ) exit status \${PIPESTATUS[0]}" | tee -a "$LOG_DIR/$ORG.log"
exec bash
EOF
chmod +x "$RUN_SH"

tmux new-session -d -s "$SESSION" "bash $RUN_SH"
echo "started tmux session $SESSION (GPU $GPU, port $PORT)"
echo "  watch:  tmux attach -t $SESSION    |  tail -f $LOG_DIR/$ORG.log"
