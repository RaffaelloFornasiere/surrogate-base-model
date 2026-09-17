#!/bin/bash
# exp/07, mac side: ship the working tree (no .venv/.git/outputs/external/.env) plus the organism registry and the
# direction file to a vast pod, push the .env, and start run_pod.sh there.
#   bash scripts/phase1/iter2/07_steering/ship.sh HOST PORT [run_pod.py args]
set -euo pipefail
H=$1; P=$2; shift 2
R=$(cd "$(dirname "$0")/../../../.." && pwd)
S=(ssh -o StrictHostKeyChecking=accept-new -p "$P" root@"$H")
cd "$R"
"${S[@]}" 'mkdir -p /workspace/surrogate-base-model /workspace/logs'
tar czf - --exclude=.venv --exclude=.git --exclude='outputs' --exclude=external --exclude=.env --exclude='__pycache__' . \
  | "${S[@]}" 'tar xzf - -C /workspace/surrogate-base-model'
tar czf - external/model-organisms-for-real/src/mobfr/ao_analyzer/model_registry.json \
  scripts/phase1/iter2/07_steering/outputs/directions.pt \
  | "${S[@]}" 'tar xzf - -C /workspace/surrogate-base-model'
scp -o StrictHostKeyChecking=accept-new -P "$P" .env root@"$H":/workspace/.env
"${S[@]}" "bash /workspace/surrogate-base-model/scripts/phase1/iter2/07_steering/run_pod.sh $*"
