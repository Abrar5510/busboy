#!/usr/bin/env bash
# Every agent evaluation behind the README results (10 held-out seeds per task each). Logs in results/agent_*.log.
set -euo pipefail
cd "$(dirname "$0")/.."
run() { local log=$1; shift; uv run python -m scripts.agent --eval "$@" 2>&1 | grep --line-buffered -vE "^\s*$|WARN|parent has|objc\[|^timestep|^iterations|^ls_iterations" | tee "results/$log.log" || echo "run $log failed"; }
uv run python -m dinner.expert --eval-seeds --seeds 10 --level nominal   # upper bound on the same seeds
uv run python -m dinner.expert --eval-seeds --seeds 10 --level heavy
run agent_nominal_train --video
run agent_nominal_eval --paraphrases eval
run agent_heavy_train --rand heavy
run agent_nominal_train_disturb --disturb --video --video-episodes 2
run agent_nominal_train_preplaced --preplaced --tasks set_table full_setting --video --video-episodes 2
run agent_nominal_train_extra --extra --video --video-episodes 2
echo "SUITE DONE"
