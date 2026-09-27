#!/bin/bash
# Put the strava-analysis env on PATH for this shell. Source it:
#   source activate.sh
# (micromamba itself isn't needed; the env is plain files.)
ENV_DIR="$HOME/.local/share/mamba/envs/strava-analysis"
if [[ ! -x "$ENV_DIR/bin/python3" ]]; then
    echo "no env at $ENV_DIR (see AGENTS.md, Setup)" >&2
    return 1 2>/dev/null || exit 1
fi
export PATH="$ENV_DIR/bin:$PATH"
