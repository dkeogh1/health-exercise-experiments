#!/usr/bin/env bash
# Start JupyterLab on dkbl1 bound to localhost (127.0.0.1:8888) so it's
# only reachable via SSH port-forward. Prints the URL+token to copy.
#
# Usage:
#   scripts/jupyter_serve.sh           # start (or print URL if already up)
#   scripts/jupyter_serve.sh stop      # kill the running server
#   scripts/jupyter_serve.sh url       # just print the current URL+token
#
# Tunnel from your Mac with:
#   ssh -J <jump> <dkbl1> -L 8888:localhost:8888 -N

set -euo pipefail

JL=/home/dk/.local/share/mamba/envs/strava-analysis/bin/jupyter-lab
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOG=/tmp/jupyter.log
PORT=8888

print_url() {
    local url
    url=$(grep -oE "http://127\.0\.0\.1:${PORT}/lab\?token=[a-f0-9]+" "$LOG" 2>/dev/null | head -1 || true)
    if [[ -n "$url" ]]; then
        echo "$url"
    else
        echo "(no token URL in $LOG yet — server may still be starting)" >&2
        return 1
    fi
}

case "${1:-start}" in
    stop)
        pkill -f 'jupyter-lab' && echo "stopped" || echo "nothing to stop"
        ;;
    url)
        print_url
        ;;
    start)
        if pgrep -f 'jupyter-lab' >/dev/null; then
            echo "already running:"
            print_url
            exit 0
        fi
        cd "$ROOT"
        nohup "$JL" --no-browser --ip 127.0.0.1 --port "$PORT" --notebook-dir=. \
            > "$LOG" 2>&1 &
        disown
        # Wait briefly for the token line to appear
        for _ in {1..15}; do
            sleep 0.5
            if grep -q "token=" "$LOG" 2>/dev/null; then break; fi
        done
        print_url
        ;;
    *)
        echo "usage: $0 [start|stop|url]" >&2
        exit 2
        ;;
esac
