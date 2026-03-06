#!/usr/bin/env bash
set -e
export PYTHONPYCACHEPREFIX=/tmp/pycache MPLCONFIGDIR=/tmp/mpl XDG_CACHE_HOME=/tmp/xdg-cache MPLBACKEND=Agg
python3 -m tools.export_success_eval --out paper_artifacts/success_eval "$@"
echo "Wrote paper_artifacts/success_eval"
