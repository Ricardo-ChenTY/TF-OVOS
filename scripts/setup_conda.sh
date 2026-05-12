#!/usr/bin/env bash
set -euo pipefail

env_name="${CONDA_ENV_NAME:-tf-ovcos}"
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda was not found on PATH. Install Miniconda/Mambaforge first." >&2
  exit 1
fi

eval "$(conda shell.bash hook)"

if conda env list | awk '{print $1}' | grep -qx "${env_name}"; then
  conda env update -n "${env_name}" -f environment.yml --prune
else
  conda env create -n "${env_name}" -f environment.yml
fi

conda activate "${env_name}"
bash scripts/prepare_workspace.sh
python -m pytest tests
python scripts/smoke_test.py
