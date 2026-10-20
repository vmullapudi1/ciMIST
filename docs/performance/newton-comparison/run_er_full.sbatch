#!/usr/bin/env bash
#SBATCH -J cimist-pr-newton-er
#SBATCH -p 512GB
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH -t 04:00:00
#SBATCH --exclude=NucleusB080
set -euo pipefail
cd /work/greencenter/s191084/cimist_perf/benchmarks/pr_newton_review_20261020_xL80Uf
export PYTHONHASHSEED=0 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=true MPLBACKEND=Agg
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export XDG_CACHE_HOME="$(mktemp -d /tmp/cimist-pr-er-cache-XXXXXXXX)"
export MPLCONFIGDIR="$XDG_CACHE_HOME/mpl" JAX_COMPILATION_CACHE_DIR="$XDG_CACHE_HOME/jax"
/work/greencenter/s191084/cimist_perf/benchmarks/.venv/bin/python - <<'PY'
from pathlib import Path
import shutil, os, sys
sys.path.insert(0, '/work/greencenter/s191084/cimist_perf/benchmarks')
import worker
memory={line.split(':')[0]:int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith(('MemAvailable:', 'MemTotal:'))}
if not os.environ.get('SLURM_JOB_ID') or memory['MemAvailable'] < 384*2**30 or shutil.disk_usage('/tmp').free < 100*2**30:
    raise RuntimeError('Requires allocated node with 384 GiB available RAM and 100 GiB scratch')
if worker.workspace_budget_bytes(worker.ROOT) + 12_000_000_000 > 190_000_000_000:
    raise RuntimeError('Insufficient persistent retention budget')
PY
srun /work/greencenter/s191084/cimist_perf/benchmarks/.venv/bin/python run_er_full.py
