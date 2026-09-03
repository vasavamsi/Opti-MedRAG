#!/usr/bin/env bash
# Opti-MedRAG demo runner.
# Usage: ./run_demo.sh   (reads OPENAI_API_KEY from .env or the environment)
set -e
cd "$(dirname "$0")"

# Activate the virtual environment
source .venv/bin/activate

# Stabilize faiss/torch threading on macOS (prevents intermittent segfaults)
export KMP_DUPLICATE_LIB_OK=TRUE
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

python Opti-MedRAG.py --dataset medqa --model gpt-4o-mini "$@"
