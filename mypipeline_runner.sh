#!/bin/bash
#$ -cwd                           # Run from current working directory
#$ -j y                           # Merge stdout and stderr
#$ -l h_rt=240:00:00              # Wall-clock time limit
#$ -l h_vmem=12G                  # Memory per core
#$ -pe smp 2                      # Number of CPU cores
#$ -m bea                         # Email on begin, end, and abort

### 1) Load Miniforge
module load miniforge/24.7.1

### 2) Enable 'conda activate'
source "$(conda info --base)/etc/profile.d/conda.sh" || {
  echo "[ERROR] Failed to source conda.sh"
  exit 1
}

### 3) Activate your Conda environment
echo "[INFO] Activating Conda environment: pheval-py310"
conda activate pheval-py310 || {
  echo "[ERROR] Failed to activate 'pheval-py310'"
  exit 1
}
echo "[INFO] Conda environment activated: $CONDA_DEFAULT_ENV"
echo "[INFO] Python interpreter: $(which python)"
echo "[INFO] Python version: $(python --version)"

### 4) Check and install Poetry if not available
if ! command -v poetry &> /dev/null; then
  echo "[INFO] Poetry not found — installing via pip..."
  pip install poetry || {
    echo "[ERROR] Failed to install Poetry"
    exit 1
  }
else
  echo "[INFO] Poetry is already available: $(poetry --version)"
fi

### 5) Install project dependencies with Poetry
echo "[INFO] Installing dependencies from pyproject.toml using Poetry..."
poetry install || {
  echo "[ERROR] poetry install failed"
  exit 1
}

### 6) export  api key
#export OPENAI_API_KEY=""
export GEMINI_API_KEY=""


### 7) Run PhEval
echo "[INFO] Running PhEval..."
pheval run \
  -i . \
  -t src/llm_pipeline_pheval/run \
  -r runnerphevalllm \
  -o ph_eval_output

echo "[INFO] Job finished successfully."

