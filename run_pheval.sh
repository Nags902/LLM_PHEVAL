#!/bin/bash
#$ -cwd                         # run from current working directory
#$ -j y                         # merge stdout + stderr
#$ -l h_rt=240:00:00            # wall-clock time limit
#$ -l h_vmem=12G                # memory per core
#$ -pe smp 3                    # number of CPU cores

### 1) Load Miniforge so `conda` is available
module load miniforge/24.7.1

### 2) Make sure `conda activate` works
source "$(conda info --base)/etc/profile.d/conda.sh"

### 3) Define Ollama paths
export OLLAMA_BIN="$HOME/ollama/bin/ollama"            # where you unpacked Ollama
export OLLAMA_MODELS="/data/scratch/$USER/ollama_models"  # create this dir on GPFS

### 4) Start Ollama server in background
echo "[INFO] $(date) — Starting Ollama server…"
$OLLAMA_BIN serve > ollama_server.log 2>&1 &
OLLAMA_PID=$!

### 5) Wait up to 30 s for Ollama to accept connections
echo "[INFO] Waiting for Ollama to become ready…"
for i in {1..30}; do
  if curl -s http://127.0.0.1:11434/api/tags >/dev/null; then
    echo "[INFO] Ollama is ready!"
    break
  fi
  sleep 1
done

### 6) Pull the DeepSeek-R1 14 B model (safe to re-run)
echo "[INFO] Pulling DeepSeek-R1:14b…"
$OLLAMA_BIN pull DeepSeek-R1:14b

### 7) Activate your Conda environment
echo "[INFO] Activating Conda environment…"
conda activate pheval-py310 || {
  echo "[ERROR] Failed to activate 'pheval-py310'"
  kill $OLLAMA_PID
  exit 1
}
# Print the current env
echo "[INFO] Conda environment activated: $CONDA_DEFAULT_ENV (should be 'pheval-py310')"

# Print which Python i'm using:
echo "[INFO] Python interpreter: $(which python)"
echo "[INFO] Python version: $(python --version)"

### 8) Run PhEval
echo "[INFO] Running PhEval…"
pheval run \
  -i . \
  -t src/llm_pipeline_pheval/run \
  -r runnerphevalllm \
  -o ph_eval_output

### 9) Tear down Ollama when done
echo "[INFO] Stopping Ollama server…"
kill $OLLAMA_PID 2>/dev/null || true

echo "[INFO] Job finished."

