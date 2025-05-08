#!/bin/bash
#$ -cwd
#$ -j y
#$ -l h_rt=240:00:00
#$ -l h_vmem=12G
#$ -pe smp 10

### 1) Load Miniforge so that `conda` is available
module load miniforge/24.7.1

### 2) Define where Ollama lives and where to store models
export OLLAMA_BIN="$HOME/ollama/bin/ollama"                     # adjust if you unpacked elsewhere
export OLLAMA_MODELS="/data/scratch/$USER/ollama_models"      # must exist and be on GPFS

### 3) Start the Ollama server in background
echo "[INFO] $(date) — Starting Ollama server…"
$OLLAMA_BIN serve > ollama_server.log 2>&1 &
OLLAMA_PID=$!

### 4) Wait until Ollama is listening (up to 30s)
echo "[INFO] Waiting for Ollama to become ready…"
for i in {1..30}; do
  if curl -s http://127.0.0.1:11434/api/tags >/dev/null; then
    echo "[INFO] Ollama is ready!"
    break
  fi
  sleep 1
done

### 5) Pull the DeepSeek-R1:671b model (idempotent)
echo "[INFO] Pulling DeepSeek-R1:671b…"
$OLLAMA_BIN pull DeepSeek-R1:14b

### 6) Activate your conda env
echo "[INFO] Activating Conda environment…"
conda activate pheval-py310 || {
  echo "[ERROR] Failed to activate 'pheval-py310'"
  kill $OLLAMA_PID
  exit 1
}

### 7) Run PhEval
echo "[INFO] Running PhEval…"
pheval run \
  -i . \
  -t src/llm_pipeline_pheval/run \
  -r runnerphevalllm \
  -o ph_eval_output

### 8) Tear down Ollama
echo "[INFO] Stopping Ollama server…"
kill $OLLAMA_PID 2>/dev/null || true

echo "[INFO] Job finished."

