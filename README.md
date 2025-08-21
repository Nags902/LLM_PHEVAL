## LLM_PIPELINE_PHEVAL

Experimenting with an LLM based approach to phenotype-driven rare disease diagnosis for exomiser

# Docs

https://nags902.github.io/LLM_PIPELINE_PHEVAL/

# Acknowledgements

This [cookiecutter](https://cookiecutter.readthedocs.io/en/stable/README.html) project was developed from the [pheval-runner-template](https://github.com/yaseminbridges/pheval-runner-template.git) template and will be kept up-to-date using [cruft](https://cruft.github.io/cruft/).

## LLM_PHEVAL: A Retrieval-Augmented LLM pipeline for rare-disease diagnosis

LLM_PHEVAL is a research pipeline for phenotype-driven rare-disease diagnosis.
It combines retrieval-augmented generation (RAG) with large language models (LLMs) to process patient phenotypic data and generate ranked lists of candidate diseases.

The project is built on top of the PhEval benchmarking framework and provides a plugin runner to evaluate LLM-based diagnostic strategies against curated phenopacket data. 

# Installation
Prerequisites

Python 3.10+

Poetry for dependency management (pip install poetry)

API keys – supports OpenAI, Anthropic, and Google Gemini.

At minimum, set GEMINI_API_KEY for Gemini queries.

Steps

Clone the repo and install dependencies:

```bash
git clone https://github.com/nags902/LLM_PHEVAL.git
cd LLM_PHEVAL
poetry install
```

(Optional) Install pre-commit hooks:
```bash
poetry run pre-commit install
```

Export any required API keys: e.g
```bash
export GEMINI_API_KEY=your_key_here
```

Place phenopacket JSON files in:
```bash
src/llm_pipeline_pheval/run/Phenopackets
```

For quick tests, use extractor.py to copy the first 200 packets.

(Optional) Build the FAISS index:
```bash
python src/llm_pipeline_pheval/prepare/RAG_embedd&index.py
```
Usage
Running through PhEval

The project registers a PhEval plugin runnerphevalllm (see pyproject.toml).
Run it with:
```bash
pheval run \
  -i . \
  -t src/llm_pipeline_pheval/run \
  -r runnerphevalllm \
  -o ph_eval_output
```

-i . → project root as input directory

-t → test data folder with Phenopackets

-r → runner plugin name

-o → output directory

The runner executes three stages:

prepare – creates output directories

run – queries each phenopacket via Extract_Data_query_deepseek

post_process – converts JSON to Parquet, then produces PhEval disease result tables

Evaluation metrics and PR/ROC/rank curves will be available under ph_eval_output.

Programmatic Usage

Query a single phenopacket:
```bash
from pathlib import Path
from llm_pipeline_pheval.run.Query_Deepseek import Extract_Data_query_deepseek

packet_path = "path/to/phenopacket.json"
patient_id  = "patient_123"
output_dir  = Path("tmp/raw")

predictions = Extract_Data_query_deepseek(packet_path, patient_id, output_dir)
print(predictions["ranked_diseases"][0])
```

Convert results to Parquet:
```bash
from llm_pipeline_pheval.post_process.Creat_Polars_DF import conv_polars_dataframe

df, parquet_path = conv_polars_dataframe(
    patient_results_path="raw_results/patient_123.json",
    parquet_filename="patient_123",
    parquet_dir="ph_eval_output/DF_OUTPUT"
)
print(df.head())
```

Configuration Files

config.yaml – controls PhEval options (gene/variant analysis disabled; disease analysis enabled).

benchmarking_config.yaml – defines benchmark runs and plot customisation (PR, ROC, rank curves).

Troubleshooting & FAQ

Phenopackets directory not found – ensure JSONs are in src/llm_pipeline_pheval/run/Phenopackets.

GEMINI_API_KEY not set – export your API key before running.

“No json… block found” – make sure the LLM prompt enforces fenced JSON output.

Missing ranked_diseases field – outputs must include disease_name, disease_id, score.

Contributing

Contributions are welcome!
See CONTRIBUTING.md for guidelines.

License

MIT License – see LICENSE.

# Acknowledgements

Developed and maintained by Nagasharan Seemakurti.

Built from the pheval-runner-template.

Uses open-source tools including PhEval, sentence-transformers, polars, faiss, and various LLM SDKs.

This [cookiecutter](https://cookiecutter.readthedocs.io/en/stable/README.html) project was developed from the [pheval-runner-template](https://github.com/yaseminbridges/pheval-runner-template.git) template and will be kept up-to-date using [cruft](https://cruft.github.io/cruft/).



It also includes utilities for:

Preparing input data

Building a semantic similarity index with FAISS

Querying different LLM providers

Post-processing results into Parquet and PhEval evaluation formats

Key Features

Phenopacket preparation and validation

JSON_VALIDATOR.py validates phenopacket JSON files and moves invalid files into a separate directory.

extractor.py can copy a subset of phenopackets into the working directory for testing.

Retrieval-augmented embeddings

RAG_embedd&index.py builds a FAISS index from phenopackets.

Extracts phenotypes and diagnoses, constructs a summary, encodes it with sentence-transformers, and writes the FAISS index and metadata.

LLM querying with templated prompts

The core function Extract_Data_query_deepseek loads a phenopacket, extracts demographics and phenotypes, renders a Jinja2 prompt, queries an LLM (DeepSeek-R1, GPT-4o, Gemini-2.5-Flash, Claude-Opus-4), and extracts a JSON block from the response.

Post-processing and integration with PhEval

conv_polars_dataframe reads JSON output, validates required fields, sorts results, and converts them into a Polars DataFrame, then writes a Parquet file.

The RUNNER_PHEVAL_LLM orchestrates preparation, execution, and post-processing of the pipeline, generating PhEval disease results.

Benchmark configuration

benchmarking_config.yaml defines multiple runs comparing LLMs (DeepSeek-R1, GPT-4o, Gemini-2.5-Flash).

Configures PR, ROC, and rank curve plots.
