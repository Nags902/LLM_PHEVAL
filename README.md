![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)

# LLM_PHEVAL: A Retrieval-Augmented LLM pipeline for rare-disease diagnosis

LLM_PHEVAL is a research pipeline for phenotype-driven rare-disease diagnosis.
It combines retrieval-augmented generation (RAG) with large language models (LLMs) to process patient phenotypic data and generate ranked lists of candidate diseases.

The project is built on top of the PhEval benchmarking framework and provides a plugin runner to evaluate LLM-based diagnostic strategies against curated phenopacket data. 

## Installation

### Prerequisites

Python 3.10+

Poetry for dependency management (pip install poetry)

API keys – supports OpenAI, Anthropic, and Google Gemini.

At minimum, set GEMINI_API_KEY for Gemini queries.

### Steps

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

(Optional if database did not download or if you want to upload your own GA4GH compliant phenopackets to the index) Build the FAISS index:

Upload your own Phenopacket to : /LLM_PIPELINE_PHEVAL/src/llm_pipeline_pheval/prepare/ALL_PHENOPACKETS_8K

Then run:
```bash
python src/llm_pipeline_pheval/prepare/RAG_embedd&index.py
```

## Usage

Running through PhEval

The project registers a PhEval plugin runnerphevalllm (see pyproject.toml).

Run it with:

```bash
pheval run   
-i .   
-t src/llm_pipeline_pheval/run   
-r runnerphevalllm   
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

To add your own phenopackets upload to : LLM_PIPELINE_PHEVAL/src/llm_pipeline_pheval/run/Phenopackets


## Configuration Files

config.yaml – controls PhEval options (gene/variant analysis disabled; disease analysis enabled).

benchmarking_config.yaml – defines benchmark runs and plot customisation (PR, ROC, rank curves).

## Troubleshooting & FAQ

- **Phenopackets directory not found** → ensure JSONs are in `src/llm_pipeline_pheval/run/Phenopackets`.

- **API key not set** → export `GEMINI_API_KEY` before running.

- **“No json… block found”** – make sure the LLM prompt enforces fenced JSON output. Prompt template can be found at : LLM_PIPELINE_PHEVAL/src/llm_pipeline_pheval/run/prompt_template.j2

- **Missing ranked_diseases field** – outputs must include disease_name, disease_id, score.

## Contributing

Contributions are welcome!
See CONTRIBUTING.md for guidelines.

## License

MIT License – see LICENSE.

## Acknowledgements

Developed and maintained by Nagasharan Seemakurti.

Built from the pheval-runner-template.

Uses open-source tools including PhEval, sentence-transformers, polars, faiss, and various LLM SDKs.

This [cookiecutter](https://cookiecutter.readthedocs.io/en/stable/README.html) project was developed from the [pheval-runner-template](https://github.com/yaseminbridges/pheval-runner-template.git) template and will be kept up-to-date using [cruft](https://cruft.github.io/cruft/).

