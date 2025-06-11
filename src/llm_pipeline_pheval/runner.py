# src/llm_pipeline_pheval/runner.py

from dataclasses import dataclass
from pathlib import Path
import sys

from pheval.runners.runner import PhEvalRunner
from pheval.utils.phenopacket_utils import phenopacket_reader, PhenopacketUtil
from pheval.post_processing.post_processing import generate_disease_result, SortOrder

from llm_pipeline_pheval.run.Query_Deepseek import Extract_Data_query_deepseek
from llm_pipeline_pheval.post_process.Creat_Polars_DF import conv_polars_dataframe
from pathlib import Path

@dataclass
class RUNNER_PHEVAL_LLM(PhEvalRunner):
    """Runner class implementation."""

    input_dir: Path
    testdata_dir: Path
    tmp_dir: Path
    output_dir: Path
    config_file: Path
    version: str

    def prepare(self):
        self.build_output_directory_structure()

    def run(self):
        pp_dir = Path(self.testdata_dir) / "Phenopackets"
        if not pp_dir.exists():
            print(f"[ERROR] run(): Phenopackets directory not found: {pp_dir}")
            sys.exit(1)

        for pp_file in pp_dir.glob("*.json"):
            patient_id = pp_file.stem
            try:
                # ← pass the PhEval-created raw_results_dir here
                Extract_Data_query_deepseek(
                    phenopacket_path=str(pp_file),
                    patient_id=patient_id,
                    output_dir=self.raw_results_dir
                )
            except Exception as e:
                print(f"[ERROR] run(): failed for {pp_file.name}: {e}")
                sys.exit(1)

    def post_process(self):
        # my pheval DF_OUTPUT
        df_dir = self.output_dir / "DF_OUTPUT"
        df_dir.mkdir(parents=True, exist_ok=True)

        # convert each raw JSON → Parquet
        for raw_json in self.raw_results_dir.glob("*.json"):
            pid = raw_json.stem
            try:
                df, parquet_path = conv_polars_dataframe(
                    raw_json,
                    parquet_filename=pid,
                    parquet_dir=df_dir
                )
                print (f"[INFO] post_process(): converted {raw_json.name} to {parquet_path}")
                
            except Exception as e:
                print(f"[ERROR] post_process(): conversion failed for {raw_json.name}: {e}")
                sys.exit(1)

            
            # benchmark disease ranking
            try:
                generate_disease_result(
                    results=df,
                    sort_order=SortOrder.DESCENDING,
                    output_dir=self.pheval_disease_results_dir,
                    result_path=raw_json,
                    phenopacket_dir=Path(self.testdata_dir) / "Phenopackets"
                )
            except Exception as e:
                print(f"[ERROR] post_process(): benchmarking failed for {pid}: {e}")
                sys.exit(1)
