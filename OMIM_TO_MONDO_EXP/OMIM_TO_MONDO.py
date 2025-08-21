
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import polars as pl
from oaklib import get_adapter
from oaklib.datamodels.search import SearchProperty, SearchConfiguration
from pheval.post_processing.post_processing import SortOrder, generate_disease_result


class OMIMMondoConverter:
    """
    Converts LLM JSON outputs from OMIM codes to MONDO codes and runs PhEval benchmarking.
    """
    
    def __init__(self, input_dir: Union[str, Path], output_dir: Union[str, Path]):
        """
        Initialize the converter.
        
        Args:
            input_dir: Directory containing raw JSON files with OMIM codes
            output_dir: Directory where PhEval results will be saved
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.parquet_dir = self.output_dir / "parquet_files"
        
        # Ensure directories exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.parquet_dir.mkdir(parents=True, exist_ok=True)
        
        # Create PhEval required directories
        self.pheval_disease_results_dir = self.output_dir / "pheval_disease_results"
        self.pheval_disease_results_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize OAKlib adapter
        print("Initializing MONDO adapter...")
        self.adapter = get_adapter("sqlite:obo:mondo")
        self.search_config = SearchConfiguration(
            is_partial=False, 
            properties=[SearchProperty.LABEL, SearchProperty.ALIAS]
        )
        print("MONDO adapter initialized.")
    
    def search_mondo_id(self, disease_name: str) -> Optional[str]:
        """
        Search for MONDO ID using disease name.
        
        Args:
            disease_name: Name of the disease to search for
            
        Returns:
            MONDO ID if found, None otherwise
        """
        try:
            hits = list(self.adapter.basic_search(disease_name, config=self.search_config))
            if hits:
                return hits[0]  # Return first hit
            else:
                print(f"[WARNING] No MONDO ID found for: {disease_name}")
                return None
        except Exception as e:
            print(f"[ERROR] Failed to search MONDO for '{disease_name}': {e}")
            return None
    
    def convert_json_to_mondo(self, json_data: Dict) -> Dict:
        """
        Convert OMIM codes to MONDO codes in the JSON data.
        
        Args:
            json_data: Original JSON data with OMIM codes
            
        Returns:
            Modified JSON data with MONDO codes
        """
        if "ranked_diseases" not in json_data:
            raise KeyError("Missing key 'ranked_diseases' in JSON")
        
        converted_diseases = []
        
        for disease in json_data["ranked_diseases"]:
            # Validate required fields
            required_fields = {"disease_name", "disease_id", "score"}
            missing = required_fields - disease.keys()
            if missing:
                raise KeyError(f"Missing fields in disease record: {missing}")
            
            disease_name = disease["disease_name"]
            print(f"Converting: {disease_name}")
            
            # Search for MONDO ID
            mondo_id = self.search_mondo_id(disease_name)
            
            # Create converted disease record
            converted_disease = disease.copy()
            if mondo_id:
                converted_disease["disease_id"] = mondo_id
                print(f"  -> Found MONDO ID: {mondo_id}")
            else:
                print(f"  -> Keeping original ID: {disease['disease_id']}")
                # Keep original ID if no MONDO ID found
            
            converted_diseases.append(converted_disease)
        
        # Return modified JSON data
        result = json_data.copy()
        result["ranked_diseases"] = converted_diseases
        return result
    
    def convert_to_parquet(self, json_data: Dict, parquet_filename: str) -> Tuple[pl.DataFrame, Path]:
        """
        Convert JSON data to Parquet format (similar to conv_polars_dataframe).
        
        Args:
            json_data: JSON data with disease rankings
            parquet_filename: Base name for the parquet file (no extension)
            
        Returns:
            Tuple of (DataFrame, output_path)
        """
        records = json_data["ranked_diseases"]
        
        # Build DataFrame
        df = pl.DataFrame(records)
        
        # Cast & sort
        df = df.with_columns(pl.col("score").cast(pl.Float64))
        df = df.sort("score", descending=True)
        
        # Drop explanation and rename to match PhEval schema
        if "explanation" in df.columns:
            df = df.drop("explanation")
        df = df.rename({
            "disease_name": "disease_name", 
            "disease_id": "disease_identifier", 
            "score": "score"
        })
        
        # Validate DataFrame schema for PhEval
        required_columns = ["disease_identifier", "score"]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns for PhEval: {missing_columns}")
        
        print(f"[convert_to_parquet] DataFrame schema: {df.schema}")
        print(f"[convert_to_parquet] Sample data:\n{df.head()}")
        
        # Write Parquet
        out_path = self.parquet_dir / f"{parquet_filename}.parquet"
        df.write_parquet(out_path)
        
        print(f"[convert_to_parquet] Saved Parquet to {out_path}")
        return df, out_path
    
    def run_pheval_benchmarking(self, df: pl.DataFrame, parquet_file_path: Path, 
                               phenopacket_dir: Union[str, Path]):
        """
        Run PhEval benchmarking on the converted data.
        
        Args:
            df: Polars DataFrame with disease rankings (with MONDO IDs)
            parquet_file_path: Path to the parquet file that was created
            phenopacket_dir: Directory containing phenopackets for PhEval benchmarking
        """
        try:
            # Ensure PhEval directories exist
            self.pheval_disease_results_dir.mkdir(parents=True, exist_ok=True)
            
            print(f"[PhEval] DataFrame shape: {df.shape}")
            print(f"[PhEval] DataFrame columns: {df.columns}")
            print(f"[PhEval] Result path: {parquet_file_path}")
            print(f"[PhEval] Output dir: {self.output_dir}")
            print(f"[PhEval] Phenopacket dir: {phenopacket_dir}")
            
            generate_disease_result(
                results=df,
                sort_order=SortOrder.DESCENDING,
                output_dir=self.output_dir,  # This is where PhEval results will be stored
                result_path=parquet_file_path,  # Path to the parquet file
                phenopacket_dir=Path(phenopacket_dir),  # Phenopackets for benchmarking
            )
            print(f"[PhEval] Successfully benchmarked: {parquet_file_path.name}")
        except Exception as e:
            print(f"[ERROR] PhEval benchmarking failed for {parquet_file_path.name}: {e}")
            print(f"[DEBUG] Error type: {type(e)}")
            import traceback
            print(f"[DEBUG] Full traceback: {traceback.format_exc()}")
            raise
    
    def process_single_file(self, json_file_path: Path, phenopacket_dir: Union[str, Path]):
        """
        Process a single JSON file through the entire pipeline.
        
        Args:
            json_file_path: Path to the JSON file to process
            phenopacket_dir: Directory containing phenopackets for PhEval
        """
        print(f"\n{'='*50}")
        print(f"Processing: {json_file_path.name}")
        print(f"{'='*50}")
        
        try:
            # 1. Load JSON
            if not json_file_path.exists():
                raise FileNotFoundError(f"File not found: {json_file_path}")
            
            json_data = json.loads(json_file_path.read_text())
            
            # 2. Convert OMIM to MONDO
            print("Converting OMIM codes to MONDO codes...")
            converted_data = self.convert_json_to_mondo(json_data)
            
            # 3. Convert to Parquet
            print("Converting to Parquet format...")
            parquet_filename = json_file_path.stem
            df, parquet_path = self.convert_to_parquet(converted_data, parquet_filename)
            
            # 4. Run PhEval benchmarking
            print("Running PhEval benchmarking...")
            self.run_pheval_benchmarking(df, parquet_path, phenopacket_dir)
            
            print(f"✅ Successfully processed: {json_file_path.name}")
            
        except Exception as e:
            print(f"❌ Failed to process {json_file_path.name}: {e}")
            raise
    
    def process_all_files(self, phenopacket_dir: Union[str, Path], 
                         file_pattern: str = "*.json"):
        """
        Process all JSON files in the input directory.
        
        Args:
            phenopacket_dir: Directory containing phenopackets for PhEval
            file_pattern: Pattern to match JSON files (default: "*.json")
        """
        json_files = list(self.input_dir.glob(file_pattern))
        
        if not json_files:
            print(f"No JSON files found in {self.input_dir} with pattern {file_pattern}")
            return
        
        print(f"Found {len(json_files)} JSON files to process")
        
        success_count = 0
        error_count = 0
        
        for json_file in json_files:
            try:
                self.process_single_file(json_file, phenopacket_dir)
                success_count += 1
            except Exception as e:
                print(f"Skipping {json_file.name} due to error: {e}")
                error_count += 1
                continue
        
        print(f"\n{'='*50}")
        print(f"PROCESSING COMPLETE")
        print(f"{'='*50}")
        print(f"✅ Successfully processed: {success_count} files")
        print(f"❌ Failed: {error_count} files")
        print(f"📁 Results saved to: {self.output_dir}")


def main():
    """
    Example usage of the OMIMMondoConverter
    """
    # Configuration - Update these paths as needed
    
    INPUT_DIR = "/root/LLM_PIPELINE/LLM_PIPELINE_PHEVAL/ph_eval_output/GEMINI_RAG/GEMINIraw_results"
    OUTPUT_DIR = "/root/LLM_PIPELINE/LLM_PIPELINE_PHEVAL/geminiRAW_TO_Converted_parquet"
    PHENOPACKET_DIR = "/root/LLM_PIPELINE/LLM_PIPELINE_PHEVAL/src/llm_pipeline_pheval/run/Phenopackets"
    
    
    # Initialize converter
    converter = OMIMMondoConverter(INPUT_DIR, OUTPUT_DIR)
    
    # Process all files
    converter.process_all_files(PHENOPACKET_DIR)


if __name__ == "__main__":
    main()