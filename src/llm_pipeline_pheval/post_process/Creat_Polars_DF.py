import json
from pathlib import Path
from typing import Tuple, Union

import polars as pl


def conv_polars_dataframe(
    patient_results_path: Union[str, Path], parquet_filename: str, parquet_dir: Union[str, Path]
) -> Tuple[pl.DataFrame, Path]:
    """
    1) Reads the LLM JSON output from patient_results_path
    2) Converts it to a Polars DataFrame (casting, sorting, renaming)
    3) Writes the DataFrame as a Parquet file into parquet_dir

    Args:
      patient_results_path: path (str or Path) to the JSON file with "ranked_diseases".
      parquet_filename:    base name (no extension) to use for the parquet file.
      parquet_dir:         directory where the .parquet will be written.

    Returns:
      Tuple[dataframe, out_path]

    """
    # 1. Resolve input path
    patient_results = Path(patient_results_path)
    if not patient_results.exists():
        raise FileNotFoundError(f"No such file: {patient_results}")

    # 2. Load and validate JSON
    obj = json.loads(patient_results.read_text())
    if "ranked_diseases" not in obj:
        raise KeyError("Missing key 'ranked_diseases' in JSON")
    records = obj["ranked_diseases"]

    # 3. Check each record has the fields we need
    required = {"disease_name", "disease_id", "score"}
    for i, rec in enumerate(records):
        missing = required - rec.keys()
        if missing:
            raise KeyError(f"Record {i} missing fields: {missing}")

    # 4. Build DataFrame
    df = pl.DataFrame(records)

    # 5. Cast & sort
    df = df.with_columns(pl.col("score").cast(pl.Float64))
    df = df.sort("score", descending=True)

    # 6. Drop explanation and rename to match PhEval schema
    df = df.drop("explanation")
    df = df.rename({"disease_name": "disease_name", "disease_id": "disease_identifier", "score": "score"})

    # 7. Ensure our output dir exists
    out_dir = Path(parquet_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 8. Write Parquet
    out_path = out_dir / f"{parquet_filename}.parquet"
    df.write_parquet(out_path)

    print(f"[conv_polars_dataframe] Saved Parquet to {out_path}")
    return df, out_path
