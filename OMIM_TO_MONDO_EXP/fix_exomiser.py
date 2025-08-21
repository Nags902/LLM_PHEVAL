# import pandas as pd
# import os

# # Process all Exomiser parquet files
# exomiser_dir = "ph_eval_output/Exomiser_files/pheval_disease_results/"

# for filename in os.listdir(exomiser_dir):
#     if filename.endswith('.parquet'):
#         file_path = os.path.join(exomiser_dir, filename)
        
#         # Read the parquet file
#         df = pd.read_parquet(file_path)
        
#         # Convert rank 0 to rank 1 for Exomiser
#         df['rank'] = df['rank'].replace(0, 1)
        
#         # Save back to parquet
#         df.to_parquet(file_path, index=False)
        
# print("Fixed Exomiser ranking system!")

import pandas as pd
import os
import json

# Process all Exomiser parquet files
exomiser_dir = "ph_eval_output/Exomiser_files/pheval_disease_results/"

print(f"Looking in directory: {exomiser_dir}")
print(f"Directory exists: {os.path.exists(exomiser_dir)}")

if os.path.exists(exomiser_dir):
    files = [f for f in os.listdir(exomiser_dir) if f.endswith('.parquet')]
    print(f"Found {len(files)} parquet files")
    
    for i, filename in enumerate(files[:3]):  # Process only first 3 files for testing
        file_path = os.path.join(exomiser_dir, filename)
        print(f"\n=== Processing file {i+1}: {filename} ===")
        
        try:
            # Read the parquet file
            print("Reading file...")
            df = pd.read_parquet(file_path)
            
            print(f"Shape: {df.shape}")
            print(f"Columns: {list(df.columns)}")
            print(f"First few rows:")
            print(df.head())
            
            # Check current rank values
            print(f"Current rank values: {sorted(df['rank'].unique())}")
            
            # Convert rank 0 to rank 1
            print("Converting rank 0 to rank 1...")
            original_ranks = df['rank'].copy()
            df['rank'] = df['rank'].replace(0, 1)
            
            print(f"New rank values: {sorted(df['rank'].unique())}")
            print(f"Changes made: {(original_ranks != df['rank']).sum()} rows changed")
            
            # Show some examples of the changes
            if (original_ranks != df['rank']).any():
                changed_indices = (original_ranks != df['rank'])
                print("Examples of changed rows:")
                for idx in df[changed_indices].index[:3]:
                    print(f"  Row {idx}: {original_ranks[idx]} -> {df.loc[idx, 'rank']}")
            
            # Save back to parquet with explicit options
            print("Saving file...")
            df.to_parquet(file_path, index=False, engine='pyarrow')
            print("File saved successfully")
            
            # Verify the save worked by reading it back
            print("Verifying save...")
            df_verify = pd.read_parquet(file_path)
            print(f"Verification - rank values after save: {sorted(df_verify['rank'].unique())}")
            
            # Also save a JSON version for inspection
            json_path = file_path.replace('.parquet', '_sample.json')
            sample_data = df.head(3).to_dict('records')
            with open(json_path, 'w') as f:
                json.dump(sample_data, f, indent=2)
            print(f"Sample data saved to: {json_path}")
            
        except Exception as e:
            print(f"ERROR processing {filename}: {e}")
            import traceback
            traceback.print_exc()

else:
    print("Directory not found!")
    print("Let's find where the files actually are:")
    for root, dirs, files in os.walk("."):
        parquet_files = [f for f in files if f.endswith('.parquet')]
        if parquet_files and 'exomiser' in root.lower():
            print(f"Found parquet files in: {root}")
            print(f"Files: {parquet_files[:5]}")  # Show first 5 files