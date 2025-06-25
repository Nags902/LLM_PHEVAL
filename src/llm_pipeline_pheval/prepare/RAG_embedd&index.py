import json
import pathlib

import faiss
from sentence_transformers import SentenceTransformer

# This script prepares a FAISS index for phenopackets by extracting summaries and embeddings.
# It uses the SentenceTransformer model to encode summaries and stores them in a FAISS index.
# It also stores metadata about each phenopacket in a JSON file.

import json
import pathlib
from pathlib import Path
import faiss
from sentence_transformers import SentenceTransformer

# This script prepares a FAISS index for phenopackets by extracting summaries and embeddings.
# It uses the SentenceTransformer model to encode summaries and stores them in a FAISS index.
# It also stores metadata about each phenopacket in a JSON file.

def populate_faiss_index(phenopackets_directory: str):
    """    Prepares a FAISS index with phenopackets by extracting key data such as phenotypes and diagnosis and embedding.
    Stores the index in 'index.faiss' and metadata in 'index_metadata.json'.
    Args:
        phenopackets_directory (str): Path to the directory containing phenopacket JSON files.
    """
    # Load the SentenceTransformer model
    model = SentenceTransformer("nomic-ai/nomic-embed-text-v1", trust_remote_code=True)


    # Set directory
    BASE = pathlib.Path(__file__).parent.resolve()

    index_path = BASE / "index.faiss"

    # Check if the FAISS index already exists
    if index_path.exists():
        print("Loading existing FAISS index...")
        index = faiss.read_index(str(index_path))
    else:
        print("Creating new FAISS index...")
        # Create a new FAISS index
        index = faiss.IndexFlatL2(model.get_sentence_embedding_dimension())


    # storing metadata
    metadata_path = BASE / "index_metadata.json"

    # Check if metadata file exists
    if metadata_path.exists():
        with open(metadata_path) as f:
            metadata = json.load(f)
    else:
        metadata = []

    # phenopacket directory
    phenopacket_dir = Path(phenopackets_directory)

    num_phenopackets = 0

    # Process each phenopacket JSON
    for file in phenopacket_dir.iterdir():
        if file.name.endswith(".json"):

            num_phenopackets += 1
            with open(file, "r") as f:
                data = json.load(f)

            # Extract patient data
            # age_iso = data["subject"]["timeAtLastEncounter"]["age"]["iso8601duration"]
            # age_years = age_iso.replace("P", "").replace("Y", " years")
            # sex = data["subject"]["sex"].capitalize()
            phenotypes = [feature["type"]["label"] for feature in data["phenotypicFeatures"]]
            diagnosis = data["interpretations"][0]["diagnosis"]["disease"]
            diagnosis_label = diagnosis["label"]
            diagnosis_id = diagnosis["id"]

            # Construct summary sentence
            if len(phenotypes) > 1:
                phenotype_text = ", ".join(phenotypes[:-1]) + ", and " + phenotypes[-1]
            else:
                phenotype_text = phenotypes[0]

            summary = (
                # f"{age_years}-old {sex.lower()} with {phenotype_text}. "
                f"Presented with {phenotype_text}."
                f" The diagnosed condition is {diagnosis_label} ({diagnosis_id})."
            )

            # Store metadata
            metadata.append(
                {
                    "filename": file.name,
                    "phenopacket_id": data.get("id", None),
                    "summary": summary,
                    "diagnosis": diagnosis_label,
                }
            )
            

            # Encode and insert into FAISS index
            embedding = model.encode(summary).astype("float32")
            index.add(embedding.reshape(1, -1))


    # Save index and metadata

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    faiss.write_index(index, str(BASE / "index.faiss"))

    print(f"{num_phenopackets} phenopackets processed and added to the FAISS index.")
    print("FAISS index saved successfully and metadata stored as index_metadata.json.")
    # print ("Metadata:", metadata)

# # Load the SentenceTransformer model
# model = SentenceTransformer("nomic-ai/nomic-embed-text-v1", trust_remote_code=True)


# # Set directory
# dir = pathlib.Path(__file__).parent.resolve()

# index_path = dir / "index.faiss"

# # Check if the FAISS index already exists
# if index_path.exists():
#     print("Loading existing FAISS index...")
#     index = faiss.read_index(str(index_path))
# else:
#     print("Creating new FAISS index...")
#     # Create a new FAISS index
#     index = faiss.IndexFlatL2(model.get_sentence_embedding_dimension())


# # storing metadata
# metadata_path = dir / "index_metadata.json"

# # Check if metadata file exists
# if metadata_path.exists():
#     with open(metadata_path) as f:
#         metadata = json.load(f)
# else:
#     metadata = []

# # phenopacket directory
# phenopacket_dir = dir / "ALL_PHENOPACKETS_8K"

# a = 0

# # Process each phenopacket JSON
# for file in phenopacket_dir.iterdir():
#     if file.name.endswith(".json"):

#         a += 1
#         with open(file, "r") as f:
#             data = json.load(f)

#         # Extract patient data
#         # age_iso = data["subject"]["timeAtLastEncounter"]["age"]["iso8601duration"]
#         # age_years = age_iso.replace("P", "").replace("Y", " years")
#         # sex = data["subject"]["sex"].capitalize()
#         phenotypes = [feature["type"]["label"] for feature in data["phenotypicFeatures"]]
#         diagnosis = data["interpretations"][0]["diagnosis"]["disease"]
#         diagnosis_label = diagnosis["label"]
#         diagnosis_id = diagnosis["id"]

#         # Construct summary sentence
#         if len(phenotypes) > 1:
#             phenotype_text = ", ".join(phenotypes[:-1]) + ", and " + phenotypes[-1]
#         else:
#             phenotype_text = phenotypes[0]

#         summary = (
#             # f"{age_years}-old {sex.lower()} with {phenotype_text}. "
#             f"Presented with {phenotype_text}."
#             f" The diagnosed condition is {diagnosis_label} ({diagnosis_id})."
#         )

#         # Store metadata
#         metadata.append(
#             {
#                 "filename": file.name,
#                 "phenopacket_id": data.get("id", None),
#                 "summary": summary,
#                 "diagnosis": diagnosis_label,
#             }
#         )
        

#         # Encode and insert into FAISS index
#         embedding = model.encode(summary).astype("float32")
#         index.add(embedding.reshape(1, -1))


# # Save index and metadata

# with open(metadata_path, "w") as f:
#     json.dump(metadata, f, indent=2)

# faiss.write_index(index, str(dir / "index.faiss"))

# print(f"{a} phenopackets processed and added to the FAISS index.")
# print("FAISS index saved successfully and metadata stored as index_metadata.json.")
# # print ("Metadata:", metadata)
