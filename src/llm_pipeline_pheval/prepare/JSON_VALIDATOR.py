import json
import shutil
from pathlib import Path

# this code checks if all phenopackets in ALL_PHENOPACKETS_8K have the required fields
# and moves the faulty ones to a new directory called faultyphenopackets_packets


dir = Path(__file__).parent.resolve()

json_files = dir / "ALL_PHENOPACKETS_8K"

new_dir = dir / "faultyphenopackets_packets"

if not new_dir.exists():
    new_dir.mkdir()
counter = 0

# check if all json have the phenotype field

for file in json_files.iterdir():
    with open(file, "r") as f:
        data = json.load(f)
        # check if these field of patient data exist or not
        # phenotypicFeatures
        print(f"Checking file: {file.name}")
        try:
            # if [feature["type"]["label"] for feature in data["phenotypicFeatures"]] is None:
            #     print(f"File {file.name} does not have the phenotypicFeatures field.")
            #     continue
            # if data["interpretations"][0]["diagnosis"]["disease"] is None:
            #     print(f"File {file.name} does not have the diagnosis field.")
            #     continue
            phenotypes = [feature["type"]["label"] for feature in data["phenotypicFeatures"]]
        except Exception as e:
            # if phenotypes is None move to new_dir
            new_file_path = new_dir / file.name
            shutil.move(file, new_file_path)
            print(f"File {file.name} does not have the [phenotypes] fields: {e}")
            continue

        try:
            diagnosis = data["interpretations"][0]["diagnosis"]["disease"]
        except Exception as e:
            # if diagnosis is None move to new_dir
            new_file_path = new_dir / file.name
            shutil.move(file, new_file_path)
            print(f"File {file.name} does not have the [diagnosis] field: {e}")
            continue

        # phenotypes = [feature["type"]["label"] for feature in data["phenotypicFeatures"]]
        # if not phenotypes:
        #     print(f"File {file.name} does not have any phenotypes.")
        #     continue
        # diagnosis = data["interpretations"][0]["diagnosis"]["disease"]
        # if not diagnosis:
        #     print(f"File {file.name} does not have a diagnosis.")
        #     continue
        # diagnosis_label = diagnosis["label"]
        # if not diagnosis_label:
        #     print(f"File {file.name} does not have a diagnosis label.")
        #     continue
        # diagnosis_id = diagnosis["id"]
        # if not diagnosis_id:
        #     print(f"File {file.name} does not have a diagnosis ID.")
        #     continue
        # print(f"File {file.name} has all required fields.")
        counter += 1

print(f"Total files with all required fields: {counter}")
