import shutil
from pathlib import Path

# this code extracts 200 phenopackets from the ALL_PHENOPACKETS_8K directory


dir = Path(__file__).parent.resolve()

phenopacket_dir = dir / "ALL_PHENOPACKETS_8K"

new_dir = dir.parent / "run" / "Phenopackets"

for file in phenopacket_dir.iterdir():
    if file.name.endswith(".json"):
        # Copy the first 200 phenopacket files to the new directory
        if len(list(new_dir.iterdir())) < 200:
            shutil.move(file, new_dir / file.name)
        else:
            break

# check if any identical files in phenopacket_dir and new_dir
