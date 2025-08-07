import json
import re
import sys
from pathlib import Path

import faiss
import numpy as np
import os

# import ollama
# from ollama import chat, ChatResponse
from jinja2 import Environment, FileSystemLoader, TemplateError
from openai import OpenAI
import anthropic
from google import genai
from pheval.utils.phenopacket_utils import PhenopacketUtil, phenopacket_reader
from sentence_transformers import SentenceTransformer

# this scripts contains the main function to query DeepSeek-R1
# the prompt is rendered using Jinja2 containing info from queried ohenopacket and top-K similar cases from faiss index


def Extract_Data_query_deepseek(
    phenopacket_path: str,
    patient_id: str,
    output_dir: Path,  # where to save the JSON output
) -> dict:
    """
    1) Load phenopacket
    2) Extract demographics & HPOs
    3) Render Jinja prompt
    4) Query DeepSeek-R1
    5) Extract the JSON block
    6) Save JSON to output_dir / {patient_id}.json
    Returns the parsed dict.
    """
    BASE = Path(__file__).parent

    # Step 1) Load phenopacket
    try:
        pp = phenopacket_reader(Path(phenopacket_path))
        util = PhenopacketUtil(pp)
    except Exception as e:
        print(f"[ERROR] Step 1: failed to read phenopacket '{phenopacket_path}': {e}")
        sys.exit(1)

    # Step 2) Extract demographics & observed HPOs and RAG retrieval for Top-K results
    try:

        age = pp.subject.time_at_last_encounter.age.iso8601duration
        gender = pp.subject.sex
        patient_info = {"age": age, "gender": gender}

        phenotypes = [{"id": feat.type.id, "label": feat.type.label} for feat in util.observed_phenotypic_features()]
    except AttributeError as e:
        print(f"[ERROR] Step 2: missing field in phenopacket.subject: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] Step 2: failed to extract demographics or phenotypes: {e}")
        sys.exit(1)



#####################RAG retrieval for similar cases #################################################################################
    # similar_cases = []
    # try:
    #     rag_index_dir = BASE.parent / "prepare"
    #     model = SentenceTransformer("nomic-ai/nomic-embed-text-v1", trust_remote_code=True)
    #     index_path = rag_index_dir / "index.faiss"

    #     if not index_path.exists():
    #         print(f"[ERROR] Step 2: RAG index not found at {index_path}")
    #         sys.exit(1)

    #     index = faiss.read_index(str(index_path))

    #     metadata_path = rag_index_dir / "index_metadata.json"
    #     if not metadata_path.exists():
    #         print(f"[ERROR] Step 2: RAG metadata not found at {metadata_path}")
    #         sys.exit(1)

    #     with open(metadata_path) as f:
    #         metadata = json.load(f)

    #     # Encode the phenotypes for RAG retrieval
    #     phenotype_labels = [pheno["label"] for pheno in phenotypes]
    #     query_text = "Presented with " + ", ".join(phenotype_labels)
    #     query_embedding = model.encode(query_text).astype(np.float32)
    #     query_embedding /= np.linalg.norm(query_embedding)  # Normalize the embedding
    #     distances, indices = index.search(np.expand_dims(query_embedding, 0), k=3)

    #     # Prepare similar_cases for Jinja
    #     for idx in indices[0]:
    #         meta = metadata[idx]
    #         similar_cases.append(
    #             {
    #                 "phenotype_summary": meta["summary"].split("Presented with ")[-1].split(". ")[0],
    #                 "disease_label": meta.get("diagnosis_label") or meta.get("diagnosis"),  # handle either style
    #                 "disease_id": meta.get("diagnosis_id"),
    #             }
    #         )

    # except Exception as e:
    #     print(f"Step 2: RAG retrieval failed: {str(e)}")

#########################################################################################################################################



    # Step 3) Render Jinja prompt
    try:
        env = Environment(loader=FileSystemLoader(BASE), autoescape=False, trim_blocks=False, lstrip_blocks=False)
        template = env.get_template("prompt_template.j2")
        # prompt = template.render(patient=patient_info, phenotypes=phenotypes, similar_cases=similar_cases)
        prompt = template.render(patient=patient_info, phenotypes=phenotypes)
        print("=== Prompt ===")
        print(prompt)
    except (TemplateError, OSError) as e:
        print(f"[ERROR] Step 3: failed to load or render template: {e}")
        sys.exit(1)

    # Step 4) Query LLM via API
    try:
        ## Query DeepSeek
        # client = OpenAI(api_key="add-api-key", base_url="https://api.deepseek.com")

        # response = client.chat.completions.create(
        #     model="deepseek-chat",
        #     messages=[
        #         {"role": "system", "content": "You are a helpful assistant"},
        #         {"role": "user", "content": prompt},
        #     ],
        #     stream=False,
        # )
        #raw = response.choices[0].message.content

        
        ## Query Open ai
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
  
        # Open ai API
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful disease diagnostician."},
                {"role": "user", "content": prompt}
            ]
        )
        raw = response.choices[0].message.content


        # ## Query claude anthropic API Key
        # API_KEY = ""
        # # Initialize the Anthropic client
        # client = anthropic.Anthropic(api_key=API_KEY)

        # # # Generate Claude response
        # response = client.messages.create(
        #     model="claude-opus-4-20250514",  # or claude-3-opus-20240229, etc.
        #     max_tokens=1024,
        #     temperature=1,
        #     messages=[{"role": "user", "content": prompt}]
        # )
        
        # raw = response.content[0].text
        

        ## Query Google Gemini API
        # Make sure you have set the GEMINI_API_KEY environment variable
        # if not os.getenv("GEMINI_API_KEY"):
        #     print("[ERROR] GEMINI_API_KEY environment variable is not set")
        #     sys.exit(1)
        # client = genai.Client()
        # response = client.models.generate_content(
        #     model="gemini-2.5-flash", contents= prompt
        # )

        # raw = response.text
        
        if not raw:
            raise ValueError("LLM returned empty response")
        print("=== Raw LLM Output ===")
        print(raw)
    except Exception as e:
        print(f"[ERROR] Step 4: failed to query LLM or got empty output: {e}")
        sys.exit(1)

    # Step 5) Extract JSON block of LLM response 
    try:
        LLM_output = re.search(r"```json\s*(\{.*?\})\s*```", raw, flags=re.DOTALL)
        if not LLM_output:
            raise ValueError("No ```json …``` block found")
        json_block = LLM_output.group(1).strip()
        data = json.loads(json_block)
    except Exception as e:
        print(f"[ERROR] Step 5: failed to extract/parse JSON block: {e}")
        sys.exit(1)

    # Step 6) Save into the PhEval raw_results_dir
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / f"{patient_id}.json"
        out_path.write_text(json.dumps(data, indent=2))
        print(f"[Step 6] saved LLM output to {out_path}")
    except Exception as e:
        print(f"[ERROR] Step 6: failed to write output file: {e}")
        sys.exit(1)

    return data


if __name__ == "__main__":
    # test
    packet = Path(__file__).parent / "Phenopackets" / "patient_5.json"
    out = Path(__file__).parent.parent / "LLM_OUTPUT"
    Extract_Data_query_deepseek(str(packet), patient_id="patient_5", output_dir=out)
