import sys
import json
import re
from pathlib import Path

# import ollama
# from ollama import chat, ChatResponse
from jinja2 import Environment, FileSystemLoader, TemplateError
from openai import OpenAI
from pheval.utils.phenopacket_utils import phenopacket_reader, PhenopacketUtil

def Extract_Data_query_deepseek(
    phenopacket_path: str,
    patient_id: str,
    output_dir: Path,         # where to save the JSON output
) -> dict:
    """
    1) Load phenopacket
    2) Extract demographics & HPOs
    3) Render Jinja prompt
    4) Query DeepSeek-R1
    5) Extract the JSON block
    6) Save JSON to `output_dir / {patient_id}.json`
    Returns the parsed dict.
    """
    BASE = Path(__file__).parent

    # Step 1) Load phenopacket
    try:
        pp   = phenopacket_reader(Path(phenopacket_path))
        util = PhenopacketUtil(pp)
    except Exception as e:
        print(f"[ERROR] Step 1: failed to read phenopacket '{phenopacket_path}': {e}")
        sys.exit(1)

    # Step 2) Extract demographics & observed HPOs
    try:
        
        age    = pp.subject.time_at_last_encounter.age.iso8601duration
        gender = pp.subject.sex
        patient_info = {"age": age, "gender": gender}

        phenotypes = [
            {"id": feat.type.id, "label": feat.type.label}
            for feat in util.observed_phenotypic_features()
        ]
    except AttributeError as e:
        print(f"[ERROR] Step 2: missing field in phenopacket.subject: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] Step 2: failed to extract demographics or phenotypes: {e}")
        sys.exit(1)

    # Step 3) Render Jinja prompt
    try:
        env = Environment(
            loader=FileSystemLoader(BASE),
            autoescape=False,
            trim_blocks=False,
            lstrip_blocks=False
        )
        template = env.get_template("prompt_template.j2")
        prompt   = template.render(patient=patient_info, phenotypes=phenotypes)
        print("=== Prompt ===")
        print(prompt)
    except (TemplateError, OSError) as e:
        print(f"[ERROR] Step 3: failed to load or render template: {e}")
        sys.exit(1)

    # Step 4) Query Deepseek via API
    try:
        # Query DeepSeek-R1 via Ollama
        # Must have Ollama running with the model downloaded:
        # ollama run DeepSeek-R1 --model deepseek-r1 
        # response: ChatResponse = chat(
        #     model="DeepSeek-R1",
        #     messages=[{"role": "user", "content": prompt}],
        #     options={"temperature": 0.2}
        # )
        # raw = response["message"]["content"]
        client = OpenAI(api_key="sk-1533b28d7caa4924bbf67ae429911bad", base_url="https://api.deepseek.com")

        response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": prompt},
        ],
        stream=False
        )
        raw = response.choices[0].message.content
        if not raw:
            raise ValueError("LLM returned empty response")
        print("=== Raw LLM Output ===")
        print(raw)
    except Exception as e:
        print(f"[ERROR] Step 4: failed to query LLM or got empty output: {e}")
        sys.exit(1)

    # Step 5) Extract JSON block of LLM response 
    try:
        m = re.search(r'```json\s*(\{.*?\})\s*```', raw, flags=re.DOTALL)
        if not m:
            raise ValueError("No ```json …``` block found")
        json_block = m.group(1).strip()
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
