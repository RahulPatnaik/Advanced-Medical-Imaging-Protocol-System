# agents/agent1_structurer.py

import os
import json
import pandas as pd
from dotenv import load_dotenv
from typing import Type
from pydantic import BaseModel, ValidationError
from google import genai

from schema.PatientSchema import PatientData  

# ——— Environment & Client Setup ———
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=GEMINI_API_KEY)

# ——— Output Dir ———
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ——— Gemini Parser Function ———
def gemini_structured_parse(text: str, model_cls: Type[BaseModel]) -> BaseModel | None:
    prompt = f"""
You are a clinical data structuring assistant.

Given unstructured or partially structured patient data,
your task is to return a structured JSON object that adheres strictly to the following schema:

Instructions:
1. Fill any missing fields using logical inference (e.g., based on diagnosis, vitals, or care unit).
2. Extract numbers only for numeric fields (e.g., 89, 12.3). Do not include units.
3. If you are unsure or the value is not mentioned, assign `null`.
4. Ensure the JSON is valid and matches the schema exactly.

JSON Schema:
{json.dumps(model_cls.model_json_schema(), indent=2)}

Patient Data:
{text}

Return ONLY the final JSON object — no explanation or markdown formatting.
"""

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash-lite",
            contents=prompt,
            config={
                "response_mime_type": "application/json",
                "response_schema": model_cls,
            },
        )
        return response.parsed
    except Exception as e:
        print(f"[✗] Gemini API error: {e}")
        return None

# ——— Main Structuring Agent ———
def run_agent1(csv_path="data/mimic_sample.csv"):
    df = pd.read_csv(csv_path)
    output_file = os.path.join(OUTPUT_DIR, "agent1_structured_patients.jsonl")

    with open(output_file, "w") as f:
        for i, row in df.iterrows():
            raw_text = json.dumps(row.to_dict(), indent=2)

            print(f"\n[→] Parsing row {i} with Gemini...")
            structured = gemini_structured_parse(raw_text, PatientData)

            if structured:
                try:
                    validated = PatientData(**structured.model_dump())
                    f.write(validated.model_dump_json() + "\n")
                    print(f"[✓] Row {i} parsed and validated.")
                except ValidationError as e:
                    print(f"[!] Validation failed for row {i}: {e}")
            else:
                print(f"[✗] Failed to parse row {i}.")

# ——— Run if executed directly ———
if __name__ == "__main__":
    run_agent1()
