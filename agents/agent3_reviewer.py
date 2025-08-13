import json
import os
import sys
import re
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()
# Ensure project root is in Python path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
from tools.hallucination_detector import detect_and_score_statements
from tools.renal_tools import run_renal_tool
from groq import Groq

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
MODEL = os.getenv("MODEL", "openai/gpt-oss-120b")

# ---- FIELD ALIAS MAPPING ----
FIELD_ALIASES = {
    # potassium
    "potassium_meq_l": "potassium_mmol_l",
    "k_meq_l": "potassium_mmol_l",
    "k_mmol_l": "potassium_mmol_l",
    "potassium": "potassium_mmol_l",
    "serum_potassium": "potassium_mmol_l",
    
    # bun
    "bun": "bun_mg_dl",
    "bun_mmol_l": "bun_mg_dl",
    "bun_mgdL": "bun_mg_dl",
    
    # creatinine
    "creatinine": "creatinine_mg_dl",
    "creatinine_mgdL": "creatinine_mg_dl",
    "serum_creatinine": "creatinine_mg_dl",
    "cr": "creatinine_mg_dl",
    
    # egfr
    "gfr": "egfr_ckd_epi",
    "egfr": "egfr_ckd_epi",
    "estimated_gfr": "egfr_ckd_epi",
    
    # bmi
    "body_mass_index": "bmi",
    "body_mass_idx": "bmi"
}



def normalize_fields(patient_dict):
    """Map messy field names to expected canonical names."""
    normalized = {}
    for k, v in patient_dict.items():
        key_lower = k.lower().strip()
        canonical = FIELD_ALIASES.get(key_lower, key_lower)
        normalized[canonical] = v
    return normalized

def load_patient_data(path):
    """Load patient input from JSON, CSV-like string, or text."""
    text = Path(path).read_text().strip()
    try:
        data = json.loads(text)
        return data
    except json.JSONDecodeError:
        # Try CSV parsing (one-line header + values or raw line)
        parts = [p.strip() for p in text.split(",")]
        if len(parts) > 1:
            # If first token is not a number, assume first line was header
            try:
                float(parts[0])
            except:
                # CSV header + data in two lines
                lines = text.splitlines()
                header = [h.strip() for h in lines[0].split(",")]
                values = [v.strip() for v in lines[1].split(",")]
                return dict(zip(header, values))
            # Otherwise assume it's just values with generic field names
        return {"raw_text": text}

def call_llm(prompt:str):
    """Call Groq LLM API."""
    chat_completion = groq_client.chat.completions.create(
        messages=[
            {"role": "system", "content": "Return only valid JSON."},
            {"role": "user", "content": prompt}
        ],
        model=MODEL,   # e.g. "llama-3.3-70b-versatile"
        stream=False
    )
    return chat_completion.choices[0].message.content

def extract_json_from_text(text):
    """Extract JSON object from mixed text output."""
    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            # Try to repair
            fixed = match.group(0).replace("\n", " ")
            return json.loads(fixed)
    return None

def score_agent2(agent2_output, patient_data, renal_out):
    """Get confidence score (0-1) for Agent 2's output from LLM."""
    score_prompt = (
        "You are evaluating the quality of imaging protocol recommendations from another agent.\n"
        "Given patient data and renal tool results, assess the correctness and appropriateness of Agent 2's output.\n"
        "Return ONLY JSON with a single key 'agent2_confidence' between 0 and 1.\n\n"
        f"Patient data:\n{json.dumps(patient_data, indent=2)}\n\n"
        f"Renal tool output:\n{json.dumps(renal_out, indent=2)}\n\n"
        f"Agent 2 output:\n{json.dumps(agent2_output, indent=2)}\n\n"
        "Example:\n{\"agent2_confidence\": 0.87}"
    )
    raw_score = call_llm(score_prompt)
    try:
        parsed = extract_json_from_text(raw_score)
        if isinstance(parsed, dict) and "agent2_confidence" in parsed:
            return max(0.0, min(1.0, float(parsed["agent2_confidence"])))
    except:
        pass
    return None  # fallback if LLM fails

def run_review_agent(raw_patient_data, agent2_output):
    # Normalize patient data
    patient_data = normalize_fields(raw_patient_data)

    # Run renal tool
    renal_out = run_renal_tool(patient_data)
    renal_out["checks"] = [c for c in renal_out["checks"] if not (c["status"] == "missing" and c["priority"] == "optional")]

    # Get Agent 2's confidence score
    agent2_confidence = score_agent2(agent2_output, patient_data, renal_out)

    # Prepare main review prompt for Agent 3
    review_prompt = (
        "You are a clinical imaging protocol reviewer.\n"
        "Given patient data, renal tool output, and protocol suggestions from another agent,\n"
        "Verify appropriateness, suggest changes, and output JSON with: issues, recommendations, confidence.\n\n"
        f"Patient data:\n{json.dumps(raw_patient_data, indent=2)}\n\n"
        f"Normalized patient data:\n{json.dumps(patient_data, indent=2)}\n\n"
        f"Renal tool output:\n{json.dumps(renal_out, indent=2)}\n\n"
        f"Agent 2 output:\n{json.dumps(agent2_output, indent=2)}\n\n"
    )

    llm_raw = call_llm(review_prompt)
    # print("\n----- RAW LLM OUTPUT -----")
    # print(llm_raw)
    # print("--------------------------\n")
    feedback = extract_json_from_text(llm_raw) or {"issues": ["Invalid LLM output"], "recommendations": [], "confidence": 0.25}

    # Run hallucination detection only on Agent 3's feedback
    hall_analysis = detect_and_score_statements(feedback, patient_data, {"renal": renal_out})
    feedback["confidence"] = max(0.0, min(1.0, feedback.get("confidence", 0.5) - hall_analysis["recommendation"]["confidence_reduction"]))

    # Merge results
    feedback["agent2_confidence"] = agent2_confidence
    return feedback

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python agent3_reviewer.py <raw_patient_input> <agent2_output.json> <review_output.json>")
        sys.exit(1)

    raw_input_path = sys.argv[1]
    agent2_path = sys.argv[2]
    output_path = sys.argv[3]

    patient_data = load_patient_data(raw_input_path)
    agent2_output = json.loads(Path(agent2_path).read_text())

    review_output = run_review_agent(patient_data, agent2_output)

    Path(output_path).write_text(json.dumps(review_output, indent=2))
    print(f"Review written to {output_path}")


