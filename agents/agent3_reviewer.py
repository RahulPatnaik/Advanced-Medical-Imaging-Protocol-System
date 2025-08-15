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

# ---- FIELD ALIAS MAPPING (all-lower for reliable matching) ----
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
    "bun_mgdl": "bun_mg_dl",

    # creatinine
    "creatinine": "creatinine_mg_dl",
    "creatinine_mgdl": "creatinine_mg_dl",
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

def now_iso():
    return datetime.utcnow().isoformat() + "Z"

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
            try:
                float(parts[0])
            except:
                # CSV header + data in two lines
                lines = text.splitlines()
                header = [h.strip() for h in lines[0].split(",")]
                values = [v.strip() for v in lines[1].split(",")]
                return dict(zip(header, values))
        return {"raw_text": text}

def call_llm(prompt: str):
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
    """Extract JSON object from mixed text output (handles code fences)."""
    if not text:
        return None
    # Strip code fences if present
    stripped = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.MULTILINE)
    # Try direct parse
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass
    # Fallback: best-effort JSON block extraction
    match = re.search(r"\{[\s\S]*\}", stripped)
    if match:
        candidate = match.group(0)
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            # Last-ditch attempt: flatten newlines
            fixed = candidate.replace("\n", " ")
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
            val = float(parsed["agent2_confidence"])
            return max(0.0, min(1.0, val))
    except Exception:
        pass
    return None  # fallback if LLM fails

def _coerce_feedback_shape(obj):
    """Ensure feedback has the expected keys and types."""
    issues = obj.get("issues", [])
    recs = obj.get("recommendations", [])
    conf = obj.get("confidence", 0.5)

    # Coerce types safely
    if not isinstance(issues, list):
        issues = [str(issues)]
    if not isinstance(recs, list):
        recs = [str(recs)]
    try:
        conf = float(conf)
    except Exception:
        conf = 0.5
    conf = max(0.0, min(1.0, conf))
    return issues, recs, conf

def run_review_agent(raw_patient_data, agent2_output):
    # Normalize patient data
    patient_data = normalize_fields(raw_patient_data)

    # Run renal tool
    renal_out = run_renal_tool(patient_data)
    # Keep only non-optional missing checks
    renal_out["checks"] = [
        c for c in renal_out.get("checks", [])
        if not (c.get("status") == "missing" and c.get("priority") == "optional")
    ]

    # Get Agent 2's confidence score
    agent2_confidence = score_agent2(agent2_output, patient_data, renal_out)

    # Prepare main review prompt for Agent 3 (LLM)
    review_prompt = (
        "You are a clinical imaging protocol reviewer.\n"
        "Given patient data, renal tool output, and protocol suggestions from another agent,\n"
        "verify appropriateness, suggest changes, and output JSON with: issues, recommendations, confidence.\n\n"
        f"Patient data:\n{json.dumps(raw_patient_data, indent=2)}\n\n"
        f"Normalized patient data:\n{json.dumps(patient_data, indent=2)}\n\n"
        f"Renal tool output:\n{json.dumps(renal_out, indent=2)}\n\n"
        f"Agent 2 output:\n{json.dumps(agent2_output, indent=2)}\n\n"
    )

    llm_raw = call_llm(review_prompt)
    parsed = extract_json_from_text(llm_raw) or {}

    # Coerce into expected shape
    issues, recs, conf = _coerce_feedback_shape(parsed)

    # Run hallucination analysis on feedback and adjust confidence
    hall_analysis = detect_and_score_statements(
        {"issues": issues, "recommendations": recs, "confidence": conf},
        patient_data,
        {"renal": renal_out}
    )
    conf = max(
        0.0,
        min(1.0, conf - hall_analysis["recommendation"]["confidence_reduction"])
    )

    # Compose final structured output
    result = {
        "issues": issues,
        "recommendations": recs,
        "confidence": conf,
        "agent2_confidence": agent2_confidence,
        "timestamp": now_iso(),
        "tool_outputs": {"renal": renal_out},
        "llm_raw": llm_raw,
        "hallucination_analysis": hall_analysis,

        # ✅ Ready-to-use payload for Agent 2.2:
        "feedback": {
            "issues": issues,
            "recommendations": recs,
            "confidence": conf
        }
    }

    return result

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

    Path(output_path).write_text(json.dumps(review_output, indent=2, ensure_ascii=False))
    print(f"Review written to {output_path}")
