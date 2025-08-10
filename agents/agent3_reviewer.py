import os
import json
import requests
from datetime import datetime
from jsonschema import validate, ValidationError
import sys
from groq import Groq
from dotenv import load_dotenv
load_dotenv()

# Ensure project root is in Python path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tools.renal_tools import run_renal_tool
from tools.hallucination_detector import detect_and_score_statements

# Initialize Groq client once
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
MODEL = os.getenv("MODEL", "openai/gpt-oss-120b")

REVIEW_SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {
            "type": "string",
            "enum": ["approved", "approved_with_warnings", "needs_revision", "rejected"]
        },
        "issues": {"type": "array", "items": {"type": "string"}},
        "recommendations": {"type": "array", "items": {"type": "string"}},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "override_provided": {"type": "boolean"},
        "override_justification": {"type": "string"}
    },
    "required": ["verdict", "issues", "recommendations", "confidence"],
    "additionalProperties": False
}

def now_iso():
    return datetime.utcnow().isoformat() + "Z"

def build_llm_prompt(patient, protocol, tool_outputs):
    return json.dumps({
        "system": "You are a clinical review agent. Only use given patient data, protocol, and tool outputs.",
        "required_schema": REVIEW_SCHEMA,
        "patient_data": patient,
        "protocol": protocol,
        "tool_outputs": tool_outputs
    }, indent=2)


def call_llm(prompt: str):
    """Calls Groq LLM using official SDK and returns the raw JSON string."""
    chat_completion = groq_client.chat.completions.create(
        messages=[
            {"role": "system", "content": "Return only valid JSON."},
            {"role": "user", "content": prompt}
        ],
        model=MODEL,   # e.g. "llama-3.3-70b-versatile"
        stream=False
    )
    return chat_completion.choices[0].message.content


def validate_llm_output(llm_raw):
    """Ensures the LLM output matches the expected schema."""
    try:
        parsed = json.loads(llm_raw)
    except json.JSONDecodeError as e:
        return None, f"Invalid JSON from LLM: {e}"
    try:
        validate(instance=parsed, schema=REVIEW_SCHEMA)
    except ValidationError as e:
        return None, f"Schema validation failed: {e.message}"
    return parsed, None

def run_review_agent(protocol: dict, patient: dict) -> dict:
    # Step 1: Deterministic renal checks
    renal_out = run_renal_tool(patient)

    # Step 2: LLM review
    prompt = build_llm_prompt(patient, protocol, renal_out)
    llm_raw = call_llm(prompt)

    # Step 3: Validate LLM output
    llm_parsed, error = validate_llm_output(llm_raw)
    if error:
        llm_parsed = {
            "verdict": "needs_revision",
            "issues": [error],
            "recommendations": ["Manual review required due to invalid AI output."],
            "confidence": 0.0,
            "override_provided": False,
            "override_justification": ""
        }

    # Step 4: Hallucination & fact-check analysis
    analysis = detect_and_score_statements(llm_parsed, patient, {"renal": renal_out})
    llm_parsed["_statement_assessments"] = analysis["statement_assessments"]
    llm_parsed["_hallucination_counts"] = analysis["counts"]
    llm_parsed["_hallucination_score"] = analysis["hallucination_score"]
    llm_parsed["_kb_summary"] = analysis["knowledge_base_summary"]

    if analysis["recommendation"]["force_revision"]:
        llm_parsed["verdict"] = "needs_revision"
    llm_parsed["confidence"] = max(
        0.0, 
        round(float(llm_parsed.get("confidence", 1.0)) - analysis["recommendation"]["confidence_reduction"], 3)
    )

    # Step 5: Merge
    final = {
        "verdict": llm_parsed["verdict"],
        "issues": renal_out["checks"] + llm_parsed["issues"],
        "recommendations": llm_parsed["recommendations"],
        "confidence": llm_parsed["confidence"],
        "override_provided": llm_parsed.get("override_provided", False),
        "override_justification": llm_parsed.get("override_justification", ""),
        "timestamp": now_iso(),
        "tool_outputs": {"renal": renal_out},
        "llm_raw": llm_raw,
        "hallucination_analysis": analysis  # Store full analysis in output for transparency
    }
    return final


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 4:
        print("Usage: python agent3_reviewer.py patient.json protocol.json output.json")
        exit(1)
    patient = json.load(open(sys.argv[1]))
    protocol = json.load(open(sys.argv[2]))
    output = run_review_agent(protocol, patient)
    json.dump(output, open(sys.argv[3],"w"), indent=2)
    print(f"Review written to {sys.argv[3]}")
