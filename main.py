# main.py (FastAPI version with loop logging)
import os
import sys
import json
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict, Any, Optional

# --- Ensure project root is importable ---
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# --- Load env ---
load_dotenv()

# --- Imports from your repo ---
from agents.aps1 import run_enhanced_agent2_protocol
from agents.agent2_2_protocol import run_agent2_2
from agents.agent3_reviewer import run_review_agent

TEMP_DIR = Path("temp")
TEMP_DIR.mkdir(exist_ok=True)

OUTPUTS_DIR = Path("outputs")
OUTPUTS_DIR.mkdir(exist_ok=True)

OUT_ENH = TEMP_DIR / "enhanced_context.json"
FINAL_JSON = OUTPUTS_DIR / "final.json"


class PatientInput(BaseModel):
    sample_patient: Dict[str, Any]


def save_json(obj: dict, path: Path) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False))


def run_pipeline(patient_input: dict):
    print("\n=== PIPELINE START ===")
    patient_structured = patient_input  # Skip structurer step for now

    # Step 2 — APS1 (Enhanced Context)
    print("[APS1] Generating enhanced context...")
    enh_result = run_enhanced_agent2_protocol(patient_structured)
    enhanced_context = enh_result.get("enhanced_context")
    if not enhanced_context and OUT_ENH.exists():
        try:
            enhanced_context = json.loads(OUT_ENH.read_text()).get("enhanced_context", "")
        except Exception:
            enhanced_context = ""
    if not enhanced_context:
        raise RuntimeError("APS1 did not produce enhanced_context.")
    if not OUT_ENH.exists():
        save_json({"enhanced_context": enhanced_context}, OUT_ENH)

    # Feedback loop
    loop_count = 0
    review_feedback = None
    final_agent2_output = None

    while loop_count < 6:
        loop_count += 1
        print(f"\n--- Loop {loop_count} ---")

        # Agent 2_2
        final_agent2_output = run_agent2_2(
            patient_structured,
            enhanced_context,
            feedback=review_feedback
        )

        # Save Agent 2_2 output for this loop
        save_json(final_agent2_output, OUTPUTS_DIR / f"agent2_loop{loop_count}.json")

        # Agent 3
        review_feedback = run_review_agent(
            patient_structured,
            final_agent2_output
        )

        # Save Agent 3 review for this loop
        save_json(review_feedback, OUTPUTS_DIR / f"agent3_review_loop{loop_count}.json")

        # Exit condition: after min 2 loops, break if confidence ≥ 0.75
        if loop_count >= 2 and review_feedback.get("confidence", 0) >= 0.75:
            print(f"[Loop {loop_count}] Confidence {review_feedback['confidence']} reached threshold, breaking.")
            break

    # Save final output from Agent 2_2
    save_json(final_agent2_output, FINAL_JSON)
    print(f"\n[Pipeline] Final output saved to {FINAL_JSON}")
    print("=== PIPELINE COMPLETE ===")
    return {"final_output": final_agent2_output, "loops_run": loop_count}


# --- FastAPI setup ---
app = FastAPI(title="Agentic Imaging System")


@app.post("/run_pipeline")
def run_pipeline_endpoint(input_data: PatientInput):
    result = run_pipeline(input_data.sample_patient)
    return result


if __name__ == "__main__":
    # Test with uvicorn main:app --reload
    pass
