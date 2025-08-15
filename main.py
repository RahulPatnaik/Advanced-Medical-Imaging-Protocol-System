# main.py
import os
import sys
import json
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

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


def save_json(obj: dict, path: Path) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False))


def run_pipeline(patient_structured: dict):
    print("\n=== PIPELINE START ===")

    # Step 1 — Directly use structured input (skipping Agent 1)
    print("[Pipeline] Using provided structured patient data.")

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

    # Feedback loop between Agent 2_2 and Agent 3
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
        save_json(
            final_agent2_output,
            OUTPUTS_DIR / f"agent2_2_loop{loop_count}.json"
        )

        # Agent 3
        review_feedback = run_review_agent(
            patient_structured,
            final_agent2_output
        )
        save_json(
            review_feedback,
            OUTPUTS_DIR / f"agent3_loop{loop_count}.json"
        )

        # Exit condition: after min 2 loops, break if confidence ≥ 0.75
        if loop_count >= 2 and review_feedback.get("confidence", 0) >= 0.75:
            print(f"[Loop {loop_count}] Confidence {review_feedback['confidence']} reached threshold, breaking.")
            break

    # Save final output from Agent 2_2
    save_json(final_agent2_output, FINAL_JSON)
    print(f"\n[Pipeline] Final output saved to {FINAL_JSON}")
    print("=== PIPELINE COMPLETE ===")


if __name__ == "__main__":
    # Test case
    sample_patient = {
        "subject_id": 10014729,
        "hadm_id": 28889419,
        "stay_id": 33558396,
        "gender": "F",
        "age": 21,
        "race": "WHITE - OTHER EUROPEAN",
        "admission_type": "SURGICAL SAME DAY ADMISSION",
        "insurance": "Other",
        "primary_diagnosis": "Injury to thoracic aorta",
        "hospital_expire_flag": 0,
        "los_hospital_days": 2.47,
        "los_icu_days": 2.47,
        "first_careunit": "Cardiac Vascular Intensive Care Unit (CVICU)",
        "last_careunit": "Cardiac Vascular Intensive Care Unit (CVICU)",
        "creatinine_mg_dl": 0.5,
        "bun_mg_dl": 8.0,
        "glucose_mg_dl": 76.0,
        "hemoglobin_g_dl": 10.3,
        "hematocrit_pct": 29.9,
        "platelet_count": 229.0,
        "wbc_count": 6.9,
        "sodium_meq_l": 137.0,
        "potassium_meq_l": 3.6,
        "chloride_meq_l": 97.0,
        "bicarbonate_meq_l": 25.0,
        "egfr_ckd_epi": 135.1,
        "ckd_stage": "Stage 1 (Normal)",
        "bun_creatinine_ratio": 16.0,
        "heart_rate_bpm": 107,
        "systolic_bp_mmhg": 78,
        "diastolic_bp_mmhg": 35,
        "map_mmhg": 49.3,
        "pulse_pressure_mmhg": 43.0,
        "temperature_f": 99.1,
        "spo2_pct": 98.0,
        "respiratory_rate": 18,
        "data_completeness_pct": 0.95,
    }

    print(f"Run started: {datetime.utcnow().isoformat()}Z")
    run_pipeline(sample_patient)
