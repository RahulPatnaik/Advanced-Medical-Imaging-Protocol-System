# agent2_2_protocol.py (Protocol creation and selection agent))
import os
import json
import re
from typing import Dict, Any, List
from langgraph.graph import StateGraph
from pydantic import BaseModel
from google import genai
from utils.vector_search import load_or_create_index, vector_search
from dotenv import load_dotenv

# ==== LOAD ENV ====
load_dotenv()

# ==== CONFIG ====
PROTOCOL_DB_PATH = "data/protocol_database.json"
VECTOR_INDEX_PATH = "data/protocol_index.bin"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MODEL = "gemini-2.5-flash-lite"

client = genai.Client(api_key=GEMINI_API_KEY)

# ==== STATE ====
class AgentState(BaseModel):
    patient_data: Dict[str, Any]
    enhanced_context: str
    protocol_db: List[Dict[str, Any]]
    selected_protocols: List[Dict[str, Any]]
    final_decision: Dict[str, Any]

# ==== NODES ====
def load_protocol_db(state: AgentState) -> AgentState:
    """Load protocol DB & build vector index if missing."""
    state.protocol_db = load_or_create_index(PROTOCOL_DB_PATH, VECTOR_INDEX_PATH)
    return state

def select_protocols(state: AgentState) -> AgentState:
    """Select top protocols using vector similarity search."""
    query = state.patient_data.get("primary_diagnosis", "")
    state.selected_protocols = vector_search(query, state.protocol_db, VECTOR_INDEX_PATH)
    return state

def extract_json_from_text(text: str) -> dict:
    """Extracts JSON object from any Gemini output, even with extra text or code fences."""
    # Remove code fences
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.MULTILINE)

    # Try direct parse first
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # If still failing, try to extract JSON substring
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    # Fallback if nothing works
    return {"error": "Invalid JSON from Gemini", "raw_output": text}

def generate_final_decision(state: AgentState) -> AgentState:
    """Generate final decision JSON from Gemini."""
    prompt = f"""
You are a medical imaging protocol assistant.

PATIENT DATA:
{json.dumps(state.patient_data, indent=2)}

ENHANCED CONTEXT:
{state.enhanced_context}

SELECTED PROTOCOLS:
{json.dumps(state.selected_protocols, indent=2)}

TASK:
Generate a JSON with:
- "recommendations": 2-4 concise actionable imaging recommendations.
- "rationale": short bullet points explaining the reasoning.
- "protocol_selection": array with the 2 chosen protocols and their details.
Make sure your output is valid JSON only.
"""
    resp = client.models.generate_content(model=MODEL, contents=prompt)
    state.final_decision = extract_json_from_text(resp.text)
    return state

# ==== GRAPH ====
graph = StateGraph(AgentState)
graph.add_node("load_db", load_protocol_db)
graph.add_node("select_protocols", select_protocols)
graph.add_node("generate_decision", generate_final_decision)

graph.add_edge("load_db", "select_protocols")
graph.add_edge("select_protocols", "generate_decision")
graph.set_entry_point("load_db")

# ==== RUN ====
if __name__ == "__main__":
    sample_patient = {
        "subject_id": 10014729,
        "hadm_id": 28889419,
        "stay_id": 33558396,
        "gender": "F",
        "age": 72,
        "primary_diagnosis": "Right Chest pain, history of diabetes and hypertension",
        "creatinine_mg_dl": 1.69,
        "egfr_ckd_epi": 59,
        "systolic_bp_mmhg": 78,
        "map_mmhg": 49.3,
        "admission_type": "EMERGENCY",
        "potassium_mmol_l": 4.5
    }

    enhanced_context = (
        "According to KDIGO (Kidney Disease: Improving Global Outcomes) guidelines, "
        "patients with eGFR <60 mL/min/1.73m² are considered to have chronic kidney disease (CKD) "
        "and face increased risk from iodinated contrast exposure. This patient’s eGFR is 59, "
        "placing them in Stage G3a CKD. The ACR (American College of Radiology) Manual on Contrast Media "
        "advises caution and potential prophylactic measures for such patients, especially when hypotension is present "
        "(MAP 49.3 mmHg), as this further increases the risk of contrast-induced nephropathy (CIN). "
        "Hydration protocols and consideration of alternative imaging modalities are recommended before proceeding with contrast-enhanced CT."
    )

    state = AgentState(
        patient_data=sample_patient,
        enhanced_context=enhanced_context,
        protocol_db=[],
        selected_protocols=[],
        final_decision={}
    )

    app = graph.compile()
    final_state = app.invoke(state)
    print(json.dumps(final_state["final_decision"], indent=2, ensure_ascii=False))
