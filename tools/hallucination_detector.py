import re
from typing import List, Dict, Any, Tuple
from tools import renal_tools

# safe default recommendations that should never be considered hallucinations
DOMAIN_SAFE_WHITELIST = {
    "obtain informed consent",
    "monitor serum creatinine",
    "monitor creatinine",
    "post-contrast hydration",
    "document renal function",
    "use low-osmolar contrast",
    "use iso-osmolar contrast",
    "obtain labs",
    "obtain bun",
    "obtain potassium"
}

# handy regexes
NUM_RE = re.compile(r"([-+]?\d+(?:\.\d+)?)")
MAP_RE = re.compile(r"\bMAP\s*(?:=|is|:)?\s*([-+]?\d+(?:\.\d+)?)\b", flags=re.I)
EGFR_RE = re.compile(r"\begfr\b.*?([-+]?\d+(?:\.\d+)?)", flags=re.I)
CR_RE = re.compile(r"\bcreatinine\b.*?([-+]?\d+(?:\.\d+)?)", flags=re.I)
POT_RE = re.compile(r"\bpotassium\b.*?([-+]?\d+(?:\.\d+)?)", flags=re.I)
BUN_RE = re.compile(r"\bbun\b.*?([-+]?\d+(?:\.\d+)?)", flags=re.I)

def _safe_str(x) -> str:
    if x is None:
        return ""
    return str(x).lower()

def build_knowledge_base(patient: Dict[str,Any], tool_outputs: Dict[str,Any]) -> Dict[str,Any]:
    """
    Aggregate known facts from:
      - patient dict (labs, vitals)
      - tool_outputs (checks and their values/reasons)
      - derived metrics (BMI if height+weight)
    Returns a dictionary of facts for quick lookup and numeric values.
    """
    kb = {
        "raw_strings": set(),   # string facts for substring matching
        "numbers": {},          # numeric facts by canonical name
        "reasons": set()
    }

    # from patient top-level keys
    for k, v in patient.items():
        if v is None:
            continue
        s = _safe_str(v)
        kb["raw_strings"].add(s)
        # numeric capture
        if isinstance(v, (int, float)):
            kb["numbers"][k.lower()] = float(v)
        else:
            m = NUM_RE.search(str(v))
            if m:
                try:
                    kb["numbers"][k.lower()] = float(m.group(1))
                except:
                    pass

    # from tool outputs: flatten checks
    if isinstance(tool_outputs, dict) and "checks" in tool_outputs:
        for check in tool_outputs["checks"]:
            rid = check.get("id")
            reason = check.get("reason")
            val = check.get("value")
            status = _safe_str(check.get("status"))

            if rid:
                kb["raw_strings"].add(rid.lower())
            if reason:
                kb["reasons"].add(reason.lower())
                kb["raw_strings"].add(reason.lower())

                # NEW: Treat "missing / not available / no ... provided" as factual
                if any(kw in reason.lower() for kw in ["missing", "not available", "no ", "not provided"]):
                    kb["raw_strings"].add(reason.lower())
                    # also store the simplified key form for matching
                    kb["raw_strings"].add(_safe_str(rid))
            
            if val is not None:
                kb["raw_strings"].add(str(val).lower())
                try:
                    kb["numbers"][rid.lower()] = float(val)
                except Exception:
                    pass

            # NEW: Treat warning status with missing-like reason as factual
            if status == "warning" and reason:
                if any(kw in reason.lower() for kw in ["missing", "not available", "no ", "not provided"]):
                    kb["raw_strings"].add(reason.lower())

    # derived metrics: BMI from height & weight if exist
    height_m = None
    weight_kg = None
    if "height_cm" in patient:
        try:
            height_m = float(patient["height_cm"]) / 100.0
        except:
            pass
    if "height_m" in patient:
        try:
            height_m = float(patient["height_m"])
        except:
            pass
    if "weight_kg" in patient:
        try:
            weight_kg = float(patient["weight_kg"])
        except:
            pass
    if "weight_lbs" in patient and weight_kg is None:
        try:
            weight_kg = float(patient["weight_lbs"]) * 0.45359237
        except:
            pass
    if height_m and weight_kg:
        try:
            bmi = weight_kg / (height_m ** 2)
            kb["numbers"]["bmi"] = round(bmi, 1)
            kb["raw_strings"].add(f"bmi {kb['numbers']['bmi']}")
        except:
            pass

    return kb

def classify_statement(statement: str, kb: Dict[str,Any], renal_cfg: Dict[str,Any]=renal_tools.CONFIG
                       ) -> Tuple[str, str]:
    """
    Classify one LLM statement into:
      - 'supported' : directly matches facts in KB or numeric matches
      - 'inferred' : derivable from KB by simple rules (e.g., egfr high -> low renal risk)
      - 'domain_safe' : present in whitelist of safe recommendations
      - 'contradicted' : conflicts with KB numeric values
      - 'unverified' : neither supported nor contradicted nor inferable
    Returns (classification, rationale)
    """
    s = statement.strip().lower()

    # 1) domain-safe quick check (whitelist)
    for w in DOMAIN_SAFE_WHITELIST:
        if w in s:
            return "domain_safe", f"matches safe-whitelist '{w}'"

    # 2) exact substring match with KB raw strings or reasons
    for fact in kb["raw_strings"]:
        if fact and fact in s:
            return "supported", f"matches known fact '{fact}'"

    for reason in kb.get("reasons", set()):
        if reason and reason in s:
            return "supported", f"matches tool reason '{reason}'"

    # 3) numeric checks — look for explicit numbers in statement and compare with KB numbers
    # eGFR
    m = EGFR_RE.search(s)
    if m:
        try:
            v = float(m.group(1))
            if "egfr_ckd_epi" in kb["numbers"]:
                known = kb["numbers"]["egfr_ckd_epi"]
                # allow some tolerance
                if abs(known - v) <= max(1.0, 0.05*known):
                    return "supported", f"egfr {v} matches known {known}"
                else:
                    return "contradicted", f"egfr {v} contradicts known {known}"
            else:
                return "unverified", f"egfr {v} not present in KB"
        except:
            pass

    # MAP checks
    m = MAP_RE.search(s)
    if m:
        try:
            v = float(m.group(1))
            if "map_mmhg" in kb["numbers"]:
                known = kb["numbers"]["map_mmhg"]
                if abs(known - v) <= 3.0:
                    return "supported", f"MAP {v} matches known {known}"
                else:
                    return "contradicted", f"MAP {v} contradicts known {known}"
            else:
                return "unverified", f"MAP {v} not present in KB"
        except:
            pass

    # creatinine mention
    m = CR_RE.search(s)
    if m:
        try:
            v = float(m.group(1))
            if "creatinine_mg_dl" in kb["numbers"]:
                known = kb["numbers"]["creatinine_mg_dl"]
                if abs(known - v) <= 0.05*max(1.0, known):
                    return "supported", f"creatinine {v} matches known {known}"
                else:
                    return "contradicted", f"creatinine {v} contradicts known {known}"
            else:
                return "unverified", f"creatinine {v} not present in KB"
        except:
            pass

    # potassium mention
    m = POT_RE.search(s)
    if m:
        try:
            v = float(m.group(1))
            if "potassium_meq_l" in kb["numbers"]:
                known = kb["numbers"]["potassium_meq_l"]
                if abs(known - v) <= 0.1:
                    return "supported", f"potassium {v} matches known {known}"
                else:
                    return "contradicted", f"potassium {v} contradicts known {known}"
            else:
                return "unverified", f"potassium {v} not present in KB"
        except:
            pass

    # 4) simple inference rules: e.g., "low renal risk" derivable from egfr > safe threshold
    if "low renal risk" in s or "low renal" in s:
        egfr_known = kb["numbers"].get("egfr_ckd_epi")
        if egfr_known is not None:
            if egfr_known >= renal_cfg.get("egfr_safe", 45.0):
                return "inferred", f"low renal risk inferred from egfr {egfr_known}"
            elif egfr_known >= renal_cfg.get("egfr_caution", 30.0):
                return "inferred", f"low/borderline renal risk inferred from egfr {egfr_known}"
            else:
                return "contradicted", f"low renal risk claim contradicted by egfr {egfr_known}"
        else:
            return "unverified", "low renal risk claim but egfr not present"

    # hemodynamic instability inference
    if "hemodynam" in s or "hypotens" in s or "unstable" in s:
        map_known = kb["numbers"].get("map_mmhg")
        sys_known = kb["numbers"].get("systolic_bp_mmhg")
        # use MAP threshold 60 from renal_tools CONFIG
        map_thresh = renal_cfg.get("hemodynamics", {}).get("map_warning_threshold", 60)
        if map_known is not None:
            if map_known < map_thresh:
                return "supported", f"hemodynamic instability supported by MAP {map_known}"
            else:
                return "contradicted", f"hemodynamic instability claim contradicted by MAP {map_known}"
        elif sys_known is not None:
            if sys_known < 90:
                return "inferred", f"low systolic BP ({sys_known}) suggests instability"
            else:
                return "unverified", "hemodynamic claim not supported by known vitals"
    
    # 5) otherwise: unverified
    return "unverified", "no direct or inferred support found"


def detect_and_score_statements(llm_parsed: Dict[str,Any], patient: Dict[str,Any], tool_outputs: Dict[str,Any]
                               ) -> Dict[str,Any]:
    """
    Analyze each statement in LLM's issues + recommendations.
    Returns:
      {
        "statement_assessments": [
            {"text":..., "type":"issue|recommendation", "classification":..., "rationale":...}
        ],
        "counts": {"supported":..., "inferred":..., "domain_safe":..., "unverified":..., "contradicted":...},
        "hallucination_score": 0.0,   # 0..1, higher = more hallucination
        "recommendation": {"force_revision": bool, "confidence_adjustment": float}
      }
    """
    kb = build_knowledge_base(patient, tool_outputs.get("renal") if isinstance(tool_outputs, dict) else tool_outputs)
    stmts = []
    counts = {"supported":0, "inferred":0, "domain_safe":0, "unverified":0, "contradicted":0}

    # combine issues + recommendations into one list of statements to assess
    all_texts = []
    for it in llm_parsed.get("issues", []):
        if isinstance(it, str):
            all_texts.append( ("issue", it) )
    for it in llm_parsed.get("recommendations", []):
        if isinstance(it, str):
            all_texts.append( ("recommendation", it) )

    # classify each
    for typ, text in all_texts:
        classification, rationale = classify_statement(text, kb)
        counts[classification] += 1
        stmts.append({
            "text": text,
            "type": typ,
            "classification": classification,
            "rationale": rationale
        })

    # compute hallucination_score:
    # weight contradicted = 1.0, unverified = 0.5, inferred/domain_safe/supported = 0.0 penalty
    total = max(1, sum(counts.values()))
    penalty = counts["contradicted"] * 1.0 + counts["unverified"] * 0.5
    hallucination_score = min(1.0, penalty / total)

    # determine recommended action:
    force_revision = False
    confidence_adj = 0.0  # amount to reduce confidence (0..1)
    if counts["contradicted"] > 0:
        force_revision = True
        confidence_adj = max(confidence_adj, 0.5)
    elif hallucination_score >= 0.3:
        # many unverified -> require human check
        force_revision = True
        confidence_adj = max(confidence_adj, 0.25)
    elif counts["unverified"] > 0:
        # minor unverified content -> small confidence hit
        confidence_adj = max(confidence_adj, 0.1)

    return {
        "statement_assessments": stmts,
        "counts": counts,
        "hallucination_score": round(hallucination_score, 3),
        "recommendation": {
            "force_revision": force_revision,
            "confidence_reduction": round(confidence_adj, 3)
        },
        "knowledge_base_summary": {
            "numbers": kb["numbers"],
            "raw_strings_sample": list(list(kb["raw_strings"])[:10])
        }
    }
