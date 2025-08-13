import os
from datetime import datetime, timedelta

CONFIG = {
    "egfr_safe": 45.0,
    "egfr_caution": 30.0,
    "creatinine_attention": 1.1,
    "creatinine_high": 1.5,
    "potassium_critical": 5.5,
    "potassium_warning": 5.0,
    "lab_stale_hours": 48,
    "bun_cr_ratio_warning": 20.0
}

# Assign importance levels so missing non-critical fields won't reduce confidence
# critical → must have, important → nice to have, optional → not essential
CHECK_PRIORITIES = {
    "egfr_ckd_epi": "critical",
    "creatinine_mg_dl": "critical",
    "bun_mg_dl": "important",
    "potassium_mmol_l": "important",
    "bmi": "optional",
    "labs_ts": "optional"
}

def now_iso():
    return datetime.utcnow().isoformat() + "Z"

def check_egfr(patient):
    egfr = patient.get("egfr_ckd_epi")
    out = {"id": "egfr_ckd_epi", "units": "ml/min/1.73m2", "timestamp": now_iso(), "priority": CHECK_PRIORITIES["egfr_ckd_epi"]}
    if egfr is None:
        out.update({"value": None, "status": "missing", "reason": "eGFR not available"})
        return out
    egfr = float(egfr)
    if egfr >= CONFIG["egfr_safe"]:
        out.update({"value": egfr, "status": "pass", "reason": "Low renal risk for contrast."})
    elif egfr >= CONFIG["egfr_caution"]:
        out.update({"value": egfr, "status": "warning", "reason": "Borderline renal function."})
    else:
        out.update({"value": egfr, "status": "fail", "reason": "High renal risk; contrast caution."})
    return out

def check_creatinine(patient):
    cr = patient.get("creatinine_mg_dl")
    out = {"id": "creatinine_mg_dl", "units": "mg/dL", "timestamp": now_iso(), "priority": CHECK_PRIORITIES["creatinine_mg_dl"]}
    if cr is None:
        out.update({"value": None, "status": "missing", "reason": "Creatinine not available"})
        return out
    cr = float(cr)
    if cr >= CONFIG["creatinine_high"]:
        out.update({"value": cr, "status": "fail", "reason": "Creatinine significantly elevated."})
    elif cr >= CONFIG["creatinine_attention"]:
        out.update({"value": cr, "status": "warning", "reason": "Creatinine mildly elevated."})
    else:
        out.update({"value": cr, "status": "pass", "reason": "Creatinine within normal range."})
    return out

def check_bun_ratio(patient):
    bun = patient.get("bun_mg_dl")
    cr = patient.get("creatinine_mg_dl")
    out = {"id": "bun_creatinine_ratio", "timestamp": now_iso(), "priority": CHECK_PRIORITIES["bun_mg_dl"]}
    if bun is None or cr is None:
        out.update({"value": None, "status": "missing", "reason": "Missing BUN or creatinine for ratio."})
        return out
    ratio = round(float(bun) / float(cr), 1)
    out["value"] = ratio
    if ratio >= CONFIG["bun_cr_ratio_warning"]:
        out.update({"status": "warning", "reason": "BUN/Cr ratio suggests prerenal azotemia."})
    else:
        out.update({"status": "pass", "reason": "BUN/Cr ratio normal."})
    return out

def check_potassium(patient):
    k = patient.get("potassium_mmol_l")
    out = {"id": "potassium_mmol_l", "units": "mEq/L", "timestamp": now_iso(), "priority": CHECK_PRIORITIES["potassium_mmol_l"]}
    if k is None:
        out.update({"value": None, "status": "missing", "reason": "Potassium not available"})
        return out
    k = float(k)
    if k >= CONFIG["potassium_critical"]:
        out.update({"value": k, "status": "fail", "reason": "Critical hyperkalemia."})
    elif k >= CONFIG["potassium_warning"]:
        out.update({"value": k, "status": "warning", "reason": "Mild hyperkalemia."})
    else:
        out.update({"value": k, "status": "pass", "reason": "Potassium normal."})
    return out

def check_bmi_creatinine(patient):
    bmi = patient.get("bmi")
    cr = patient.get("creatinine_mg_dl")
    out = {"id": "bmi_creatinine_correlation", "timestamp": now_iso(), "priority": CHECK_PRIORITIES["bmi"]}
    if bmi is None or cr is None:
        out.update({"value": None, "status": "missing", "reason": "Missing BMI or creatinine."})
        return out
    bmi = float(bmi); cr = float(cr)
    out["value"] = {"bmi": bmi, "creatinine": cr}
    if bmi < 18.5 and cr < 0.8:
        out.update({"status": "warning", "reason": "Low BMI with low creatinine — may underestimate renal impairment."})
    elif bmi > 35 and cr > 1.2:
        out.update({"status": "warning", "reason": "High BMI with elevated creatinine — interpret with caution."})
    else:
        out.update({"status": "pass", "reason": "BMI and creatinine concordant."})
    return out

def check_lab_staleness(patient):
    labs_ts = patient.get("labs_ts", {})
    out = {"id": "lab_staleness", "timestamp": now_iso(), "value": {}, "priority": CHECK_PRIORITIES["labs_ts"]}
    stale_threshold = timedelta(hours=CONFIG["lab_stale_hours"])
    statuses = []
    now = datetime.utcnow()
    for lab_name, ts in labs_ts.items():
        try:
            t = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            age = now - t
            out["value"][lab_name] = {"ts": ts, "age_hours": round(age.total_seconds() / 3600, 1)}
            if age > stale_threshold:
                out["value"][lab_name]["status"] = "stale"
                statuses.append("stale")
            else:
                out["value"][lab_name]["status"] = "fresh"
        except:
            out["value"][lab_name] = {"ts": ts, "status": "unknown"}
            statuses.append("unknown")
    if not labs_ts:
        out.update({"status": "missing", "reason": "No lab timestamps provided."})
    elif "stale" in statuses:
        out.update({"status": "warning", "reason": "Some labs are stale."})
    else:
        out.update({"status": "pass", "reason": "All labs recent."})
    return out

def run_renal_tool(patient: dict) -> dict:
    checks = [
        check_egfr(patient),
        check_creatinine(patient),
        check_bun_ratio(patient),
        check_potassium(patient),
        check_bmi_creatinine(patient),
        check_lab_staleness(patient)
    ]
    # Determine overall status using priority
    statuses = []
    for c in checks:
        if c["status"] == "fail":
            statuses.append("fail")
        elif c["status"] == "warning" and c["priority"] in ("critical", "important"):
            statuses.append("warning")
        elif c["status"] == "missing" and c["priority"] == "critical":
            statuses.append("fail")  # Missing critical = fail
        elif c["status"] == "missing" and c["priority"] == "important":
            statuses.append("warning")  # Missing important = warning
        # missing optional does not affect
    if "fail" in statuses:
        overall = "fail"
    elif "warning" in statuses:
        overall = "warning"
    else:
        overall = "pass"

    confidence = 0.92
    if "fail" in statuses:
        confidence -= 0.4
    elif "warning" in statuses:
        confidence -= 0.2

    return {
        "tool_id": "renal_tool_v2",
        "checks": checks,
        "summary": {
            "overall": overall,
            "reasons": [c["reason"] for c in checks if c["status"] not in ("pass", "optional")],
            "confidence": round(confidence, 2)
        },
        "metadata": {"version": "2.0.0", "run_ts": now_iso()}
    }
