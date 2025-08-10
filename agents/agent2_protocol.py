
import os
import time
import json
import math
import random
import logging
from typing import Dict, List, Tuple

import requests
from bs4 import BeautifulSoup


try:
    from google import genai
    from google.genai import types as genai_types
    GEMINI_SDK_AVAILABLE = True
except Exception:
    genai = None
    genai_types = None
    GEMINI_SDK_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    EMB_MODEL_AVAIL = True
except Exception:
    SentenceTransformer = None
    np = None
    EMB_MODEL_AVAIL = False


try:
    import hnswlib
    HNSW_AVAILABLE = True
except Exception:
    hnswlib = None
    HNSW_AVAILABLE = False


logging.basicConfig(level=logging.INFO)
log = logging.getLogger("protocol_agent")

GUIDELINE_URLS = {
    "ACR_contrast_manual": "https://www.acr.org/clinical-resources/clinical-tools-and-reference/contrast-manual",
    "KDIGO_guidelines": "https://kdigo.org/guidelines/",
    "NICE_NG203_CHRONIC_KIDNEY_DISEASE": "https://www.nice.org.uk/guidance/ng203",
}
CACHE_DIR = "./guideline_cache"
os.makedirs(CACHE_DIR, exist_ok=True)


GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
GEMINI_DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")
GEMINI_HTTP_TIMEOUT = int(os.getenv("GEMINI_HTTP_TIMEOUT", "30"))
GEMINI_MAX_RETRIES = int(os.getenv("GEMINI_MAX_RETRIES", "5"))


def fetch_and_cache(url: str, force_refresh: bool = False, timeout: int = 10) -> str:
    fname = os.path.join(CACHE_DIR, (url.replace("://", "_").replace("/", "_")[:200]) + ".html")
    if (not force_refresh) and os.path.exists(fname):
        with open(fname, "r", encoding="utf-8") as f:
            return f.read()
    resp = requests.get(url, timeout=timeout, headers={"User-Agent": "protocol-agent/1.0"})
    resp.raise_for_status()
    html = resp.text
    with open(fname, "w", encoding="utf-8") as f:
        f.write(html)
    return html

def extract_text_blocks(html: str, max_block_chars: int = 1600) -> List[str]:
    soup = BeautifulSoup(html, "html.parser")
    for s in soup(["script", "style", "noscript"]):
        s.decompose()
    blocks = []
    for tag in soup.find_all(["h1","h2","h3","h4","p","li"]):
        txt = tag.get_text(separator=" ", strip=True)
        if not txt:
            continue
        while len(txt) > max_block_chars:
            blocks.append(txt[:max_block_chars])
            txt = txt[max_block_chars:]
        if txt:
            blocks.append(txt)
    dedup = []
    seen = set()
    for b in blocks:
        k = b.strip()[:200]
        if k not in seen:
            dedup.append(b.strip())
            seen.add(k)
    return dedup

class GuidelineIndex:
    """
    Stores text fragments and provides nearest-neighbour search over embeddings.
    Tries to use hnswlib (fast ANN). If unavailable, uses a numpy brute-force cosine search.
    """
    def __init__(self, emb_model_name: str = "all-MiniLM-L6-v2"):
        self.blocks: List[str] = []
        self.ids: List[str] = []
        self.embeddings = None  # numpy array NxD
        self.hnsw_index = None
        self.dim = None
        self.emb_model = None
        if EMB_MODEL_AVAIL:
            self.emb_model = SentenceTransformer(emb_model_name)
        else:
            log.warning("SentenceTransformer not available; GuidelineIndex will not build embeddings.")

    def add_fragments(self, fragments: List[Tuple[str,str]]):
        """
        fragments: list of (id, text)
        We rebuild the index from scratch on each call (safe for small-medium corpora).
        """
        new_texts = [t for (_id, t) in fragments]
        new_ids = [_id for (_id, t) in fragments]
        self.blocks = new_texts
        self.ids = new_ids
        self._build_index()

    def _build_index(self):
        if not EMB_MODEL_AVAIL or not self.blocks:
            self.embeddings = None
            self.hnsw_index = None
            return

        # compute embeddings (dense float32 matrix)
        embs = self.emb_model.encode(self.blocks, convert_to_numpy=True, show_progress_bar=False)
        # ensure float32
        embs = embs.astype('float32')
        # normalize rows for cosine similarity
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        norms[norms==0] = 1.0
        embs = embs / norms
        self.embeddings = embs
        self.dim = embs.shape[1]

        # Try to build an HNSW index if available
        if HNSW_AVAILABLE:
            try:
                # recreate index
                index = hnswlib.Index(space='cosine', dim=self.dim)
                index.init_index(max_elements=len(embs), ef_construction=200, M=16)
                index.add_items(embs, ids=list(range(len(embs))))
                index.set_ef(50)  # query-time parameter
                self.hnsw_index = index
                log.info("Built hnswlib index for guideline fragments.")
                return
            except Exception as e:
                log.warning(f"hnswlib index build failed, falling back to brute-force: {e}")
                self.hnsw_index = None

        # fallback: no ANN lib available -> keep embeddings for brute-force search
        log.info("Using numpy brute-force search (no hnswlib).")

    def query(self, q: str, k: int = 4) -> List[Tuple[str,float,str]]:
        """
        Returns list of (id, score, text) with score in [0,1] where higher is more similar.
        """
        if not EMB_MODEL_AVAIL or not self.blocks:
            # no indexing possible, return empty
            return []

        # embed the query
        q_emb = self.emb_model.encode([q], convert_to_numpy=True)[0].astype('float32')
        q_norm = np.linalg.norm(q_emb)
        if q_norm == 0:
            q_norm = 1.0
        q_emb = q_emb / q_norm

        if self.hnsw_index is not None:
            # hnswlib's 'knn_query' returns labels and distances (for 'cosine', distances are 1 - cosine_similarity)
            labels, distances = self.hnsw_index.knn_query(q_emb, k=k)
            out = []
            for lbl, dist in zip(labels[0], distances[0]):
                if int(lbl) < 0 or int(lbl) >= len(self.blocks):
                    continue
                sim = 1.0 - float(dist)  # convert to similarity (approx)
                out.append((self.ids[int(lbl)], float(sim), self.blocks[int(lbl)]))
            # sort by sim descending
            out.sort(key=lambda x: -x[1])
            return out
        else:
            # brute-force cosine (dot product of normalized vectors)
            sims = np.dot(self.embeddings, q_emb)  # shape (N,)
            # get top-k indices
            if k >= len(sims):
                idxs = np.argsort(-sims)
            else:
                # partial sort for speed
                idxs = np.argpartition(-sims, k-1)[:k]
                idxs = idxs[np.argsort(-sims[idxs])]
            out = []
            for i in idxs:
                out.append((self.ids[int(i)], float(sims[int(i)]), self.blocks[int(i)]))
            return out

def _sleep_with_jitter(base_seconds: float, attempt: int):
    backoff = base_seconds * (2 ** attempt)
    jitter = random.uniform(0, backoff * 0.25)
    to_sleep = min(60.0, backoff + jitter)
    log.info(f"Sleeping for {to_sleep:.1f}s before retry (attempt {attempt})")
    time.sleep(to_sleep)

def _call_gemini_rest(prompt: str, model: str = GEMINI_DEFAULT_MODEL, timeout: int = GEMINI_HTTP_TIMEOUT, max_retries: int = GEMINI_MAX_RETRIES) -> str:
    assert GEMINI_API_KEY, "GEMINI_API_KEY environment variable not set for REST fallback"
    endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": GEMINI_API_KEY,
    }
    body = {
        "contents":[ {"parts":[{"text": prompt}]} ],
        "maxOutputTokens": 512
    }

    attempt = 0
    while True:
        try:
            r = requests.post(endpoint, headers=headers, json=body, timeout=timeout)
        except requests.RequestException as e:
            attempt += 1
            if attempt > max_retries:
                raise
            _sleep_with_jitter(1.0, attempt)
            continue

        if r.status_code == 200:
            j = r.json()
            # try to extract the text in several common formats
            # docs may vary; attempt best-effort extraction
            out = ""
            if isinstance(j, dict):
                # new-style: j.get("candidates") or j.get("output")
                if "candidates" in j:
                    for c in j["candidates"]:
                        out += c.get("content", "") + " "
                    return out.strip()
                if "output" in j and isinstance(j["output"], dict):
                    # look for textual pieces
                    if "candidates" in j["output"]:
                        for c in j["output"]["candidates"]:
                            out += c.get("content", "") + " "
                        return out.strip()
            return json.dumps(j)
        elif r.status_code == 429:
            retry_after = r.headers.get("Retry-After")
            if retry_after:
                try:
                    wait = float(retry_after)
                except Exception:
                    wait = 5.0
                log.warning(f"REST 429 received; respecting Retry-After={wait}s")
                time.sleep(wait)
            else:
                attempt += 1
                if attempt > max_retries:
                    r.raise_for_status()
                _sleep_with_jitter(1.0, attempt)
            continue
        elif 500 <= r.status_code < 600:
            attempt += 1
            if attempt > max_retries:
                r.raise_for_status()
            _sleep_with_jitter(1.0, attempt)
            continue
        else:
            r.raise_for_status()

def call_gemini(prompt: str, model: str = GEMINI_DEFAULT_MODEL, timeout: int = GEMINI_HTTP_TIMEOUT, max_retries: int = GEMINI_MAX_RETRIES) -> str:
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY not set")

    if GEMINI_SDK_AVAILABLE:
        attempt = 0
        while True:
            try:
                client = genai.Client(api_key=GEMINI_API_KEY)
                resp = client.models.generate_content(model=model, contents=prompt)
                # Try common resp shapes
                if hasattr(resp, "text") and resp.text:
                    return resp.text
                if isinstance(resp, dict) and "output" in resp:
                    return json.dumps(resp["output"])
                return str(resp)
            except Exception as e:
                s = str(e).lower()
                if "429" in s or "rate limit" in s or "quota" in s or "too many requests" in s:
                    attempt += 1
                    if attempt > max_retries:
                        log.exception("Gemini SDK: rate-limited and max retries exceeded")
                        break
                    _sleep_with_jitter(0.8, attempt)
                    continue
                else:
                    log.warning(f"Gemini SDK call failed ({e}), falling back to REST once")
                    break

    return _call_gemini_rest(prompt=prompt, model=model, timeout=timeout, max_retries=max_retries)

def compose_decision(patient: Dict, retrieved: List[Tuple[str,float,str]]) -> Dict:
    age = patient.get("age")
    eGFR = patient.get("egfr_ckd_epi")
    sbp = patient.get("systolic_bp_mmhg")
    map_mmhg = patient.get("map_mmhg")
    primary = (patient.get("primary_diagnosis","") or "").lower()
    admission_type = (patient.get("admission_type","") or "").lower()

    flags = []
    rationale = []
    if eGFR is None:
        flags.append("no_egfr")
        rationale.append("No eGFR available; obtain recent creatinine/eGFR if feasible.")
    else:
        if eGFR >= 60:
            flags.append("low_renal_risk")
            rationale.append(f"eGFR {eGFR} — lower renal risk.")
        elif 30 <= eGFR < 60:
            flags.append("moderate_renal_risk")
            rationale.append(f"eGFR {eGFR} — moderate renal risk; consider hydration/minimize contrast.")
        else:
            flags.append("high_renal_risk")
            rationale.append(f"eGFR {eGFR} — high renal risk; consult nephrology if time allows.")

    if map_mmhg is not None and map_mmhg < 65:
        flags.append("hemodynamic_instability")
        rationale.append(f"MAP {map_mmhg} mmHg — haemodynamic compromise.")

    emergent_indications = ["aorta","aortic","thoracic aorta","massive bleed","active hemorrhage","unstable chest trauma"]
    emergent = any(tok in primary for tok in emergent_indications)
    if emergent:
        flags.append("emergent_imaging_required")
        rationale.append("Life/organ-threatening diagnosis; imaging benefit may outweigh renal risk.")

    actions = []
    if "emergent_imaging_required" in flags and "low_renal_risk" in flags:
        actions.append("Proceed with emergent contrast-enhanced CT/CTA immediately.")
        actions.append("Notify radiology and document renal function and contrast volume.")
    elif "emergent_imaging_required" in flags and "hemodynamic_instability" in flags:
        actions.append("Stabilize rapidly; do not delay life-saving imaging for routine labs if clinically necessary.")
        actions.append("Minimize contrast volume; alert interventional team and document renal status.")
    else:
        if "high_renal_risk" in flags or "moderate_renal_risk" in flags:
            if ("inpatient" in admission_type) or patient.get("first_careunit"):
                actions.append("If non-urgent, consider IV volume expansion (e.g., 0.9% NaCl) prior to contrast for inpatients.")
                actions.append("If eGFR < 30 consider nephrology consult.")
            else:
                actions.append("Encourage oral hydration for outpatients; consider point-of-care creatinine if no recent result.")
            actions.append("Consider alternative non-contrast imaging if diagnostically adequate.")
        else:
            actions.append("Proceed with indicated contrast imaging; use lowest effective contrast volume and document.")

    actions.append("Document discussion of renal risk and obtain consent when practicable; monitor creatinine post-contrast if indicated.")

    snippets = [{"id": rid, "score": score, "text": txt} for (rid, score, txt) in retrieved]

    decision_lines = []
    decision_lines.append(f"Patient: age {age}, primary: {patient.get('primary_diagnosis')}.")
    decision_lines.append("Actionable summary:")
    for a in actions:
        decision_lines.append("- " + a)
    decision_text = "\n".join(decision_lines)

    return {
        "decision_text": decision_text,
        "rationale": rationale,
        "flags": flags,
        "guideline_snippets": snippets,
        "timestamp": time.time()
    }

def run_protocol_agent(patient: Dict, guideline_urls: Dict = GUIDELINE_URLS,
                       use_gemini_polish: bool = True,
                       gemini_model: str = GEMINI_DEFAULT_MODEL) -> Dict:
    # Fetch guidelines
    fragments = []
    for key, url in guideline_urls.items():
        try:
            html = fetch_and_cache(url)
            blocks = extract_text_blocks(html, max_block_chars=1200)
            for i, b in enumerate(blocks):
                fragments.append((f"{key}#{i}", b))
        except Exception as e:
            log.warning(f"Failed to fetch {url}: {e}")
            fragments.append((f"{key}#error", f"Failed to fetch {url}: {e}"))

    # Build index & retrieve snippets using only fixed parameters
    idx = GuidelineIndex()
    idx.add_fragments(fragments)
    q_terms = f"contrast administration eGFR {patient.get('egfr_ckd_epi')} MAP {patient.get('map_mmhg')} hypotension hydration nephrology consult"
    retrieved = idx.query(q_terms, k=6)

    # Compose decision using fixed params
    decision_struct = compose_decision(patient, retrieved)

    # Add Gemini polish with full JSON (extra fields included)
    if use_gemini_polish:
        if GEMINI_API_KEY:
            try:
                prompt = (
                    "You are a concise assistant for clinical imaging protocol routing. "
                    "Given the patient JSON (which may contain additional fields beyond the core ones) "
                    "and the guideline snippets, produce a 3–5 sentence instruction to the imaging agent "
                    "that includes: recommended imaging choice, immediate safety flags, necessary pre-imaging steps, "
                    "and who to call. Use extra fields if they are relevant.\n\n"
                    f"PATIENT_JSON: {json.dumps(patient, indent=2)}\n\n"
                    "GUIDELINE_SNIPPETS:\n" +
                    "\n---\n".join([s["text"] for s in decision_struct["guideline_snippets"][:4]])
                )
                polished = call_gemini(prompt, model=gemini_model)
                decision_struct["decision_text_polished"] = polished
            except Exception as e:
                log.exception("Gemini polishing failed; continuing without polish")
                decision_struct.setdefault("notes", []).append(f"Gemini polish failed: {e}")
        else:
            decision_struct.setdefault("notes", []).append("GEMINI_API_KEY not set; skipping Gemini polish")

    return decision_struct


if __name__ == "__main__":
    sample = {
        "subject_id":10014729,"hadm_id":28889419,"stay_id":33558396,"gender":"F","age":72,
        "primary_diagnosis":"Right Chest pain, history of diabetes and hypertension","creatinine_mg_dl":1.69,"egfr_ckd_epi":59,
        "systolic_bp_mmhg":78,"map_mmhg":49.3,"admission_type":"EMERGENCY" , "potassium_mmol_l":4.5,
    }
    out = run_protocol_agent(sample, use_gemini_polish=False)
    print(json.dumps(out, indent=2))
