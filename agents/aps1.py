import os
import json
import time
import requests
from typing import Dict, List, Any, Optional
from utils.vector_search import load_or_create_index, vector_search, build_vector_index
from google import genai
from dotenv import load_dotenv

try:
    from serpapi import GoogleSearch
    SERPAPI_AVAILABLE = True
except ImportError:
    SERPAPI_AVAILABLE = False

load_dotenv()

PROTOCOL_DB_PATH = "data/protocol_database.json"
VECTOR_INDEX_PATH = "data/protocol_index.bin"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MODEL = "gemini-2.5-flash-lite"
RELEVANCE_THRESHOLD = 0.6

client = genai.Client(api_key=GEMINI_API_KEY)

def enhanced_search_medical_sources(patient_data: Dict) -> List[Dict]:
    diagnosis = patient_data.get("primary_diagnosis", "")
    serpapi_key = os.getenv("SERPAPI_API_KEY") or os.getenv("SERP_API_KEY")
    
    if not serpapi_key:
        print("SERPAPI_API_KEY not found in environment")
        return []
    
    search_results = []
    
    try:
        queries = [
            f'"{diagnosis}" CT protocol imaging contrast radiology',
            f'"{diagnosis}" imaging guidelines ACR appropriateness',
            f'"{diagnosis}" contrast enhanced CT MRI protocol',
        ]
        
        for query in queries:
            print(f"SerpAPI search: {query}")
            
            search = GoogleSearch({
                "q": query,
                "api_key": serpapi_key,
                "num": 5,
                "hl": "en",
                "gl": "us"
            })
            
            results = search.get_dict()
            organic_results = results.get("organic_results", [])
            
            for result in organic_results:
                title = result.get("title", "")
                url = result.get("link", "")
                snippet = result.get("snippet", "")
                
                medical_domains = [
                    "acr.org", "radiologyinfo.org", "radiopaedia.org",
                    "ncbi.nlm.nih.gov", "pubmed.ncbi.nlm.nih.gov",
                    "nejm.org", "radiology.rsna.org", "ajronline.org",
                    "springer.com", "wiley.com", "elsevier.com",
                    "nice.org.uk", "kdigo.org", "uptodate.com"
                ]
                
                is_medical = (
                    any(domain in url.lower() for domain in medical_domains) or
                    any(keyword in title.lower() for keyword in ["radiology", "imaging", "ct", "mri", "protocol", "contrast"]) or
                    any(keyword in snippet.lower() for keyword in ["imaging", "radiology", "protocol", "contrast", "ct scan"])
                )
                
                if is_medical and len(snippet) > 50:
                    search_results.append({
                        "title": title,
                        "url": url,
                        "snippet": snippet,
                        "source": "serpapi_search",
                        "query_used": query,
                        "full_content": snippet
                    })
            
            time.sleep(0.5)
        
        seen_urls = set()
        unique_results = []
        for result in search_results:
            url = result.get("url", "")
            if url not in seen_urls:
                seen_urls.add(url)
                unique_results.append(result)
        
        print(f"Found {len(unique_results)} unique medical sources")
        return unique_results[:5]
        
    except Exception as e:
        print(f"SerpAPI search failed: {e}")
        return []

def check_protocol_relevance(patient_data: Dict, protocols: List[Dict]) -> float:
    if not protocols:
        return 0.0
    
    diagnosis = patient_data.get("primary_diagnosis", "").lower()
    
    clinical_keywords = {
        "atlantoaxial": ["cervical", "spine", "fracture", "instability"],
        "odontoid": ["cervical", "spine", "fracture", "trauma"],
        "cauda equina": ["lumbar", "spine", "compression", "emergency"],
        "spinal cord ischemia": ["spine", "vascular", "emergency"],
        "arteriovenous malformation": ["vascular", "angiography", "hemorrhage"],
        "dural arteriovenous fistula": ["vascular", "angiography", "spine"],
        "thoracic aorta": ["aortic", "vascular", "angiography", "trauma", "chest"],
        "aorta": ["aortic", "vascular", "angiography", "trauma"],
        "pulmonary embolism": ["chest", "angiography", "contrast"],
        "brain": ["head", "neurologic", "contrast"],
        "malignant neoplasm": ["oncology", "contrast", "staging"]
    }
    
    max_score = 0.0
    best_protocol = None
    
    for protocol in protocols:
        score = 0.0
        protocol_name = protocol.get("name", "").lower()
        indications = " ".join(protocol.get("indications", [])).lower()
        description = protocol.get("description", "").lower()
        
        protocol_text = f"{protocol_name} {indications} {description}"
        
        diagnosis_words = diagnosis.split()
        for word in diagnosis_words:
            if len(word) > 3:
                if word in protocol_text:
                    score += 0.3
        
        for condition, keywords in clinical_keywords.items():
            if condition in diagnosis:
                for keyword in keywords:
                    if keyword in protocol_text:
                        score += 0.2
        
        anatomical_regions = {
            "cervical": ["neck", "cervical", "carotid"],
            "thoracic": ["chest", "thoracic", "aorta"],
            "lumbar": ["abdomen", "lumbar", "pelvis"],
            "spinal": ["spine", "spinal"],
            "head": ["head", "brain", "skull"]
        }
        
        for region, synonyms in anatomical_regions.items():
            if region in diagnosis:
                for synonym in synonyms:
                    if synonym in protocol_text:
                        score += 0.15
        
        admission_type = patient_data.get("admission_type", "").upper()
        if "EMERG" in admission_type:
            if any(word in protocol_text for word in ["trauma", "emergency", "acute"]):
                score += 0.1
        
        spine_conditions = ["atlantoaxial", "odontoid", "cauda equina", "spinal", "cervical", "lumbar", "thoracic"]
        if any(condition in diagnosis for condition in spine_conditions):
            if "head" in protocol_name and "spine" not in protocol_text:
                score -= 0.5
        
        if score > max_score:
            max_score = score
            best_protocol = protocol.get("protocol_id")
    
    print(f"Best match: {best_protocol} (score: {max_score:.3f})")
    return max_score

def convert_search_results_to_protocols(search_results: List[Dict], patient_data: Dict) -> List[Dict]:
    if not search_results:
        return []
    
    diagnosis = patient_data.get("primary_diagnosis", "Unknown condition")
    new_protocols = []
    
    for i, result in enumerate(search_results):
        title = result.get("title", f"Protocol_{i}")
        url = result.get("url", "")
        
        if "acr.org" in url:
            protocol_id = f"ACR_{diagnosis.replace(' ', '_').upper()[:20]}"
        elif "radiopaedia" in url:
            protocol_id = f"RADIOPAEDIA_{diagnosis.replace(' ', '_').upper()[:15]}"
        elif "ncbi" in url or "pubmed" in url:
            protocol_id = f"PUBMED_{diagnosis.replace(' ', '_').upper()[:18]}"
        else:
            clean_title = "".join(c.upper() if c.isalnum() else "_" for c in title[:25])
            protocol_id = f"WEB_{clean_title}"
        
        domain = ""
        if url:
            try:
                from urllib.parse import urlparse
                domain = urlparse(url).netloc
            except:
                domain = url
        
        protocol = {
            "protocol_id": protocol_id,
            "name": title[:100],
            "indications": [diagnosis.lower()],
            "description": result.get("snippet", "Medical imaging protocol found via search")[:500],
            "contrast_timing": "Refer to source for specific timing protocols",
            "contrast_dose": "Follow institutional guidelines and source recommendations", 
            "renal_safety_notes": "Standard renal safety protocols apply - verify eGFR, consider hydration, follow ACR guidelines",
            "references": [url] if url else [],
            "source": f"serpapi_search_from_{domain}",
            "search_context": {
                "query_used": result.get("query_used", ""),
                "snippet": result.get("snippet", ""),
                "search_timestamp": time.time(),
                "domain": domain
            },
            "quality_score": calculate_source_quality(url, title, result.get("snippet", ""))
        }
        
        new_protocols.append(protocol)
    
    new_protocols.sort(key=lambda x: x.get("quality_score", 0), reverse=True)
    
    return new_protocols

def calculate_source_quality(url: str, title: str, snippet: str) -> float:
    score = 0.0
    
    premium_domains = {
        "acr.org": 1.0,
        "radiologyinfo.org": 0.9,
        "radiopaedia.org": 0.9,
        "ncbi.nlm.nih.gov": 0.95,
        "pubmed.ncbi.nlm.nih.gov": 0.95,
        "nejm.org": 0.9,
        "radiology.rsna.org": 0.85,
        "nice.org.uk": 0.85,
        "kdigo.org": 0.8
    }
    
    for domain, domain_score in premium_domains.items():
        if domain in url.lower():
            score += domain_score
            break
    else:
        score += 0.3
    
    quality_keywords = [
        "protocol", "guidelines", "appropriateness", "consensus",
        "recommendation", "standard", "imaging", "radiology"
    ]
    
    text = f"{title} {snippet}".lower()
    for keyword in quality_keywords:
        if keyword in text:
            score += 0.1
    
    if len(snippet) < 100:
        score -= 0.2
    
    return min(score, 1.0)

def update_protocol_database_smart(new_protocols: List[Dict]):
    if not new_protocols:
        return 0
    
    try:
        with open(PROTOCOL_DB_PATH, 'r', encoding='utf-8') as f:
            existing_protocols = json.load(f)
    except FileNotFoundError:
        existing_protocols = []
    except UnicodeDecodeError:
        try:
            with open(PROTOCOL_DB_PATH, 'r', encoding='utf-8-sig') as f:
                existing_protocols = json.load(f)
        except:
            existing_protocols = []
    
    valid_protocols = []
    for protocol in new_protocols:
        references = protocol.get("references", [])
        if any("example" in ref or "fake" in ref for ref in references):
            print(f"Skipping fake protocol: {protocol.get('protocol_id')}")
            continue
            
        description = protocol.get("description", "")
        if "Standard imaging protocol recommendations for" in description and len(description) < 100:
            print(f"Skipping generic protocol: {protocol.get('protocol_id')}")
            continue
            
        if protocol.get("quality_score", 0) < 0.6:
            print(f"Skipping low quality protocol: {protocol.get('protocol_id')}")
            continue
            
        if any(word in description.lower() for word in ["aim of this study", "explore the scope", "limitations of", "purpose of this study", "we investigated", "we examined"]):
            print(f"Skipping research paper (not protocol): {protocol.get('protocol_id')}")
            continue
            
        if not all([
            protocol.get("protocol_id"),
            protocol.get("name"),
            protocol.get("description"),
            len(protocol.get("description", "")) > 50
        ]):
            print(f"Skipping incomplete protocol: {protocol.get('protocol_id')}")
            continue
            
        valid_protocols.append(protocol)
    
    if not valid_protocols:
        print("No valid protocols to add (all were fake/invalid)")
        return 0
    
    existing_ids = {p.get("protocol_id", "").lower() for p in existing_protocols}
    existing_names = {p.get("name", "").lower() for p in existing_protocols}
    
    added_protocols = []
    for protocol in valid_protocols:
        protocol_id = protocol.get("protocol_id", "").lower()
        protocol_name = protocol.get("name", "").lower()
        
        if protocol_id not in existing_ids and not any(
            name in protocol_name or protocol_name in name 
            for name in existing_names
        ):
            existing_protocols.append(protocol)
            added_protocols.append(protocol)
            existing_ids.add(protocol_id)
            existing_names.add(protocol_name)
    
    if added_protocols:
        os.makedirs(os.path.dirname(PROTOCOL_DB_PATH), exist_ok=True)
        with open(PROTOCOL_DB_PATH, 'w', encoding='utf-8') as f:
            json.dump(existing_protocols, f, indent=2, ensure_ascii=False)
        
        try:
            build_vector_index(existing_protocols, VECTOR_INDEX_PATH)
        except Exception as e:
            print(f"Warning: Could not rebuild vector index: {e}")
    
    return len(added_protocols)

def run_enhanced_agent2_protocol(patient_data: Dict) -> str:
    print("\nAGENT 2: ENHANCED PROTOCOL SELECTION")
    print("=" * 50)
    
    print("Checking existing protocol database...")
    
    try:
        protocols = load_or_create_index(PROTOCOL_DB_PATH, VECTOR_INDEX_PATH)
        
        if not protocols:
            print("Protocol database is empty")
            raise Exception("Empty protocol database")
            
        query = patient_data.get("primary_diagnosis", "")
        if not query:
            print("No diagnosis provided for search")
            raise Exception("No diagnosis for search")
            
        existing_protocols = vector_search(query, protocols, VECTOR_INDEX_PATH, top_k=3)
        
        if not existing_protocols:
            print("Vector search returned no results")
            raise Exception("No search results")
        
        relevance_score = check_protocol_relevance(patient_data, existing_protocols)
        print(f"Relevance score: {relevance_score:.3f}")
        
        if relevance_score >= RELEVANCE_THRESHOLD:
            print(f"Using existing protocols (relevance: {relevance_score:.3f})")
            selected_protocols = existing_protocols
            search_method = "existing_database"
        else:
            print(f"Low relevance ({relevance_score:.3f} < {RELEVANCE_THRESHOLD})")
            print("Initiating internet search...")
            
            search_results = enhanced_search_medical_sources(patient_data)
            
            if search_results:
                print(f"Found {len(search_results)} new sources")
                
                new_protocols = convert_search_results_to_protocols(search_results, patient_data)
                
                added_count = update_protocol_database_smart(new_protocols)
                print(f"Added {added_count} new protocols to database")
                
                if added_count > 0:
                    updated_protocols = load_or_create_index(PROTOCOL_DB_PATH, VECTOR_INDEX_PATH)
                    selected_protocols = vector_search(query, updated_protocols, VECTOR_INDEX_PATH, top_k=3)
                    search_method = "internet_search_updated"
                else:
                    selected_protocols = new_protocols[:2]
                    search_method = "internet_search_new"
            else:
                print("No new protocols found, using existing ones anyway")
                selected_protocols = existing_protocols
                search_method = "fallback_existing"
                
    except Exception as e:
        print(f"Database error: {e}")
        print("Falling back to internet search...")
        
        search_results = enhanced_search_medical_sources(patient_data)
        new_protocols = convert_search_results_to_protocols(search_results, patient_data)
        selected_protocols = new_protocols[:2]
        search_method = "fallback_internet"
    
    print(f"Generating decision with {len(selected_protocols)} protocols...")
    
    decision_data = generate_protocol_decision(patient_data, selected_protocols)
    
    # Format as enhanced string output
    return format_enhanced_output(patient_data, decision_data, selected_protocols, search_method)

def format_enhanced_output(patient_data: Dict, decision_data: Dict, protocols: List[Dict], search_method: str) -> str:
    """Format comprehensive output string using ALL available patient parameters"""
    
    # Core identifiers
    subject_id = patient_data.get('subject_id', 'Unknown')
    hadm_id = patient_data.get('hadm_id', 'Unknown')
    stay_id = patient_data.get('stay_id', 'Unknown')
    
    # Demographics
    age = patient_data.get('age', 'Unknown')
    gender = patient_data.get('gender', 'Unknown')
    race = patient_data.get('race', 'Unknown')
    admission = patient_data.get('admission_type', 'Unknown')
    insurance = patient_data.get('insurance', 'Unknown')
    
    # Clinical outcome markers
    hospital_expire = patient_data.get('hospital_expire_flag', 'Unknown')
    los_hospital = patient_data.get('los_hospital_days', 'Unknown')
    los_icu = patient_data.get('los_icu_days', 'Unknown')
    first_unit = patient_data.get('first_careunit', 'Unknown')
    last_unit = patient_data.get('last_careunit', 'Unknown')
    
    # Primary diagnosis
    diagnosis = patient_data.get('primary_diagnosis', 'Unknown')
    
    # Renal function
    egfr = patient_data.get('egfr_ckd_epi', 'Unknown')
    creatinine = patient_data.get('creatinine_mg_dl', 'Unknown')
    ckd_stage = patient_data.get('ckd_stage', 'Unknown')
    bun = patient_data.get('bun_mg_dl', 'Unknown')
    bun_creat_ratio = patient_data.get('bun_creatinine_ratio', 'Unknown')
    
    # Vital signs
    heart_rate = patient_data.get('heart_rate_bpm', 'Unknown')
    systolic_bp = patient_data.get('systolic_bp_mmhg', 'Unknown')
    diastolic_bp = patient_data.get('diastolic_bp_mmhg', 'Unknown')
    map_value = patient_data.get('map_mmhg', 'Unknown')
    pulse_pressure = patient_data.get('pulse_pressure_mmhg', 'Unknown')
    temperature = patient_data.get('temperature_f', 'Unknown')
    spo2 = patient_data.get('spo2_pct', 'Unknown')
    resp_rate = patient_data.get('respiratory_rate', 'Unknown')
    
    # Laboratory values
    glucose = patient_data.get('glucose_mg_dl', 'Unknown')
    hemoglobin = patient_data.get('hemoglobin_g_dl', 'Unknown')
    hematocrit = patient_data.get('hematocrit_pct', 'Unknown')
    platelets = patient_data.get('platelet_count', 'Unknown')
    wbc = patient_data.get('wbc_count', 'Unknown')
    sodium = patient_data.get('sodium_meq_l', 'Unknown')
    potassium = patient_data.get('potassium_meq_l', 'Unknown')
    chloride = patient_data.get('chloride_meq_l', 'Unknown')
    bicarbonate = patient_data.get('bicarbonate_meq_l', 'Unknown')
    
    # Data quality
    completeness = patient_data.get('data_completeness_pct', 'Unknown')
    
    # Get primary protocol details
    primary_protocol = protocols[0] if protocols else {}
    protocol_name = primary_protocol.get('name', 'No protocol selected')
    contrast_timing = primary_protocol.get('contrast_timing', 'Not specified')
    contrast_dose = primary_protocol.get('contrast_dose', 'Not specified')
    renal_notes = primary_protocol.get('renal_safety_notes', 'Standard precautions')
    
    # Get recommendations and rationale
    recommendations = decision_data.get('recommendations', ['No specific recommendations available'])
    rationale = decision_data.get('rationale', ['Clinical reasoning not available'])
    
    output = f"""
COMPREHENSIVE IMAGING PROTOCOL REPORT
=====================================

PATIENT IDENTIFIERS:
• Subject ID: {subject_id} | Admission ID: {hadm_id} | Stay ID: {stay_id}
• Demographics: {age}yr {gender} {race}
• Insurance: {insurance} | Admission: {admission}
• Care Units: {first_unit} → {last_unit}
• Length of Stay: Hospital {los_hospital}d, ICU {los_icu}d

PRIMARY PRESENTATION:
• Diagnosis: {diagnosis}
• Hospital Mortality: {'Yes' if hospital_expire == 1 else 'No' if hospital_expire == 0 else 'Unknown'}

RENAL FUNCTION ASSESSMENT:
• eGFR: {egfr} mL/min/1.73m² ({ckd_stage})
• Creatinine: {creatinine} mg/dL | BUN: {bun} mg/dL
• BUN/Creatinine Ratio: {bun_creat_ratio}

HEMODYNAMIC STATUS:
• Blood Pressure: {systolic_bp}/{diastolic_bp} mmHg (MAP {map_value})
• Heart Rate: {heart_rate} bpm | Pulse Pressure: {pulse_pressure} mmHg
• Temperature: {temperature}°F | SpO2: {spo2}% | RR: {resp_rate}/min

LABORATORY PROFILE:
• Hematology: Hgb {hemoglobin} g/dL, Hct {hematocrit}%, Platelets {platelets}K, WBC {wbc}K
• Chemistry: Glucose {glucose} mg/dL, Na {sodium}, K {potassium}, Cl {chloride}, HCO3 {bicarbonate}
• Data Completeness: {completeness}%

PROTOCOL RECOMMENDATION:
• Selected Protocol: {protocol_name}
• Search Method: {search_method.replace('_', ' ').title()}
• Contrast Timing: {contrast_timing}
• Contrast Dosing: {contrast_dose}
• Renal Safety: {renal_notes}

CLINICAL RECOMMENDATIONS:
"""
    
    for i, rec in enumerate(recommendations, 1):
        output += f"{i}. {rec}\n"
    
    output += "\nCLINICAL RATIONALE:\n"
    for i, reason in enumerate(rationale, 1):
        output += f"{i}. {reason}\n"
    
    # Enhanced risk assessment using all available parameters
    output += "\nCOMPREHENSIVE RISK ASSESSMENT:\n"
    
    # Renal risk assessment
    if isinstance(egfr, (int, float)):
        if egfr < 15:
            output += "• CRITICAL RENAL RISK: eGFR <15 - Dialysis-dependent, avoid iodinated contrast\n"
        elif egfr < 30:
            output += "• HIGH RENAL RISK: eGFR <30 - Consider non-contrast alternatives\n"
        elif egfr < 60:
            output += "• MODERATE RENAL RISK: eGFR <60 - Hydration and dose adjustment recommended\n"
        else:
            output += "• LOW RENAL RISK: Normal kidney function for contrast studies\n"
    
    # Hemodynamic risk assessment
    if isinstance(map_value, (int, float)):
        if map_value < 50:
            output += "• SEVERE HYPOTENSION: MAP <50 - High risk for contrast-induced hypotension\n"
        elif map_value < 65:
            output += "• HYPOTENSION: MAP <65 - Monitor perfusion during procedure\n"
        else:
            output += "• STABLE HEMODYNAMICS: Adequate perfusion pressure\n"
    
    # Cardiac risk assessment
    if isinstance(heart_rate, (int, float)):
        if heart_rate > 120:
            output += "• TACHYCARDIA: HR >120 - Consider cardiac stress, volume status\n"
        elif heart_rate < 50:
            output += "• BRADYCARDIA: HR <50 - Monitor for conduction abnormalities\n"
    
    # Anemia assessment
    if isinstance(hemoglobin, (int, float)):
        if hemoglobin < 7:
            output += "• SEVERE ANEMIA: Hgb <7 - Consider transfusion needs, bleeding risk\n"
        elif hemoglobin < 10:
            output += "• MODERATE ANEMIA: Hgb <10 - Monitor for hemodynamic compromise\n"
    
    # Infection/inflammation markers
    if isinstance(wbc, (int, float)):
        if wbc > 15:
            output += "• LEUKOCYTOSIS: WBC >15K - Active infection/inflammation likely\n"
        elif wbc < 4:
            output += "• LEUKOPENIA: WBC <4K - Immunocompromised state\n"
    
    # Respiratory status
    if isinstance(spo2, (int, float)) and spo2 < 95:
        output += "• HYPOXEMIA: SpO2 <95% - Respiratory compromise present\n"
    
    # Emergency context
    if 'EMERG' in admission.upper() or 'EW' in admission.upper():
        output += "• EMERGENCY CONTEXT: Time-sensitive imaging priorities apply\n"
    
    # ICU context
    if isinstance(los_icu, (int, float)) and los_icu > 0:
        output += "• ICU PATIENT: Critical care context - Monitor for complications\n"
    
    # Mortality risk
    if hospital_expire == 1:
        output += "• HIGH ACUITY: Patient expired during hospitalization\n"
    
    output += f"\nREPORT GENERATED: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
    output += "=" * 80
    
    return output

def generate_protocol_decision(patient_data: Dict, protocols: List[Dict]) -> Dict:
    prompt = f"""
    You are a radiology protocol assistant. Generate imaging protocol recommendations.

    PATIENT:
    Age: {patient_data.get('age')}
    Diagnosis: {patient_data.get('primary_diagnosis')}
    Admission: {patient_data.get('admission_type')}
    eGFR: {patient_data.get('egfr_ckd_epi')}
    Creatinine: {patient_data.get('creatinine_mg_dl')}
    MAP: {patient_data.get('map_mmhg')}

    AVAILABLE PROTOCOLS:
    {json.dumps(protocols, indent=2)}

    Return JSON with:
    {{
        "recommendations": ["2-4 specific actionable recommendations"],
        "rationale": ["brief explanations for each recommendation"],
        "protocol_selection": [
            {{
                "protocol_id": "ID",
                "name": "Protocol Name",
                "justification": "Why selected",
                "contrast_considerations": "Contrast notes"
            }}
        ]
    }}
    """

    try:
        response = client.models.generate_content(
            model=MODEL,
            contents=prompt,
            config={"response_mime_type": "application/json"}
        )
        
        return json.loads(response.text)
        
    except Exception as e:
        print(f"Decision generation failed: {e}")
        return {
            "recommendations": [
                "Manual protocol review required due to AI processing error",
                "Verify patient renal function before contrast administration",
                "Consider non-contrast alternatives if eGFR < 30"
            ],
            "rationale": [f"AI processing failed: {e}"],
            "protocol_selection": protocols[:1] if protocols else [{"error": "No protocols available"}]
        }

def run_agent2_2_enhanced(patient_data: Dict, enhanced_context: str = "", feedback: Optional[Dict] = None):
    result = run_enhanced_agent2_protocol(patient_data)
    
    if feedback:
        result["feedback_incorporated"] = feedback
    
    if enhanced_context:
        result["enhanced_context"] = enhanced_context
    
    return result

if __name__ == "__main__":
    sample_patient= {
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
    "systolic_bp_mmhg": 78,  # Hypotensive
    "diastolic_bp_mmhg": 35,
    "temperature_f": 99.1,
    "spo2_pct": 98,
    "respiratory_rate": 20,
    "map_mmhg": 49.3,  # Low MAP
    "pulse_pressure_mmhg": 43,
    "data_completeness_pct": 100
}
    
    result = run_enhanced_agent2_protocol(sample_patient)
    print("\nENHANCED AGENT 2 RESULT:")
    print(json.dumps(result, indent=2))