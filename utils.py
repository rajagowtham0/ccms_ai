import re
import json
from typing import List, Dict
# text pre-processing
def clean_text(text: str) -> str:
    if not isinstance(text, str):
        raise TypeError("Input must be a string.")

    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text

# combining symptoms and clinical notes
def combine_input(symptoms: str, clinical_note: str) -> str:
    if not isinstance(symptoms, str) or not isinstance(clinical_note, str):
        raise TypeError("Symptoms and clinical_note must be strings.")
    combined = f"{symptoms} {clinical_note}"
    return clean_text(combined)
def combine_case_text(case: Dict) -> str:
    return clean_text(f"{case['symptoms']} {case['doctor_notes']}")

# validating the case schema
def validate_case_schema(case: Dict,
                         required_fields: List[str] = None) -> None:
    if not isinstance(case, dict):
        raise TypeError("Each case must be a dictionary.")

    # Default required fields
    if required_fields is None:
        required_fields = [
            "case_id",
            "symptoms",
            "doctor_notes",
            "treatment",
            "recovery_days"
        ]

    missing = set(required_fields) - set(case.keys())

    if missing:
        raise ValueError(f"Missing required fields: {missing}")

# vaidating case list
def validate_case_list(cases: List[Dict],
                       required_fields: List[str] = None) -> None:

    if not isinstance(cases, list):
        raise TypeError("Stored cases must be a list.")

    if len(cases) == 0:
        raise ValueError("Stored cases list is empty.")

    for case in cases:
        validate_case_schema(case, required_fields)
# embedding attachment
def attach_embeddings(cases: List[Dict], embeddings) -> List[Dict]:
    if len(cases) != len(embeddings):
        raise ValueError("Mismatch between cases and embeddings.")

    for i, case in enumerate(cases):
        case["embedding"] = embeddings[i]

    return cases
# recovery trend formatter
def format_recovery_trend(recovery_days) -> str:

    if recovery_days is None:
        return "Recovery period not specified"

    try:
        days = int(recovery_days)

        if days <= 3:
            return "Rapid recovery (≤ 3 days)"
        elif days <= 7:
            return "Recovered within 1 week"
        elif days <= 14:
            return "Recovered within 2 weeks"
        else:
            return "Prolonged recovery (> 2 weeks)"

    except Exception:
        return "Recovery period not specified"
# confidence score
def calculate_confidence(similar_cases: List[Dict]) -> float:
    if not similar_cases:
        return 0.0

    scores = [case["similarity_score"] for case in similar_cases]

    return round(sum(scores) / len(scores), 4)
# output formatting
def format_output(symptoms: str,
                  clinical_note: str,
                  similar_cases: List[Dict],
    formatted_cases = []

    for case in similar_cases:
        formatted_cases.append({
            "case_id": case["case_id"],
            "similarity_score": case["similarity_score"],
            "treatment": case.get("treatment", "Not available"),
            "recovery_trend": format_recovery_trend(
                case.get("recovery_days")
            )
        })

    return {
        "query": {
            "symptoms": symptoms,
            "clinical_note": clinical_note
        },
        "top_similar_patients": formatted_cases,
        "confidence_score": confidence_score
    }