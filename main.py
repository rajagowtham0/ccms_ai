import json
import pandas as pd
from embedding import generate_embeddings, generate_single_embedding
from similarity_engine import retrieve_similar_cases
from utils import (
    combine_input,
    combine_case_text,
    validate_case_list,
    attach_embeddings,
    calculate_confidence,
    format_output
)

def load_dataset(csv_path: str) -> list:
    df = pd.read_csv(r"C:\Users\rajak\Downloads\Week_0_Prep_Week_Ssample Data_clinic_cases.csv")
    return df.to_dict(orient="records")
# similarity pipeline
def run_similarity_pipeline(symptoms: str,
                            clinical_note: str,
                            stored_cases: list,
                            top_n: int = 4):

    # Validate dataset schema
    validate_case_list(stored_cases)

    # Prepare text for embedding 
    case_texts = [
        combine_case_text(case)
        for case in stored_cases
    ]

    # Generate embeddings for stored cases
    stored_embeddings = generate_embeddings(case_texts)

    stored_cases = attach_embeddings(
        stored_cases,
        stored_embeddings
    )

    # Prepare query
    combined_query = combine_input(symptoms, clinical_note)
    query_embedding = generate_single_embedding(combined_query)

    # Retrieve similar cases
    similar_cases = retrieve_similar_cases(
        new_embedding=query_embedding,
        stored_cases=stored_cases,
        top_n=top_n
    )

    # Calculate confidence score
    confidence_score = calculate_confidence(similar_cases)

    # Format structured analytical output
    output = format_output(
        symptoms=symptoms,
        clinical_note=clinical_note,
        similar_cases=similar_cases,
        stored_cases=stored_cases,
        confidence_score=confidence_score
    )

    return output
# execution block
if __name__ == "__main__":

    print("\n=== Clinical Case Similarity Engine ===\n")

    # Take user input
    symptoms_input = input("Enter patient symptoms: ")
    clinical_note_input = input("Enter clinical notes: ")

    # Load CSV dataset
    csv_file_path = r"C:\Users\rajak\Downloads\Week_0_Prep_Week_Ssample Data_clinic_cases.csv" 
    stored_cases = load_dataset(csv_file_path)

    # Run similarity pipeline
    result = run_similarity_pipeline(
        symptoms=symptoms_input,
        clinical_note=clinical_note_input,
        stored_cases=stored_cases,
        top_n=4
    )

    # Print clean structured JSON output
    print("\n=== Similarity Analysis Result ===\n")
    print(json.dumps(result, indent=4))