# functional similarity engine module
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# similarity engine to retrieve top similar cases
def retrieve_similar_cases(new_embedding: np.ndarray,
                           stored_cases: list,
                           top_n: int = 5):
    if len(stored_cases) == 0:
        raise ValueError("No stored cases available.")
    # Extract embeddings from stored cases
    stored_embeddings = np.array(
        [case["embedding"] for case in stored_cases]
    )
    # Ensure correct shape
    if new_embedding.ndim == 1:
        new_embedding = new_embedding.reshape(1, -1)
    # Compute cosine similarity
    similarity_scores = cosine_similarity(
        new_embedding,
        stored_embeddings
    )[0]
    # Rank in descending order
    ranked_indices = np.argsort(similarity_scores)[::-1][:top_n]
    results = []
    for idx in ranked_indices:
        results.append({
            "case_id": stored_cases[idx]["case_id"],
            "similarity_score": float(similarity_scores[idx])
        })

    return results