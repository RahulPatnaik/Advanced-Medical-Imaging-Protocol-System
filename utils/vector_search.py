import os
import json
import numpy as np
import hnswlib
from sentence_transformers import SentenceTransformer

# ==== CONFIG ====
EMBEDDING_MODEL_NAME = 'sentence-transformers/all-mpnet-base-v2'  # upgraded for accuracy
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
VECTOR_DIMENSION = embedding_model.get_sentence_embedding_dimension()

def build_vector_index(protocols, index_path):
    """Builds a vector index from protocol data."""
    texts_to_embed = [
        f"{p['name']}. Indications: {', '.join(p['indications'])}" 
        for p in protocols
    ]
    embeddings = embedding_model.encode(texts_to_embed, normalize_embeddings=True)

    index = hnswlib.Index(space='cosine', dim=VECTOR_DIMENSION)
    index.init_index(max_elements=len(protocols), ef_construction=300, M=16)
    index.add_items(embeddings, np.arange(len(protocols)))
    index.save_index(index_path)

def load_or_create_index(db_path, index_path):
    """Ensures protocol DB and index exist, returns DB."""
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Protocol DB not found: {db_path}")

    with open(db_path, 'r') as f:
        protocols = json.load(f)

    if not os.path.exists(index_path):
        build_vector_index(protocols, index_path)

    return protocols

def vector_search(query, protocols, index_path, top_k=2):
    """Search for the most relevant protocols."""
    index = hnswlib.Index(space='cosine', dim=VECTOR_DIMENSION)
    index.load_index(index_path)
    index.set_ef(50)  # better recall
    query_embedding = embedding_model.encode([query], normalize_embeddings=True)
    labels, _ = index.knn_query(query_embedding, k=top_k)
    return [protocols[i] for i in labels[0]]
