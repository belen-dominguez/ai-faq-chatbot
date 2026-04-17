
import numpy as np
from shared.config_loader import load_config

config = load_config()

MIN_SCORE = config["retrieval"]["min_score"]

def cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    """
    Calcula la similitud coseno entre dos vectores.
    Devuelve un valor entre 0 y 1 — más cercano a 1 = más similares.
    """
    a = np.array(vec1)
    b = np.array(vec2)
   
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0

    return np.dot(a, b) / denom

#  str = "text-embedding-004"
def embed_query(query: str, embeddings_model) -> list[float]:
    """
    Convierte la pregunta del usuario en un vector usando Vertex AI.

    Argumentos:
        query: pregunta del usuario
        embeddings_model: modelo de embeddings a usar

    Retorna:
        Vector de la pregunta
    """
    return embeddings_model.embed_query(query)


def search_similar_chunks(
    query: str,
    chunks: list[str],
    chunk_embeddings: list[list[float]],
    embeddings_model, 
    top_k: int = 5
) -> list[dict]:
    """
    Busca los chunks más similares a la pregunta usando k-NN con similitud coseno.

    Argumentos:
        query: pregunta del usuario en texto
        chunks: textos originales de los chunks
        chunk_embeddings: vectores de todos los chunks
        top_k: cantidad de chunks a devolver (entre 2 y 5)

    Retorna:
        Lista de dicts con texto del chunk y su score de similitud
    """
    if len(chunks) == 0 or len(chunk_embeddings) == 0:
        print("⚠️ No hay datos para hacer búsqueda")
        return []

    query_embedding = embed_query(query, embeddings_model)

# 1. calculamos scores originales
    scores = [
    {
        "chunk": chunk,
        "score": cosine_similarity(query_embedding, emb),
        "chunk_id": i
    }
    for i, (chunk, emb) in enumerate(zip(chunks, chunk_embeddings))
    ]

# 2. ordenar por score
    scores.sort(key=lambda x: x["score"], reverse=True)


    filtered = [s for s in scores if s["score"] >= MIN_SCORE]
    top_chunks = filtered[:top_k]

    if not top_chunks:
        print("⚠️ No se encontraron chunks con threshold, devolviendo top_k sin filtrar")
        top_chunks = scores[:top_k]

    print(f"✅ {len(top_chunks)} chunks relevantes (score máx norm: {top_chunks[0]['score']:.4f})")
    return top_chunks