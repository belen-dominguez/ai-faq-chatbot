# utils/search.py
import numpy as np
from langchain_google_vertexai import VertexAIEmbeddings


def cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    """
    Calcula la similitud coseno entre dos vectores.
    Devuelve un valor entre 0 y 1 — más cercano a 1 = más similares.
    """
    a = np.array(vec1)
    b = np.array(vec2)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def embed_query(query: str, model: str = "text-embedding-004") -> list[float]:
    """
    Convierte la pregunta del usuario en un vector usando Vertex AI.

    Argumentos:
        query: pregunta del usuario
        model: modelo de embeddings a usar

    Retorna:
        Vector de la pregunta
    """
    embeddings_model = VertexAIEmbeddings(model=model)
    return embeddings_model.embed_query(query)


def search_similar_chunks(
    query: str,
    chunks: list[str],
    chunk_embeddings: list[list[float]],
    top_k: int = 3
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
    query_embedding = embed_query(query)

    scores = [
        {"chunk": chunk, "score": cosine_similarity(query_embedding, emb)}
        for chunk, emb in zip(chunks, chunk_embeddings)
    ]

    scores.sort(key=lambda x: x["score"], reverse=True)

    top_chunks = scores[:top_k]
    print(f"✅ Top {top_k} chunks recuperados (score más alto: {top_chunks[0]['score']:.4f})")
    return top_chunks