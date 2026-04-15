from langchain_google_vertexai import VertexAIEmbeddings


def generate_embeddings(chunks: list[str], model: str = "text-embedding-004") -> list[list[float]]:
    """
    Genera embeddings para cada chunk usando Vertex AI.

    Argumentos:
        chunks: lista de strings a embeddear
        model: modelo de embeddings a usar

    Retorna:
        Lista de vectores (cada vector es una lista de floats)
    """
    embeddings_model = VertexAIEmbeddings(model=model)
    embeddings = embeddings_model.embed_documents(chunks)
    print(f"✅ Embeddings generados: {len(embeddings)} vectores de dimensión {len(embeddings[0])}")
    return embeddings