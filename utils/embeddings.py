import chromadb
import numpy as np
from langchain_google_vertexai import VertexAIEmbeddings

embeddings_model = VertexAIEmbeddings(model='text-embedding-004')

def generate_embeddings(chunks: list[str]) -> list[list[float]]:
    """
    Genera embeddings para cada chunk usando Vertex AI.

    Argumentos:
        chunks: lista de strings a embeddear

    Retorna:
        Lista de vectores (cada vector es una lista de floats)
    """
    embeddings = embeddings_model.embed_documents(chunks)
    print(f"✅ Embeddings generados: {len(embeddings)} vectores de dimensión {len(embeddings[0])}")
    return embeddings


def store_embeddings(chunks: list[str], embeddings: list[list[float]], collection_name: str = "faq_chunks") -> chromadb.Collection:
    """
    Guarda los chunks y sus embeddings en ChromaDB (en memoria).

    Argumentos:
        chunks: textos originales de los chunks
        embeddings: vectores correspondientes a cada chunk
        collection_name: nombre de la colección en ChromaDB

    Retorna:
        La colección de ChromaDB con los datos guardados
    """
    client = chromadb.Client()
    collection = client.get_or_create_collection(name=collection_name)

    collection.add(
        documents=chunks,
        embeddings=embeddings,
        ids=[f"chunk_{i}" for i in range(len(chunks))]
    )

    print(f"✅ {len(chunks)} chunks almacenados en ChromaDB (colección: '{collection_name}')")
    return collection


def get_all_embeddings(collection: chromadb.Collection) -> tuple[list[str], list[list[float]]]:
    """
    Recupera todos los chunks y embeddings almacenados en ChromaDB.

    Argumentos:
        collection: colección de ChromaDB

    Retorna:
        Tupla de (chunks, embeddings)
    """
    result = collection.get(include=["documents", "embeddings"])
    
    chunks = result["documents"]
    embeddings = np.array(result["embeddings"])

    return chunks, embeddings