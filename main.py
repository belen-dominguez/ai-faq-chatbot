import json
from langchain_google_vertexai import ChatVertexAI
from langchain_core.messages import HumanMessage

from utils.chunker import document_load_and_generate_chunk
from utils.embeddings import generate_embeddings, store_embeddings, get_all_embeddings
from utils.search import search_similar_chunks
from agents.evaluator_agent import evaluate_response


# ── Configuración ──────────────────────────────────────────────────────────────

KNOWLEDGE_BASE_PATH = "knowledge_base.txt"
TOP_K = 3


# ── Pipeline de Indexación ─────────────────────────────────────────────────────

def build_index(filepath: str) -> tuple:
    """
    Ejecuta el pipeline completo de indexación:
    1. Carga y divide el documento en chunks
    2. Genera embeddings para cada chunk
    3. Almacena en ChromaDB

    Retorna:
        Tupla (chunks, embeddings, collection)
    """
    print("\n📄 Iniciando pipeline de indexación...")
    chunks = document_load_and_generate_chunk(filepath)
    embeddings = generate_embeddings(chunks)
    collection = store_embeddings(chunks, embeddings)
    chunks_stored, embeddings_stored = get_all_embeddings(collection)
    print(f"✅ Índice construido con {len(chunks_stored)} chunks\n")
    return chunks_stored, embeddings_stored


# ── Pipeline de Consulta ───────────────────────────────────────────────────────

def generate_answer(user_question: str, chunks: list[str], embeddings: list[list[float]]) -> dict:
    """
    Ejecuta el pipeline completo de consulta:
    1. Busca los chunks más relevantes
    2. Arma el contexto para el LLM
    3. Genera la respuesta con Gemini
    4. Devuelve JSON estructurado

    Retorna:
        Dict con user_question, system_answer y chunks_related
    """
    # Etapa 1 y 2: búsqueda vectorial
    top_chunks = search_similar_chunks(user_question, chunks, embeddings, top_k=TOP_K)
    chunks_related = [c["chunk"] for c in top_chunks]

    # Etapa 3: ensamblado de contexto
    context = "\n\n".join([f"Fragmento {i+1}:\n{chunk}" for i, chunk in enumerate(chunks_related)])

    prompt = f"""Eres un asistente de soporte para PeopleCore, un sistema HR SaaS.
Respondé la pregunta del usuario basándote ÚNICAMENTE en el contexto provisto.
Si la información no está en el contexto, indicá que no tenés esa información disponible.

CONTEXTO:
{context}

PREGUNTA:
{user_question}

Respondé de forma clara, precisa y en español."""

    # Etapa 4: generación con LLM
    llm = ChatVertexAI(model="gemini-2.0-flash")
    response = llm.invoke([HumanMessage(content=prompt)])

    return {
        "user_question": user_question,
        "system_answer": response.content,
        "chunks_related": chunks_related
    }


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    # 1. Construir el índice
    chunks, embeddings = build_index(KNOWLEDGE_BASE_PATH)

    # 2. Preguntas de prueba
    test_questions = [
        "¿Cómo recupero mi contraseña?",
        "¿Qué roles existen en PeopleCore y qué puede hacer cada uno?",
        "¿Cómo se procesa la nómina y qué pasa si hay un error?"
    ]

    results = []

    for question in test_questions:
        print(f"\n🔍 Pregunta: {question}")

        # 3. Generar respuesta
        result = generate_answer(question, chunks, embeddings)

        # 4. Evaluar respuesta (bonus)
        evaluation = evaluate_response(
            result["user_question"],
            result["system_answer"],
            result["chunks_related"]
        )
        result["evaluation"] = evaluation

        results.append(result)

        # 5. Mostrar resultado
        print(json.dumps(result, ensure_ascii=False, indent=2))
        print("\n" + "─"*60)

    print(f"\n✅ Pipeline completado — {len(results)} preguntas procesadas")


if __name__ == "__main__":
    main()