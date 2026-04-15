from langchain_google_vertexai import ChatVertexAI
from langchain_core.messages import HumanMessage


def evaluate_response(
    user_question: str,
    system_answer: str,
    chunks_related: list[str]
) -> dict:
    """
    Agente evaluador que analiza la calidad de la respuesta RAG.
    Considera relevancia de chunks, calidad y completitud de la respuesta.

    Argumentos:
        user_question: pregunta original del usuario
        system_answer: respuesta generada por el sistema
        chunks_related: chunks usados para generar la respuesta

    Retorna:
        Dict con 'score' (0-10) y 'reason' (justificación detallada)
    """
    llm = ChatVertexAI(model="gemini-2.0-flash")

    chunks_text = "\n\n".join([f"Chunk {i+1}: {chunk}" for i, chunk in enumerate(chunks_related)])

    prompt = f"""Eres un evaluador experto de sistemas RAG (Retrieval Augmented Generation).
Analizá la siguiente interacción y evaluá su calidad considerando estas tres dimensiones:

1. RELEVANCIA DE CHUNKS: ¿Los chunks recuperados se relacionan con la pregunta?
2. CALIDAD DE LA RESPUESTA: ¿La respuesta usa información de los chunks y es precisa?
3. COMPLETITUD: ¿La respuesta cubre totalmente la pregunta del usuario?

PREGUNTA DEL USUARIO:
{user_question}

CHUNKS RECUPERADOS:
{chunks_text}

RESPUESTA DEL SISTEMA:
{system_answer}

Respondé ÚNICAMENTE con este formato JSON, sin texto adicional, sin bloques de código:
{{
    "score": <entero del 0 al 10>,
    "reason": "<justificación detallada de al menos 50 caracteres mencionando las tres dimensiones evaluadas>"
}}"""

    response = llm.invoke([HumanMessage(content=prompt)])

    import json
    result = json.loads(response.content)
    print(f"✅ Evaluación completada — Score: {result['score']}/10")
    return result