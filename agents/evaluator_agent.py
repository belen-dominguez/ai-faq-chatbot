from langchain_google_vertexai import ChatVertexAI
from langchain_core.messages import HumanMessage
import json


llm = ChatVertexAI(model="gemini-2.5-flash-lite")

def clean_json_response(text):
    text = text.strip()

    if text.startswith("```"):
        text = text.replace("```json", "").replace("```", "").strip()

    return text



def evaluate_response(
    user_question: str,
    system_answer: str,
    chunks_related: list[dict]
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
    

    chunks_text = "\n\n".join([
    f"Chunk {i+1} (score: {c['score']:.2f}): {c['chunk']}"
    for i, c in enumerate(chunks_related)
    ])

    prompt = f"""Sos un evaluador experto de sistemas RAG.

Evaluá la siguiente respuesta considerando:

1. RELEVANCIA (0-10): ¿Los chunks son útiles para la pregunta?
2. EXACTITUD (0-10): ¿La respuesta es fiel a los chunks? (penalizá invenciones)
3. COMPLETITUD (0-10): ¿responde completamente la pregunta?

Reglas importantes:
- Si la respuesta contiene información que NO está en los chunks → penalizar EXACTITUD
- Si los chunks no son relevantes → penalizar RELEVANCIA
- Si falta información clave → penalizar COMPLETITUD

PREGUNTA:
{user_question}

CHUNKS:
{chunks_text}

RESPUESTA:
{system_answer}

Respondé SOLO con JSON válido (sin markdown):

{{
  "relevance": int,
  "accuracy": int,
  "completeness": int,
  "final_score": int,
  "reason": "explicación clara"
}}
"""

    response = llm.invoke([HumanMessage(content=prompt)])

    cleaned = clean_json_response(response.content)
    result = json.loads(cleaned)
    print(
    f"📊 Evaluación → "
    f"Rel: {result['relevance']} | "
    f"Acc: {result['accuracy']} | "
    f"Comp: {result['completeness']} | "
    f"Final: {result['final_score']}/10"
    )
    return result