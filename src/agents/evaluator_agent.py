from langchain_google_vertexai import ChatVertexAI
from langchain_core.messages import HumanMessage
import json

from prompts.templates import EVALUATOR_PROMPT


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

    prompt = EVALUATOR_PROMPT

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