from prompts.templates import RETRIEVER_PROMPT
from utils.chunker import document_load_and_generate_chunk
from utils.embeddings import generate_embeddings, store_embeddings, get_all_embeddings
from utils.search import search_similar_chunks
from agents.evaluator_agent import evaluate_response
from langchain_core.messages import HumanMessage


class IndexPipeline:
    def __init__(self, filepath: str):
        self.filepath = filepath

    def run(self):
        """
        Ejecuta el pipeline completo de indexación:
        1. Carga y divide el documento en chunks
        2. Genera embeddings para cada chunk
        3. Almacena en ChromaDB

        Retorna:
            Tupla (chunks, embeddings, collection)
        """
        print("\n📄 Iniciando pipeline de indexación...")
        chunks = document_load_and_generate_chunk(self.filepath)
        embeddings = generate_embeddings(chunks)
        collection = store_embeddings(chunks, embeddings)
        chunks_stored, embeddings_stored = get_all_embeddings(collection)

        print(f"✅ Índice construido con {len(chunks_stored)} chunks\n")
        return chunks_stored, embeddings_stored


class RAGPipeline:
    def __init__(self, embeddings_model, llm, top_k=5):
        self.embeddings_model = embeddings_model
        self.llm = llm
        self.top_k = top_k

    def generate_answer(self, user_question, chunks, embeddings):
        """
        Ejecuta el pipeline completo de consulta:
        1. Busca los chunks más relevantes
        2. Arma el contexto para el LLM
        3. Genera la respuesta con Gemini
        4. Devuelve JSON estructurado

        Retorna:
            Dict con user_question, system_answer y chunks_related
        """
           
        # busqueda vectorial
        top_chunks = search_similar_chunks(
            user_question,
            chunks,
            embeddings,
            self.embeddings_model,
            top_k=self.top_k
        )

        if not top_chunks:
            return {
                "user_question": user_question,
                "system_answer": "No encontré información relevante en la documentación.",
                "chunks_related": []
            }

        # contexto
        context = "\n\n".join([
            f"Fragmento {i+1} (relevancia: {c['score']:.2f}):\n{c['chunk']}"
            for i, c in enumerate(top_chunks)
        ])

        # prompt
        prompt = RETRIEVER_PROMPT

        response = self.llm.invoke([HumanMessage(content=prompt)])

        return {
            "user_question": user_question,
            "system_answer": response.content,
            "chunks_related": top_chunks
        }

    def run(self, question, chunks, embeddings):
        result = self.generate_answer(question, chunks, embeddings)

        evaluation = evaluate_response(
            result["user_question"],
            result["system_answer"],
            result["chunks_related"]
        )

        result["evaluation"] = evaluation
        return result