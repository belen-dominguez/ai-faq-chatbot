from langchain_google_vertexai import ChatVertexAI, VertexAIEmbeddings
from pipeline import IndexPipeline, RAGPipeline
import json



KNOWLEDGE_BASE_PATH = "data/knowledge_base.txt"

embeddings_model = VertexAIEmbeddings(model="text-embedding-004")
llm = ChatVertexAI(model="gemini-2.5-flash-lite")


def main():
    index_pipeline = IndexPipeline(KNOWLEDGE_BASE_PATH)
    chunks, embeddings = index_pipeline.run()

    rag = RAGPipeline(embeddings_model, llm)

    # 2. Preguntas de prueba
    test_questions = [
        "¿Cómo recupero mi contraseña?",
        "¿Qué roles existen en PeopleCore y qué puede hacer cada uno?",
        "¿Cómo se procesa la nómina y qué pasa si hay un error?"
    ]


    for question in test_questions:
        print(f"\n🔍 Pregunta: {question}")

        result = rag.run(question, chunks, embeddings)

        print(json.dumps(result, ensure_ascii=False, indent=2))
        print("\n" + "─"*60)



if __name__ == "__main__":
    main()