from langchain_google_vertexai import ChatVertexAI, VertexAIEmbeddings
from pipeline import IndexPipeline, RAGPipeline
from shared.config_loader import load_config
import json

config = load_config()

TOP_K = config["retrieval"]["top_k"]
KNOWLEDGE_BASE_PATH = config["path"]["knowledge_base"]
EMBEDDING_MODEL = config["models"]["embedding_model"]
LLM_MODEL = config["models"]["llm_model"]

embeddings_model = VertexAIEmbeddings(model=EMBEDDING_MODEL)
llm = ChatVertexAI(model=LLM_MODEL)


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