🎯Objetivos

Implementar un sistema de FAQs basado en RAG que procese un documento de texto plano, lo divida en chunks de forma inteligente (mínimo 20 chunks), genere embeddings y los almacene para una recuperación eficiente. Esto crea una base de conocimiento consultable a partir de documentación no estructurada.
Construir un pipeline de consulta que acepte preguntas de usuarios, realice búsqueda vectorial usando métodos k-NN/ANN/rango/híbridos, recupere los chunks relevantes y genere respuestas precisas con un LLM. El sistema debe devolver JSON estructurado con user_question, system_answer y chunks_related para asegurar transparencia y auditabilidad.
Opcional: implementar un agente evaluador que puntúe la calidad de la respuesta (0–10) según la relevancia de los chunks, la precisión y la completitud. Esto aporta aseguramiento automático de calidad para el sistema RAG.

📢Consigna

Crea un chatbot de soporte para FAQs usando RAG que responda preguntas basándose en un documento del sistema. El sistema debe procesar un documento de texto plano, dividirlo en chunks (al menos 20 chunks) y generar embeddings. Para cada pregunta del usuario, devuelve una salida en JSON que contenga user_question, system_answer y chunks_related usados para generar la respuesta. Utiliza métodos de búsqueda vectorial (p. ej., k-NN, ANN, rango o híbridos) para encontrar de forma eficiente los chunks relevantes. Bonus: implementa un agente evaluador que reciba user_question, system_answer y chunks_related y devuelva un puntaje de 0 a 10 junto con una justificación del resultado.

# 🤖 RAG FAQ Chatbot – PeopleCore

## 📌 Descripción

Este proyecto implementa un sistema de preguntas frecuentes (FAQ) basado en **RAG (Retrieval-Augmented Generation)** para un producto HR SaaS llamado _PeopleCore_.

El objetivo es automatizar respuestas a consultas de usuarios utilizando documentación interna, evitando la intervención manual del equipo de soporte.

El sistema procesa un documento de texto, lo divide en fragmentos (chunks), genera embeddings y, ante una consulta, recupera los fragmentos más relevantes para generar una respuesta precisa con un modelo de lenguaje.

---

## 🧠 ¿Qué es RAG?

Este sistema utiliza la arquitectura **RAG (Retrieval-Augmented Generation)**:

1. **Recuperación (Retrieval)**
   Se buscan los fragmentos más relevantes del documento usando similitud coseno.

2. **Generación (Generation)**
   Un modelo LLM genera la respuesta utilizando esos fragmentos como contexto.

👉 Ventaja: permite usar conocimiento actualizado sin necesidad de reentrenar el modelo.

---

## ⚙️ Tecnologías utilizadas

- Python 3.10+
- Vertex AI (Gemini + Embeddings)
- LangChain
- ChromaDB (almacenamiento vectorial)
- NumPy

---

## 📂 Estructura del proyecto

```
project/
│
├── data/
│   └── knowledge_base.txt
│
│── main.py
│── pipeline.py
│
│── utils/
│  ├── chunker.py
│  ├── embeddings.py
│  └── search.py
│
│── agents/
│   └── evaluator_agent.py
│
│
├── README.md
└── .env.example
```

---

## 🔄 Pipeline de Indexación

El pipeline de indexación realiza:

1. **Carga del documento**
2. **Chunking**
   - Método: `RecursiveCharacterTextSplitter`
   - Tamaño: 500 caracteres
   - Overlap: 100 caracteres

3. **Generación de embeddings**
4. **Almacenamiento en ChromaDB**

👉 Esto genera una base de conocimiento consultable.

---

## 🔍 Pipeline de Consulta

El pipeline de consulta realiza:

1. **Embedding de la pregunta**
2. **Búsqueda vectorial (k-NN)**
   - Métrica: similitud coseno
   - Se seleccionan los top_k chunks más relevantes

3. **Construcción del contexto**
4. **Generación de respuesta con LLM**
5. **Salida en formato JSON**

---

## 📥 Formato de salida

```json
{
  "user_question": "¿Cómo recupero mi contraseña?",
  "system_answer": "Para recuperar tu contraseña...",
  "chunks_related": [
    {
      "chunk": "...",
      "score": 0.82,
      "chunk_id": 3
    }
  ]
}
```

---

## 🧪 Evaluador

El sistema incluye un agente evaluador que analiza:

- Relevancia de los chunks
- Exactitud de la respuesta
- Completitud

Output:

```json
{
  "relevance": 8,
  "accuracy": 9,
  "completeness": 7,
  "final_score": 8,
  "reason": "La respuesta es correcta pero podría ser más completa..."
}
```

---

## 🛠️ Instalación

### 1. Clonar repositorio

```bash
git clone https://github.com/belen-dominguez/ai-faq-chatbot.git
```

### 2. Crear entorno virtual

```bash
python -m venv venv
source venv/bin/activate  # Mac/Linux
venv\Scripts\activate     # Windows
```

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

---

## 🔐 Configuración

Crear archivo `.env` basado en `.env.example`:

---

## ▶️ Uso

Ejecutar el pipeline completo:

```bash
python main.py
```

El sistema:

- construye el índice
- ejecuta preguntas de prueba
- muestra respuestas en consola
