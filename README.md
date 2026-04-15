Eres un/a ingeniero/a de IA en una empresa de HR SaaS con mucha documentación. El equipo de soporte al cliente recibe más de 200 preguntas repetitivas al día sobre políticas, funcionalidades y procedimientos que ya están documentados en FAQs internas y guías. Tu equipo necesita un Chatbot inteligente de soporte para FAQs que pueda responder al instante recuperando información relevante de la documentación de la empresa, sin requerir búsquedas manuales ni intervención de agentes.

🎯Objetivos

Implementar un sistema de FAQs basado en RAG que procese un documento de texto plano, lo divida en chunks de forma inteligente (mínimo 20 chunks), genere embeddings y los almacene para una recuperación eficiente. Esto crea una base de conocimiento consultable a partir de documentación no estructurada.
Construir un pipeline de consulta que acepte preguntas de usuarios, realice búsqueda vectorial usando métodos k-NN/ANN/rango/híbridos, recupere los chunks relevantes y genere respuestas precisas con un LLM. El sistema debe devolver JSON estructurado con user_question, system_answer y chunks_related para asegurar transparencia y auditabilidad.
Opcional: implementar un agente evaluador que puntúe la calidad de la respuesta (0–10) según la relevancia de los chunks, la precisión y la completitud. Esto aporta aseguramiento automático de calidad para el sistema RAG.

📢Consigna

Crea un chatbot de soporte para FAQs usando RAG que responda preguntas basándose en un documento del sistema. El sistema debe procesar un documento de texto plano, dividirlo en chunks (al menos 20 chunks) y generar embeddings. Para cada pregunta del usuario, devuelve una salida en JSON que contenga user_question, system_answer y chunks_related usados para generar la respuesta. Utiliza métodos de búsqueda vectorial (p. ej., k-NN, ANN, rango o híbridos) para encontrar de forma eficiente los chunks relevantes. Bonus: implementa un agente evaluador que reciba user_question, system_answer y chunks_related y devuelva un puntaje de 0 a 10 junto con una justificación del resultado.
