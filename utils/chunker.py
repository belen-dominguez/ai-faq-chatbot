from langchain_text_splitters import RecursiveCharacterTextSplitter


def document_load_and_generate_chunk(filepath: str, chunk_size: int = 500, chunk_overlap: int = 100) -> list[str]:
    """
    Carga un documento .txt y lo divide en chunks con solapamiento.

    Estrategia: RecursiveCharacterTextSplitter — intenta dividir por párrafos,
    luego por oraciones, luego por palabras. Preserva el contexto semántico
    mejor que un split fijo por caracteres.

    Argumentos:
        filepath: ruta al archivo .txt
        chunk_size: tamaño máximo de cada chunk en caracteres
        chunk_overlap: solapamiento entre chunks para no perder contexto

    Retorna:
        Lista de strings, cada uno es un chunk del documento
    """
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " "]
    )

    chunks = splitter.split_text(text)
    print(f"✅ Documento cargado: {len(chunks)} chunks generados")
    return chunks