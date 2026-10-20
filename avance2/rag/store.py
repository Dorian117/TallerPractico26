"""Base vectorial: Chroma persistente y propia, distancia coseno.

El indice vive en avance2/chroma_db/ (una carpeta en TU disco). No hay
servidor ni servicio externo: PersistentClient escribe SQLite + ficheros HNSW.
"""
from __future__ import annotations

import hashlib

import chromadb
from chromadb.config import Settings

from . import config
from .chunking import Chunk


def _cliente() -> chromadb.ClientAPI:
    # anonymized_telemetry=False: Chroma, por defecto, envia estadisticas de
    # uso anonimas. Se desactiva explicitamente por la restriccion de privacidad.
    return chromadb.PersistentClient(
        path=str(config.CHROMA_DIR), settings=Settings(anonymized_telemetry=False),
    )


def huella(sha_documentos: list[str], modo_embedding: str) -> str:
    """Identifica el contenido y los parametros con que se construyo el indice.
    Permite saber (p. ej. al arrancar la app) si el indice existente sigue
    vigente sin reconstruirlo."""
    partes = sorted(sha_documentos) + [
        config.MODELO_EMBEDDINGS, modo_embedding, str(config.CHUNK_TAMANO),
        str(config.CHUNK_SOLAPE), str(config.CHUNK_MINIMO), str(config.EMBEDDING_MAX_TOKENS),
    ]
    return hashlib.sha256("|".join(partes).encode()).hexdigest()[:16]


def reconstruir(chunks: list[Chunk], vectores, huella_indice: str) -> int:
    """Borra y recrea la coleccion: asi `python -m rag.ingest` es IDEMPOTENTE
    (ejecutarlo 1 o 10 veces deja exactamente el mismo indice, sin duplicados
    ni fragmentos huerfanos de documentos que ya no existen)."""
    cliente = _cliente()
    try:
        cliente.delete_collection(config.COLECCION)
    except Exception:  # la coleccion no existia todavia
        pass
    col = cliente.create_collection(
        config.COLECCION,
        # distancia coseno = 1 - similitud coseno. 0 identico, 1 ortogonal, 2 opuesto
        metadata={"hnsw:space": "cosine", "huella": huella_indice},
        # embedding_function=None: los vectores se calculan SIEMPRE nosotros
        # (local, con E5). Evita que Chroma use su modelo por defecto, que
        # ademas descargaria otro modelo sin avisar.
        embedding_function=None,
    )
    col.add(
        ids=[c.chunk_id for c in chunks],
        embeddings=vectores.tolist(),
        documents=[c.texto for c in chunks],
        metadatas=[{
            "fuente": c.fuente,
            "documento": c.documento,
            "seccion": c.seccion,
            # Chroma no admite None en metadatos: 0 significa "el formato no
            # tiene paginas" (LaTeX, Markdown, HTML...)
            "pagina": c.pagina,
            "chunk_id": c.chunk_id,
            "tipo": c.tipo,
            "formato": c.formato,
            "n_chunk": c.n_chunk,
            "excede_tokens": c.excede_tokens,
        } for c in chunks],
    )
    return col.count()


def obtener():
    return _cliente().get_collection(config.COLECCION, embedding_function=None)


def indice_vigente(huella_esperada: str) -> bool:
    try:
        return obtener().metadata.get("huella") == huella_esperada
    except Exception:
        return False
