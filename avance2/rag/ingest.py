"""Ingesta: corpus/ -> lectura -> chunking -> embeddings -> Chroma.

    python -m rag.ingest                  # reconstruye TODO el indice
    python -m rag.ingest --muestras 3     # ademas imprime 3 chunks de ejemplo
    python -m rag.ingest --sin-indexar    # solo informe (no escribe en Chroma)

Es idempotente: se puede ejecutar las veces que se quiera; el resultado es el
mismo indice (ver store.reconstruir).
"""
from __future__ import annotations

import argparse
import collections
import statistics
import sys
import time
from pathlib import Path

from . import config
from .chunking import Chunk, trocear_documento
from .readers import LECTORES, DocumentoLeido, leer_documento


def descubrir(corpus: Path) -> list[Path]:
    return sorted(
        p for p in corpus.rglob("*")
        if p.is_file() and p.suffix.lower() in LECTORES and not p.name.startswith(".")
    )


def leer_corpus(corpus: Path) -> tuple[list[DocumentoLeido], list[str]]:
    """Lee todos los archivos. Un archivo ilegible NO detiene la ingesta: se
    reporta y se sigue (mejor un indice parcial avisado que ninguno)."""
    docs, avisos, vistos = [], [], {}
    for ruta in descubrir(corpus):
        try:
            doc = leer_documento(ruta)
        except Exception as e:  # noqa: BLE001 - se informa, no se oculta
            avisos.append(f"NO SE PUDO LEER {ruta.name}: {type(e).__name__}: {e}")
            continue
        if doc.sha256 in vistos:
            # Dos copias identicas duplicarian cada fragmento y la recuperacion
            # devolveria el mismo texto varias veces en el top_k.
            avisos.append(f"DUPLICADO: {ruta.name} es identico a {vistos[doc.sha256]} (se omite)")
            continue
        vistos[doc.sha256] = ruta.name
        for a in doc.advertencias:
            avisos.append(f"{ruta.name}: {a}")
        if not doc.bloques:
            avisos.append(f"{ruta.name}: sin texto extraible (se omite)")
            continue
        docs.append(doc)
    return docs, avisos


def construir_chunks(docs: list[DocumentoLeido], modo: str, contar_tokens) -> list[Chunk]:
    chunks: list[Chunk] = []
    for d in docs:
        chunks.extend(trocear_documento(
            d, modo, contar_tokens=contar_tokens, prefijo=config.PREFIJO_PASAJE,
            max_tokens=config.EMBEDDING_MAX_TOKENS))
    return chunks


def informe(docs, chunks, avisos, contar_tokens) -> None:
    print("\n=== DOCUMENTOS ===")
    for d in docs:
        n = sum(1 for c in chunks if c.fuente == d.archivo)
        print(f"  {d.archivo[:52]:52s} {d.formato:5s} {len(d.bloques):3d} secciones -> {n:3d} chunks | {d.titulo[:40]}")
    if avisos:
        print("\n=== AVISOS ===")
        for a in avisos:
            print("  !", a)
    largos = [len(c.texto) for c in chunks]
    toks = [contar_tokens(config.PREFIJO_PASAJE + c.texto_embedding) for c in chunks]
    print("\n=== CHUNKS ===")
    print(f"  total: {len(chunks)}")
    print(f"  caracteres  min/mediana/max: {min(largos)} / {int(statistics.median(largos))} / {max(largos)}")
    print(f"  tokens      min/mediana/max: {min(toks)} / {int(statistics.median(toks))} / {max(toks)}"
          f"   (tope del modelo: {config.EMBEDDING_MAX_TOKENS})")
    print(f"  por tipo: {dict(collections.Counter(c.tipo for c in chunks))}")
    exceden = [c for c in chunks if c.excede_tokens]
    if exceden:
        print(f"  ATENCION: {len(exceden)} chunk(s) indivisibles superan el tope (el embedding ve solo el inicio):")
        for c in exceden:
            print(f"    - {c.chunk_id} | {c.seccion[:60]} | {len(c.texto)} chars")


def asegurar_indice() -> str:
    """Para la app: si el indice no existe o no corresponde al corpus y a los
    parametros actuales, lo reconstruye (el indice no se sube al repo; en un
    despliegue nuevo se construye en el primer arranque). Devuelve 'vigente' |
    'reconstruido'. Leer el corpus es barato; torch solo se carga si hay que reconstruir."""
    from . import store
    docs, _ = leer_corpus(config.CORPUS_DIR)
    if docs and store.indice_vigente(store.huella([d.sha256 for d in docs], config.EMBEDDING_TEXTO_MODO)):
        return "vigente"
    if main([]) != 0:
        raise RuntimeError("No se pudo construir el indice (corpus vacio o ilegible).")
    return "reconstruido"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Reconstruye el indice vectorial desde corpus/")
    ap.add_argument("--corpus", type=Path, default=config.CORPUS_DIR)
    ap.add_argument("--modo", default=config.EMBEDDING_TEXTO_MODO, choices=["crudo", "limpio", "verbal"])
    ap.add_argument("--muestras", type=int, default=0, help="imprime N chunks de ejemplo")
    ap.add_argument("--sin-indexar", action="store_true", help="solo informe, sin escribir en Chroma")
    args = ap.parse_args(argv)

    if not args.corpus.exists() or not descubrir(args.corpus):
        print(f"El corpus esta vacio o no existe: {args.corpus}\n"
              f"Formatos admitidos: {', '.join(sorted(LECTORES))}")
        return 1

    t0 = time.time()
    # embeddings se importa aqui: cargar torch tarda y solo hace falta a partir de este punto
    from . import embeddings, store

    docs, avisos = leer_corpus(args.corpus)
    chunks = construir_chunks(docs, args.modo, embeddings.contar_tokens)
    informe(docs, chunks, avisos, embeddings.contar_tokens)

    if args.muestras:
        print(f"\n=== {args.muestras} CHUNKS DE EJEMPLO ===")
        paso = max(len(chunks) // args.muestras, 1)
        for c in chunks[::paso][: args.muestras]:
            print(f"\n--- {c.chunk_id} | tipo={c.tipo} | {c.documento} > {c.seccion} | {len(c.texto)} chars")
            print(c.texto[:900] + (" [...]" if len(c.texto) > 900 else ""))

    if args.sin_indexar:
        return 0

    vectores = embeddings.embeber_pasajes([c.texto_embedding for c in chunks])
    huella = store.huella([d.sha256 for d in docs], args.modo)
    n = store.reconstruir(chunks, vectores, huella)
    print(f"\nIndice reconstruido: {n} chunks en {config.CHROMA_DIR} "
          f"(modelo {config.MODELO_EMBEDDINGS}, dim {vectores.shape[1]}, "
          f"huella {huella}) en {time.time()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
