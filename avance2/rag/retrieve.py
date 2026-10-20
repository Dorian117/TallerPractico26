"""Recuperacion: pregunta -> top_k fragmentos mas cercanos (coseno).

    python -m rag.retrieve "como resuelvo una integral con raiz de a^2 - x^2" [mas consultas...]
"""
from __future__ import annotations

import sys
from dataclasses import dataclass

from . import config


@dataclass
class Fragmento:
    texto: str
    fuente: str
    documento: str
    seccion: str
    pagina: int
    tipo: str
    chunk_id: str
    distancia: float   # coseno: 0 = identico, 2 = opuesto
    rango: int         # 1 = el mas cercano


def recuperar(
    pregunta: str,
    top_k: int | None = None,
    umbral: float | None = None,
) -> list[Fragmento]:
    """Devuelve los `top_k` fragmentos mas cercanos a la pregunta.

    `umbral` (distancia coseno maxima) descarta los que quedan mas lejos: si
    no queda ninguno, el corpus no cubre la pregunta. Es None mientras no se
    calibre con datos (Fase 2); por eso la distancia SIEMPRE se devuelve.
    """
    from . import embeddings, store   # import tardio: carga torch/chroma

    top_k = top_k or config.TOP_K
    umbral = config.UMBRAL_DISTANCIA if umbral is None else umbral

    # La consulta se vectoriza con el prefijo "query: " y el MISMO modelo que
    # los documentos; si no, las distancias no tendrian sentido.
    vector = embeddings.embeber_consulta(pregunta)
    r = store.obtener().query(
        query_embeddings=[vector.tolist()],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )
    fragmentos = [
        Fragmento(
            texto=doc, fuente=m["fuente"], documento=m["documento"], seccion=m["seccion"],
            pagina=m["pagina"], tipo=m["tipo"], chunk_id=m["chunk_id"], distancia=d, rango=i + 1,
        )
        for i, (doc, m, d) in enumerate(zip(r["documents"][0], r["metadatas"][0], r["distances"][0]))
    ]
    if umbral is not None:
        fragmentos = [f for f in fragmentos if f.distancia <= umbral]
    return fragmentos


def main(consultas: list[str]) -> int:
    for q in consultas:
        print(f"\n{'=' * 100}\nCONSULTA: {q}")
        for f in recuperar(q):
            sec = f.seccion[:70] + ("..." if len(f.seccion) > 70 else "")
            print(f"  #{f.rango}  dist={f.distancia:.4f}  [{f.tipo:13s}] {f.documento[:34]} > {sec}")
            print("        " + f.texto[:140].replace("\n", " ") + " ...")
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    sys.exit(main(sys.argv[1:]))
