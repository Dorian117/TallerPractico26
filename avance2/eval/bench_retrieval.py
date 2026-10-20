"""Benchmark de RECUPERACION sobre el dataset real (gratis: sin LLM, sin API).

    python -m eval.bench_retrieval base

Usa las preguntas efectivamente enviadas al buscador en la corrida `base`
(`pregunta_evaluada`: la condensada en los seguimientos) y comprueba en que
posicion aparece la SECCION esperada, para varias configuraciones del indice
construidas en memoria (no toca chroma_db). Sirve para elegir, con datos, el
parametro a iterar en el paso 2.5 antes de gastar en el juez.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from rag import config  # noqa: E402
from rag.chunking import trocear_documento  # noqa: E402
from rag.ingest import leer_corpus  # noqa: E402

RES = Path(__file__).resolve().parent / "results"
# Por pregunta del dataset (orden 1..18, solo las del corpus): (parte del nombre de archivo,
# alternativas aceptables de la seccion). Se declara aqui porque el formato de
# `seccion_esperada` del dataset es descriptivo, no un patron.
ESPERADO = {
    1: ("antiderivaci", ["Actividad Inicial"]), 2: ("antiderivaci", ["Tabla de Antiderivadas", "Reglas B"]),
    3: ("sustitucion", ["Antiderivación de una Función Compuesta"]), 4: ("partes", ["Deducci"]),
    5: ("susticion", ["Casos est"]), 6: ("Areas", ["Identidades Trigonom"]),
    7: ("susticion", ["Ejemplo 1"]), 8: ("Areas", ["Ejemplo 1"]), 9: ("partes", ["Ejemplo 2"]),
    10: ("partes", ["Ejemplo 3"]), 11: ("simplificaci", ["Ejercicio f"]), 12: ("antiderivaci", ["Reglas B"]),
    13: ("simplificaci", ["Ejercicio b"]), 14: ("antiderivaci", ["Problema de Valor Inicial", "Soluci"]),
    15: ("partes", ["Ejercicios Propuestos", "Respuestas"]), 16: ("susticion", ["Casos est"]),
    17: ("partes", ["ILATE"]), 18: ("Areas", ["Ejemplo 2", "Estrategia"]),
}
VARIANTES = [  # (nombre, modo de texto embebido, tamano, solape)
    ("actual: limpio 1800/200", "limpio", 1800, 200),
    ("texto verbal", "verbal", 1800, 200),
    ("texto crudo", "crudo", 1800, 200),
    ("chunk 1000/150", "limpio", 1000, 150),
    ("chunk 600/100", "limpio", 600, 100),
]


def main(nombre: str = "base") -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    from rag import embeddings

    datos = json.loads((RES / f"{nombre}.json").read_text(encoding="utf-8"))
    preguntas = [f["pregunta_evaluada"] for f in datos["filas"][:18]]
    docs, _ = leer_corpus(config.CORPUS_DIR)
    qv = embeddings.embeber_consultas(preguntas)   # misma vectorizacion de consulta que el pipeline
    qv = np.stack([embeddings.embeber_consulta(q) for q in preguntas])
    resumen = []
    print(f"{'variante':26s} {'chunks':>6s} | hit@1 hit@4 hit@6 hit@8 | rango medio | no rescatadas (4,6,8)")
    for nombre_v, modo, tam, sol in VARIANTES:
        config.CHUNK_TAMANO, config.CHUNK_SOLAPE = tam, sol
        chunks = [c for d in docs for c in trocear_documento(
            d, modo, contar_tokens=embeddings.contar_tokens, prefijo=config.PREFIJO_PASAJE,
            max_tokens=config.EMBEDDING_MAX_TOKENS)]
        cv = embeddings.embeber_pasajes([c.texto_embedding for c in chunks])
        rangos = []
        for i, qvec in enumerate(qv, 1):
            arch, secs = ESPERADO[i]
            orden = np.argsort(1 - cv @ qvec)
            r = next((k for k, j in enumerate(orden[:40], 1)
                      if arch in chunks[j].fuente and any(s in chunks[j].seccion for s in secs)), None)
            rangos.append(r)
        def hit(k): return sum(1 for r in rangos if r and r <= k)
        vistos = [r for r in rangos if r]
        fallan = {k: [i for i, r in enumerate(rangos, 1) if not (r and r <= k)] for k in (4, 6, 8)}
        print(f"{nombre_v:26s} {len(chunks):6d} | {hit(1):>5d} {hit(4):>5d} {hit(6):>5d} {hit(8):>5d} | "
              f"{np.mean(vistos):10.1f} | {fallan[4]} {fallan[6]} {fallan[8]}")
        resumen.append({"variante": nombre_v, "chunks": len(chunks), "rangos": rangos,
                        "hit1": hit(1), "hit4": hit(4), "hit6": hit(6), "hit8": hit(8)})
    (RES / f"bench_retrieval_{nombre}.json").write_text(json.dumps(resumen, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n(n = 18 preguntas del corpus; 'rango' = posicion de la seccion esperada, None = no aparece en el top-40)")
    print("Rangos de la variante actual:", resumen[0]["rangos"])


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "base")
