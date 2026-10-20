"""Benchmark RAPIDO de embeddings (Fase 1.4): modelo x modo de texto.

    python -m eval.bench_embeddings

Compara, sobre el corpus real y 20 preguntas redactadas leyendo los apuntes:
  - hit@1 / hit@4 / MRR: ¿aparece la seccion correcta entre los primeros k?
  - AUC fuera-de-corpus: probabilidad de que la distancia del mejor resultado
    sea MENOR para una pregunta del corpus que para una de fuera (1.0 =
    separacion perfecta, 0.5 = azar). Es lo que importa para el umbral.

LIMITES (honestos): son solo 15 + 5 preguntas escritas por quien implementa;
sirve para descartar opciones claramente peores, NO para afinar decimales.
La evaluacion formal (Ragas) es la Fase 2.
Todo en memoria: no toca el indice persistente de Chroma.
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

# (pregunta, [(fragmento-del-nombre-de-archivo, texto-en-la-seccion), ...] aceptables)
EN_CORPUS = [
    ("¿Qué sustitución uso cuando el integrando tiene raíz de a^2 - x^2?", [("susticion", "Casos est")]),
    ("¿Cuál es la regla ILATE y para qué sirve?", [("partes", "ILATE")]),
    ("Deduce la fórmula de integración por partes a partir de la regla del producto", [("partes", "Deducci")]),
    ("¿Qué es una integral cíclica? Por ejemplo la integral de e^x por sen(x)", [("partes", "Ejemplo 2")]),
    ("¿Qué es una antiderivada o primitiva de una función?", [("antiderivaci", "Actividad Inicial")]),
    ("¿Cómo se resuelve un problema de valor inicial con segunda derivada?", [("antiderivaci", "Valor Inicial")]),
    ("¿Cómo integro potencias pares de coseno con identidades de ángulo doble?", [("Areas", "Ejemplo 1")]),
    ("¿Cuál es la estrategia para integrales con potencias de tangente y secante?", [("Areas", "tan^m")]),
    ("¿Cómo simplifico el producto (x^3+5)(x^3-5) antes de integrar?", [("simplificaci", "Ejercicio b)")]),
    ("¿Qué estrategias existen para reescribir el integrando antes de integrar?", [("simplificaci", "Algoritmo")]),
    ("¿Cómo identifico u y du al hacer un cambio de variable?", [("sustitucion", "Cambio de Variable")]),
    ("¿Cómo se integra la secante de x?", [("Areas", "Ejemplo 4"), ("susticion", "Integrales b")]),
    ("Calcula la integral de dx entre x^2 por raíz de (9 - x^2)", [("susticion", "Ejemplo 1")]),
    ("¿Cuál es la integral del logaritmo natural?", [("partes", "Ejemplo 1")]),
    ("¿En qué consiste el método tabular para integrar por partes?", [("partes", "Tabular")]),
]
FUERA = [
    "¿Cómo calculo el volumen de un sólido de revolución por el método de discos?",
    "¿Cómo se calcula el área entre dos curvas?",
    "¿Qué son las sumas de Riemann y la notación sigma?",
    "¿Cómo resuelvo una ecuación diferencial separable?",
    "¿Cuál es la capital de Francia y quién construyó la Torre Eiffel?",
]

# modelo, prefijo consulta, prefijo pasaje, max tokens
MODELOS = {
    "e5-small": ("intfloat/multilingual-e5-small", "query: ", "passage: ", 512),
    "minilm-l12": ("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", "", "", 128),
}
CONFIGS = [("e5-small", "crudo"), ("e5-small", "limpio"), ("e5-small", "verbal"),
           ("minilm-l12", "limpio"), ("minilm-l12", "verbal")]


def coincide(f, aceptables) -> bool:
    return any(k.lower() in f["fuente"].lower() and s.lower() in f["seccion"].lower() for k, s in aceptables)


def auc(dist_dentro: list[float], dist_fuera: list[float]) -> float:
    pares = [(d_in < d_out) + 0.5 * (d_in == d_out) for d_in in dist_dentro for d_out in dist_fuera]
    return float(np.mean(pares))


def main() -> None:
    from sentence_transformers import SentenceTransformer

    docs, _ = leer_corpus(config.CORPUS_DIR)
    cargados: dict[str, SentenceTransformer] = {}
    resultados = []
    print(f"{'config':22s} {'chunks':>6s} {'hit@1':>6s} {'hit@4':>6s} {'MRR':>6s} {'AUC':>6s} | top1 dentro (min/max)  top1 fuera (min/max)")
    for nombre, modo in CONFIGS:
        hf, pq, pp, max_tok = MODELOS[nombre]
        if nombre not in cargados:
            try:
                m = SentenceTransformer(hf, device="cpu", local_files_only=True)
            except OSError:   # primera vez: se descarga (solo pesos)
                m = SentenceTransformer(hf, device="cpu")
            m.max_seq_length = max_tok
            cargados[nombre] = m
        modelo = cargados[nombre]
        contar = lambda t: len(modelo.tokenizer.encode(t, add_special_tokens=True))  # noqa: E731
        chunks = [c for d in docs for c in trocear_documento(
            d, modo, contar_tokens=contar, prefijo=pp, max_tokens=max_tok)]
        vec = modelo.encode([pp + c.texto_embedding for c in chunks], normalize_embeddings=True,
                            batch_size=16, show_progress_bar=False)
        metas = [{"fuente": c.fuente, "seccion": c.seccion} for c in chunks]

        def top(pregunta, k=4):
            q = modelo.encode([pq + pregunta], normalize_embeddings=True, show_progress_bar=False)[0]
            dist = 1 - vec @ q          # distancia coseno
            orden = np.argsort(dist)[:k]
            return [(metas[i], float(dist[i])) for i in orden]

        h1 = h4 = 0
        rr = []
        d_in = []
        detalle = []
        for preg, ok in EN_CORPUS:
            r = top(preg)
            d_in.append(r[0][1])
            rangos = [i for i, (m, _) in enumerate(r, 1) if coincide(m, ok)]
            h1 += bool(rangos and rangos[0] == 1)
            h4 += bool(rangos)
            rr.append(1 / rangos[0] if rangos else 0.0)
            detalle.append(dict(pregunta=preg, rango_esperado=rangos[0] if rangos else None,
                                top1=f"{r[0][0]['fuente'][:30]} > {r[0][0]['seccion'][-60:]}", dist_top1=round(r[0][1], 4)))
        fuera_top = [(p, top(p)[0]) for p in FUERA]
        d_out = [t[1][1] for t in fuera_top]
        n = len(EN_CORPUS)
        fila = dict(config=f"{nombre}/{modo}", chunks=len(chunks), hit1=h1 / n, hit4=h4 / n,
                    mrr=float(np.mean(rr)), auc=auc(d_in, d_out),
                    top1_dentro=[min(d_in), max(d_in)], top1_fuera=[min(d_out), max(d_out)],
                    detalle_dentro=detalle,
                    detalle_fuera=[dict(pregunta=p, dist_top1=round(t[1], 4),
                                        top1=f"{t[0]['fuente'][:30]} > {t[0]['seccion'][-60:]}") for p, t in fuera_top])
        if (nombre, modo) == ("e5-small", "limpio"):
            print("\n  [e5-small/limpio] preguntas con la seccion esperada fuera del rango 1:")
            for d in detalle:
                if d["rango_esperado"] != 1:
                    print(f"    rango_esperado={d['rango_esperado']}  d={d['dist_top1']}  {d['pregunta'][:70]}\n"
                          f"        top1 fue: {d['top1']}")
            print("  [e5-small/limpio] preguntas FUERA del corpus (distancia del top-1):")
            for d in sorted(fila["detalle_fuera"], key=lambda x: x["dist_top1"]):
                print(f"    d={d['dist_top1']}  {d['pregunta'][:70]}")
            print()
        resultados.append(fila)
        print(f"{fila['config']:22s} {len(chunks):6d} {h1/n:6.2f} {h4/n:6.2f} {fila['mrr']:6.2f} {fila['auc']:6.2f} | "
              f"{min(d_in):.3f}/{max(d_in):.3f}            {min(d_out):.3f}/{max(d_out):.3f}")

    salida = Path(__file__).parent / "results" / "bench_embeddings.json"
    salida.write_text(json.dumps(resultados, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nGuardado en {salida}")


if __name__ == "__main__":
    main()
