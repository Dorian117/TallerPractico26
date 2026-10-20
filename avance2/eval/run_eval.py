"""Ejecuta el pipeline RAG sobre el dataset y (opcionalmente) lo puntua con Ragas.

    python -m eval.run_eval --nombre base                      # DESARROLLO: cache o LLM falso, 0 gasto
    python -m eval.run_eval --nombre base --limite 3 --live     # 3 preguntas reales con el modelo del tutor
    python -m eval.run_eval --nombre base --live                # corrida real; juez = LLM_MODEL (barato)
    python -m eval.run_eval --nombre final --live --juez-final  # corrida final; juez = LLM_JUDGE_MODEL

Control de gasto:
  * Sin --live NO se llama a ninguna API: se sirve la cache real si existe y, si
    no, un LLM falso. Esas corridas se guardan como <nombre>.OFFLINE.json y NO
    son resultados validos.
  * Con --live, toda llamada (tutor y juez) pasa por la cache en disco
    (hash de modelo + prompt): relanzar tras un corte o un limite de uso
    continua sin volver a pagar lo ya hecho.
  * El juez de Ragas usa el modelo barato salvo con --juez-final.
  * Ante CUALQUIER error la corrida se detiene y lo explica (no se reintenta
    por su cuenta, salvo 429/5xx dentro de la capa LLM).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import unicodedata
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag import config  # noqa: E402
from rag.llm import MENSAJE_LIMITE, LimiteDeUsoError, crear_llm  # noqa: E402

RESULTADOS = Path(__file__).resolve().parent / "results"
DATASET_DEFECTO = Path(__file__).resolve().parent / "dataset.jsonl"
TIPOS = {"definicion", "procedimiento", "ejercicio", "seguimiento", "fuera_de_corpus"}
FRASE_ABSTENCION = "no encontre esto en el material del curso"
METRICAS = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]


# ---------------------------------------------------------------------------
def _norm(texto: str) -> str:
    sin = unicodedata.normalize("NFD", texto.casefold())
    return "".join(c for c in sin if unicodedata.category(c) != "Mn")


def se_abstiene(respuesta: str) -> bool:
    """El prompt exige empezar con 'No encontré esto en el material del curso'."""
    return FRASE_ABSTENCION in _norm(respuesta)[:250]


def cargar_dataset(ruta: Path, limite: int | None = None) -> list[dict]:
    if not ruta.exists():
        raise FileNotFoundError(f"No existe el dataset: {ruta}")
    filas = []
    for n, linea in enumerate(ruta.read_text(encoding="utf-8").splitlines(), 1):
        if not linea.strip():
            continue
        f = json.loads(linea)
        faltan = {"question", "ground_truth", "tipo", "fuente_esperada"} - set(f)
        if faltan or f["tipo"] not in TIPOS:
            raise ValueError(f"{ruta.name}:{n}: faltan campos {sorted(faltan)} o tipo invalido ({f.get('tipo')!r})")
        filas.append(f)
    return filas[:limite] if limite else filas


def generar(filas: list[dict], llm, top_k: int | None = None) -> list[dict]:
    """Pipeline real por pregunta (recuperacion local + LLM). Muestra el progreso."""
    from rag.generate import Turno, responder

    salida = []
    for i, f in enumerate(filas, 1):
        historial = [Turno(t["role"], t["content"]) for t in f.get("historial", [])]
        t0 = time.time()
        r = responder(f["question"], historial, top_k=top_k, llm=llm)
        print(f"[{i}/{len(filas)}] {f['tipo']:16s} {'cache' if r.desde_cache else 'nuevo':5s} "
              f"{len(r.fragmentos)} frag | {r.tokens_entrada}/{r.tokens_salida} tok | "
              f"{time.time() - t0:4.1f}s | {f['question'][:58]}", flush=True)
        salida.append({
            **f,
            "respuesta": r.texto,
            # para preguntas de seguimiento Ragas usa la pregunta autonoma (condensada)
            "pregunta_evaluada": r.pregunta_recuperacion if historial else f["question"],
            "contextos": [fr.texto for fr in r.fragmentos],
            "fuentes_recuperadas": [{"fuente": fr.fuente, "seccion": fr.seccion, "distancia": round(fr.distancia, 4)}
                                    for fr in r.fragmentos],
            "citas_invalidas": r.citas_invalidas,
            "se_abstuvo": se_abstiene(r.texto),
            "tokens_entrada": r.tokens_entrada, "tokens_salida": r.tokens_salida,
            "desde_cache": r.desde_cache, "avisos": r.avisos,
        })
    return salida


def metricas_abstencion(filas: list[dict]) -> dict:
    """Las metricas de Ragas no estan bien definidas para abstenciones
    (no hay 'referencia' que comparar), asi que las fuera_de_corpus se
    reportan aparte con metricas propias."""
    fuera = [f for f in filas if f["tipo"] == "fuera_de_corpus"]
    dentro = [f for f in filas if f["tipo"] != "fuera_de_corpus"]
    ok = sum(f["se_abstuvo"] for f in fuera)
    return {
        "n_fuera_de_corpus": len(fuera),
        "tasa_abstencion_correcta": ok / len(fuera) if fuera else None,
        "tasa_alucinacion": (len(fuera) - ok) / len(fuera) if fuera else None,
        "n_dentro_de_corpus": len(dentro),
        "tasa_abstencion_indebida": sum(f["se_abstuvo"] for f in dentro) / len(dentro) if dentro else None,
        "respuestas_con_citas_invalidas": sum(bool(f["citas_invalidas"]) for f in filas),
    }


def evaluar_con_ragas(filas: list[dict], modelo_juez: str) -> tuple[list[dict], dict]:
    """4 metricas de Ragas sobre las preguntas DENTRO del corpus."""
    from ragas import EvaluationDataset, evaluate
    from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness

    from eval import juez as mod_juez

    dentro = [f for f in filas if f["tipo"] != "fuera_de_corpus"]
    if not dentro:
        return [], {}
    juez = mod_juez.crear_juez(modelo_juez)
    ds = EvaluationDataset.from_list([{
        "user_input": f["pregunta_evaluada"], "retrieved_contexts": f["contextos"],
        "response": f["respuesta"], "reference": f["ground_truth"]} for f in dentro])
    print(f"\nRagas: {len(dentro)} preguntas x 4 metricas con el juez {modelo_juez} ...", flush=True)
    res = evaluate(ds, metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
                   llm=juez, embeddings=mod_juez.crear_embeddings(), run_config=mod_juez.run_config(),
                   raise_exceptions=False, show_progress=True)
    if juez.limite is not None:      # Ragas convierte los errores en NaN: aqui se hace visible
        raise juez.limite
    df = res.to_pandas()
    for f, (_, fila) in zip(dentro, df.iterrows()):
        f["ragas"] = {m: (None if fila.get(m) != fila.get(m) else float(fila[m])) for m in METRICAS}
    promedios, nans = {}, {}
    for m in METRICAS:
        vals = [f["ragas"][m] for f in dentro if f["ragas"][m] is not None]
        promedios[m] = sum(vals) / len(vals) if vals else None
        nans[m] = len(dentro) - len(vals)
    return dentro, {"promedios": promedios, "preguntas_sin_puntaje": nans,
                    "uso_juez": dict(mod_juez.USO)}


def guardar(nombre: str, filas: list[dict], meta: dict, offline: bool) -> Path:
    RESULTADOS.mkdir(parents=True, exist_ok=True)
    # Nombres explicitos: Path.with_suffix reemplazaria ".OFFLINE" por ".json" y una
    # corrida simulada acabaria guardada como si fuera un resultado valido.
    stem = f"{nombre}.OFFLINE" if offline else nombre
    ruta_json, ruta_csv = RESULTADOS / f"{stem}.json", RESULTADOS / f"{stem}.csv"
    ruta_json.write_text(json.dumps({**meta, "filas": filas}, ensure_ascii=False, indent=2), encoding="utf-8")
    import csv
    with ruta_csv.open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["tipo", "pregunta", "se_abstuvo", "citas_invalidas", "tokens_entrada", "tokens_salida", *METRICAS])
        for f in filas:
            r = f.get("ragas") or {}
            w.writerow([f["tipo"], f["question"], f["se_abstuvo"], len(f["citas_invalidas"]),
                        f["tokens_entrada"], f["tokens_salida"], *[r.get(m) for m in METRICAS]])
    return ruta_json


# ---------------------------------------------------------------------------
def ejecutar(args, llm=None) -> int:
    """Devuelve 0 = ok, 1 = entrada invalida, 2 = error, 3 = limite de uso."""
    try:
        filas = cargar_dataset(Path(args.dataset), args.limite)
    except (FileNotFoundError, ValueError) as e:
        print(f"ERROR: {e}")
        return 1
    live = bool(args.live)
    llm = llm or crear_llm("tutor", live=live, usar_cache=True)
    modelo_juez = config.LLM_JUDGE_MODEL if args.juez_final else config.LLM_MODEL
    print(f"Corrida '{args.nombre}': {len(filas)} preguntas | modo={'LIVE' if live else 'OFFLINE (cache o LLM falso)'} "
          f"| tutor={config.LLM_MODEL} | juez={modelo_juez if live and not args.sin_ragas else '-'}"
          f"{' (FINAL)' if args.juez_final else ''}", flush=True)

    ya = []
    try:
        ya = generar(filas, llm, args.top_k)
        meta = {
            "nombre": args.nombre, "fecha": datetime.now().isoformat(timespec="seconds"),
            "modo": "live" if live else "offline",
            "modelo_tutor": config.LLM_MODEL, "proveedor": config.LLM_PROVIDER,
            "juez_final": bool(args.juez_final),
            "parametros": {"chunk_tamano": config.CHUNK_TAMANO, "chunk_solape": config.CHUNK_SOLAPE,
                           "top_k": args.top_k or config.TOP_K, "umbral": config.UMBRAL_DISTANCIA,
                           "embeddings": config.MODELO_EMBEDDINGS, "texto_embedding": config.EMBEDDING_TEXTO_MODO,
                           "max_tokens_respuesta": config.MAX_TOKENS_RESPUESTA},
            "dataset": {"ruta": str(args.dataset), "n": len(filas),
                        "sha256": hashlib.sha256(Path(args.dataset).read_bytes()).hexdigest()[:16]},
            "abstencion": metricas_abstencion(ya),
        }
        if not live:
            meta["advertencia"] = "OFFLINE: respuestas de cache o simuladas. NO son resultados validos."
            print("\nModo OFFLINE: se omite Ragas (necesita --live). Resultado NO valido.")
        elif args.sin_ragas:
            print("\n--sin-ragas: solo generacion y metricas de abstencion.")
        else:
            _, ragas = evaluar_con_ragas(ya, modelo_juez)
            meta["modelo_juez"] = modelo_juez
            meta["ragas"] = ragas
    except LimiteDeUsoError:
        print(f"\nERROR: {MENSAJE_LIMITE}.\nSe detuvo tras {len(ya)} de {len(filas)} preguntas generadas. "
              "Lo ya hecho esta en la cache en disco: al relanzar el mismo comando continua sin volver a pagarlo.")
        return 3
    except Exception as e:  # noqa: BLE001 - se explica y se detiene, sin reintentos propios
        print(f"\nERROR: {type(e).__name__}: {str(e)[:400]}\nLa corrida se detuvo. Nada se reintento por su cuenta.")
        return 2

    ruta = guardar(args.nombre, ya, meta, offline=not live)
    print(f"\nGuardado: {ruta}")
    print("Abstencion:", json.dumps(meta["abstencion"], ensure_ascii=False))
    if "ragas" in meta:
        print("Ragas promedios:", json.dumps(meta["ragas"].get("promedios"), ensure_ascii=False))
        print("Uso del juez:", meta["ragas"].get("uso_juez"))
    return 0


def main(argv=None) -> int:
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nombre", required=True, help="nombre de la corrida (archivo de resultados)")
    ap.add_argument("--dataset", default=str(DATASET_DEFECTO))
    ap.add_argument("--limite", type=int, default=None, help="usar solo las primeras N preguntas")
    ap.add_argument("--live", action="store_true", help="llamadas REALES a la API (con cache en disco)")
    ap.add_argument("--juez-final", action="store_true", help="juez = LLM_JUDGE_MODEL (corridas finales)")
    ap.add_argument("--sin-ragas", action="store_true", help="solo generar y medir abstencion")
    ap.add_argument("--top-k", type=int, default=None)
    return ejecutar(ap.parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
