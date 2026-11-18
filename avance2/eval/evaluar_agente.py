"""Evaluacion del agente (Avance 3): Ragas (pipeline fijo vs agente), casos de comportamiento y costo.

    python -m eval.evaluar_agente --todo --live                 # las tres partes
    python -m eval.evaluar_agente --ragas --live --limite 3     # prueba rapida de una parte
    python -m eval.evaluar_agente --casos --live
    python -m eval.evaluar_agente --costo --live
    python -m eval.evaluar_agente --todo                        # SIN --live: no llama a ninguna API (resultado NO valido)

Control de gasto (heredado del Avance 2):
  * Sin --live no se llama a ninguna API y el resultado se guarda como *.OFFLINE.* (no es valido).
  * El juez de Ragas es el modelo barato (LLM_MODEL) salvo con --juez-final (LLM_JUDGE_MODEL, ~15x mas caro).
    Para comparar con el Avance 2 hay que usar el MISMO juez que la linea base (ver --base).
  * El agente y el pipeline fijo se ejecutan SIN cache en disco: asi los tiempos de la tabla de costo son reales.
    El juez si usa su cache (reanudar no vuelve a pagar lo ya puntuado).
  * Ante cualquier error la corrida se detiene y lo explica. Salida: 0 ok, 1 entrada invalida, 2 error, 3 limite de uso.

Los retrieved_contexts del agente son lo que devolvieron SUS tools (fragmentos de la base ya recortados a
MAX_CHARS, extractos web y resultados de calculo). Las preguntas de seguimiento se envian al agente con su
historial real (el agente decide su propia consulta); el pipeline fijo las condensa, como en el Avance 2.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import statistics as st
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval import compare, run_eval  # noqa: E402
from rag import config, tools_web  # noqa: E402
from rag.agent import RespuestaAgente, responder_agente  # noqa: E402
from rag.generate import Turno, responder  # noqa: E402
from rag.llm import MENSAJE_LIMITE, LimiteDeUsoError, crear_llm  # noqa: E402

RES = run_eval.RESULTADOS
CASOS_DEFECTO = Path(__file__).resolve().parent / "casos_comportamiento.json"
BASE_DEFECTO = "iter1_topk8"    # Avance 2: top_k=8 (el desplegado) con juez Haiku, el mismo juez que el agente usa por defecto
METRICAS = run_eval.METRICAS


# ---------------------------------------------------------------------------
# Medicion (agente y pipeline fijo con el mismo contador)
# ---------------------------------------------------------------------------
class Contador:
    """Envuelve un LLM y cuenta llamadas y tokens: la unica forma de medir con el mismo rasero el
    pipeline fijo (condensar + responder) y el agente."""

    def __init__(self, llm):
        self.llm, self.modelo = llm, llm.modelo
        self.llamadas = self.entrada = self.salida = 0

    def generar(self, **kw):
        r = self.llm.generar(**kw)
        self.llamadas += 1
        self.entrada += r.tokens_entrada
        self.salida += r.tokens_salida
        return r


def _contextos(r: RespuestaAgente) -> list[str]:
    """Lo que devolvieron las tools de esta pregunta, como textos para Ragas."""
    textos = [f["texto"] for f in r.fragmentos_base]
    textos += [f"{x['titulo']}: {x['extracto']}" for x in r.resultados_web]
    for c in r.llamadas_tools:
        if c.nombre in ("calcular_integral", "verificar_antiderivada") and not c.error:
            textos.append(f"{c.nombre}({json.dumps(c.argumentos, ensure_ascii=False)}) = {json.dumps(c.resultado, ensure_ascii=False)}")
    return textos


def correr_agente(filas: list[dict], llm) -> list[dict]:
    salida = []
    for i, f in enumerate(filas, 1):
        historial = [Turno(t["role"], t["content"]) for t in f.get("historial", [])]
        r = responder_agente(f["question"], historial, llm=llm)
        print(f"[agente {i}/{len(filas)}] {f['tipo']:16s} {r.n_llamadas_llm} llm | "
              f"{[c.nombre for c in r.llamadas_tools]} | {r.tokens_entrada}/{r.tokens_salida} tok | "
              f"{r.segundos:4.1f}s | {r.estado} | {f['question'][:50]}", flush=True)
        salida.append({
            **f, "respuesta": r.texto, "pregunta_evaluada": f["question"], "contextos": _contextos(r),
            "fuentes_recuperadas": [{"fuente": x["documento"], "seccion": x["seccion"], "distancia": x["distancia"]}
                                    for x in r.fragmentos_base],
            "citas_invalidas": r.citas_invalidas, "se_abstuvo": run_eval.se_abstiene(r.texto),
            "tokens_entrada": r.tokens_entrada, "tokens_salida": r.tokens_salida, "desde_cache": False,
            "avisos": r.avisos, "tools": [{"nombre": c.nombre, "argumentos": c.argumentos, "error": c.error}
                                          for c in r.llamadas_tools],
            "n_llamadas_llm": r.n_llamadas_llm, "segundos": round(r.segundos, 2), "estado": r.estado})
    return salida


def correr_pipeline_fijo(filas: list[dict], llm) -> list[dict]:
    """El pipeline del Avance 2 tal como esta desplegado (top_k = APP_TOP_K), sin cache."""
    salida = []
    for i, f in enumerate(filas, 1):
        historial = [Turno(t["role"], t["content"]) for t in f.get("historial", [])]
        c = Contador(llm)
        t0 = time.time()
        responder(f["question"], historial, top_k=config.APP_TOP_K, llm=c)
        seg = time.time() - t0
        print(f"[fijo   {i}/{len(filas)}] {f['tipo']:16s} {c.llamadas} llm | {c.entrada}/{c.salida} tok | {seg:4.1f}s", flush=True)
        salida.append({"question": f["question"], "tipo": f["tipo"], "n_llamadas_llm": c.llamadas,
                       "tokens_entrada": c.entrada, "tokens_salida": c.salida, "segundos": round(seg, 2)})
    return salida


# ---------------------------------------------------------------------------
# Costo
# ---------------------------------------------------------------------------
def resumen_costo(fijo: list[dict], agente: list[dict]) -> dict:
    def medias(rows):
        return {"llamadas_llm": st.mean(r["n_llamadas_llm"] for r in rows),
                "tokens_entrada": st.mean(r["tokens_entrada"] for r in rows),
                "tokens_salida": st.mean(r["tokens_salida"] for r in rows),
                "tokens_total": st.mean(r["tokens_entrada"] + r["tokens_salida"] for r in rows),
                "segundos_media": st.mean(r["segundos"] for r in rows),
                "segundos_mediana": st.median(r["segundos"] for r in rows)}
    # el pipeline fijo no tiene tools; el agente se desglosa por si llamo alguna
    con = [r for r in agente if r["tools"]]
    sin = [r for r in agente if not r["tools"]]
    return {"n": len(agente), "fijo": medias(fijo), "agente": medias(agente),
            "agente_con_tools": {"n": len(con), **(medias(con) if con else {})},
            "agente_sin_tools": {"n": len(sin), **(medias(sin) if sin else {})}}


def md_costo(c: dict) -> str:
    f, a = c["fijo"], c["agente"]
    fila = lambda nombre, k, fmt: f"| {nombre} | {fmt.format(f[k])} | {fmt.format(a[k])} |"  # noqa: E731
    l = [f"## Costo (promedio por pregunta, n = {c['n']} preguntas del dataset)", "",
         "| Medida | Pipeline fijo (Avance 2) | Agente (Avance 3) |", "|---|---|---|",
         fila("Llamadas al LLM", "llamadas_llm", "{:.2f}"), fila("Tokens de entrada", "tokens_entrada", "{:,.0f}"),
         fila("Tokens de salida", "tokens_salida", "{:,.0f}"), fila("Tokens totales", "tokens_total", "{:,.0f}"),
         fila("Tiempo de respuesta, media (s)", "segundos_media", "{:.1f}"),
         fila("Tiempo de respuesta, mediana (s)", "segundos_mediana", "{:.1f}"), ""]
    for etiqueta, k in (("con tools", "agente_con_tools"), ("sin tools", "agente_sin_tools")):
        x = c[k]
        if x["n"]:
            l.append(f"- Agente en las preguntas **{etiqueta}** (n = {x['n']}): {x['llamadas_llm']:.2f} llamadas, "
                     f"{x['tokens_total']:,.0f} tokens, {x['segundos_media']:.1f} s de media.")
    return "\n".join(l)


# ---------------------------------------------------------------------------
# Casos de comportamiento
# ---------------------------------------------------------------------------
def verificar(checks: dict, r: RespuestaAgente) -> list[str]:
    """Devuelve la lista de comprobaciones que FALLARON (vacia = cumple)."""
    fallos, usadas, texto = [], [c.nombre for c in r.llamadas_tools], r.texto
    if r.estado != "ok":
        fallos.append(f"estado={r.estado}")
    if "tools_exactas" in checks and sorted(usadas) != sorted(checks["tools_exactas"]):
        fallos.append(f"tools usadas {usadas} != esperadas {checks['tools_exactas']}")
    fallos += [f"no llamo a {t}" for t in checks.get("debe_llamar", []) if t not in usadas]
    fallos += [f"llamo a {t} y no debia" for t in checks.get("no_debe_llamar", []) if t in usadas]
    for t, n in checks.get("min_llamadas", {}).items():
        if usadas.count(t) < n:
            fallos.append(f"{t}: {usadas.count(t)} llamadas (minimo {n})")
    for a in checks.get("args", []):
        if not any(c.nombre == a["tool"] and re.search(a["regex"], str(c.argumentos.get(a["campo"], "")), re.I)
                   for c in r.llamadas_tools):
            fallos.append(f"argumento {a['tool']}.{a['campo']} no coincide con /{a['regex']}/")
    fallos += [f"la respuesta no contiene /{p}/" for p in checks.get("respuesta_contiene", []) if not re.search(p, texto, re.I | re.S)]
    alguno = checks.get("respuesta_contiene_alguno")
    if alguno and not any(re.search(p, texto, re.I | re.S) for p in alguno):
        fallos.append("la respuesta no contiene ninguna de las expresiones esperadas")
    fallos += [f"la respuesta contiene /{p}/" for p in checks.get("respuesta_no_contiene", []) if re.search(p, texto, re.S)]
    if "empieza_con" in checks and checks["empieza_con"] not in run_eval._norm(texto)[:160]:
        fallos.append("no empieza con la frase de abstencion")
    if "max_llamadas_llm" in checks and r.n_llamadas_llm > checks["max_llamadas_llm"]:
        fallos.append(f"{r.n_llamadas_llm} llamadas al LLM (maximo {checks['max_llamadas_llm']})")
    if checks.get("solo_dominios_web") and any(not tools_web._permitido(x["url"]) for x in r.resultados_web):
        fallos.append("hay resultados web fuera de los sitios permitidos")
    if checks.get("sin_volcado_corpus"):     # 200 caracteres seguidos copiados de la base = volcado
        t, corpus = tools_web._norm(texto), tools_web._corpus_normalizado()
        if len(t) >= 200 and any(t[i:i + 200] in corpus for i in range(0, len(t) - 199, 20)):
            fallos.append("la respuesta reproduce texto de la base en bloque")
    return fallos


def correr_casos(ruta: Path, llm, limite: int | None) -> list[dict]:
    casos = json.loads(ruta.read_text(encoding="utf-8"))["casos"][:limite]
    salida = []
    for c in casos:
        falta = c.get("requiere_clave")
        if falta and not os.getenv(falta):
            print(f"[caso {c['id']}] PENDIENTE: falta {falta}", flush=True)
            salida.append({**{k: c[k] for k in ("id", "tipo", "pregunta", "esperado")}, "obtenido": f"PENDIENTE: falta {falta}",
                           "tools": [], "n_llamadas_llm": None, "cumple": None, "fallos": []})
            continue
        r = responder_agente(c["pregunta"], [Turno(t["role"], t["content"]) for t in c.get("historial", [])], llm=llm)
        fallos = verificar(c["checks"], r)
        print(f"[caso {c['id']}] {'CUMPLE' if not fallos else 'NO CUMPLE'} | {[x.nombre for x in r.llamadas_tools]} | "
              f"{r.n_llamadas_llm} llm | {fallos}", flush=True)
        salida.append({**{k: c[k] for k in ("id", "tipo", "pregunta", "esperado")},
                       "obtenido": " ".join(r.texto.split())[:350], "tools": [x.nombre for x in r.llamadas_tools],
                       "argumentos": [x.argumentos for x in r.llamadas_tools], "n_llamadas_llm": r.n_llamadas_llm,
                       "cumple": not fallos, "fallos": fallos, "respuesta": r.texto})
    return salida


def md_casos(casos: list[dict]) -> str:
    l = ["## Casos de comportamiento", "", "| # | Tipo | Esperado | Obtenido (inicio) | Tools usadas | Llam. LLM | ¿Cumple? |",
         "|---|---|---|---|---|---|---|"]
    for c in casos:
        cumple = "PENDIENTE" if c["cumple"] is None else ("Sí" if c["cumple"] else "No: " + "; ".join(c["fallos"]))
        esc = lambda s: str(s).replace("|", "\\|")  # noqa: E731
        l.append(f"| {c['id']} | {c['tipo']} | {esc(c['esperado'])} | {esc(c['obtenido'][:160])} | "
                 f"{', '.join(c['tools']) or '(ninguna)'} | {c['n_llamadas_llm'] if c['n_llamadas_llm'] is not None else '-'} | {esc(cumple)} |")
    ok = sum(1 for c in casos if c["cumple"]); pend = sum(1 for c in casos if c["cumple"] is None)
    l += ["", f"**{ok} de {len(casos) - pend} casos ejecutados cumplen la comprobación automática**"
          + (f" ({pend} pendientes)." if pend else "."), "La valoración de la redacción la hace el estudiante."]
    return "\n".join(l)


# ---------------------------------------------------------------------------
def guardar(nombre: str, meta: dict, filas: list[dict], md: str, offline: bool) -> Path:
    RES.mkdir(parents=True, exist_ok=True)
    stem = f"agente_{nombre}" + (".OFFLINE" if offline else "")
    ruta = RES / f"{stem}.json"
    ruta.write_text(json.dumps({**meta, "filas": filas}, ensure_ascii=False, indent=2), encoding="utf-8")
    (RES / f"{stem}.md").write_text(md, encoding="utf-8")
    return ruta


def ejecutar(args, llm=None) -> int:
    quiere_ragas = args.ragas or args.todo
    quiere_casos = args.casos or args.todo
    quiere_costo = args.costo or args.todo
    if not (quiere_ragas or quiere_casos or quiere_costo):
        print("ERROR: indica --todo, --ragas, --casos o --costo.")
        return 1
    try:
        filas = run_eval.cargar_dataset(Path(args.dataset), args.limite)
    except (FileNotFoundError, ValueError) as e:
        print(f"ERROR: {e}")
        return 1
    live = bool(args.live)
    llm = llm or crear_llm("tutor", live=live, usar_cache=False)     # sin cache: tiempos reales
    modelo_juez = config.LLM_JUDGE_MODEL if args.juez_final else config.LLM_MODEL
    print(f"Evaluacion del agente '{args.nombre}': {len(filas)} preguntas | modo={'LIVE' if live else 'OFFLINE'} "
          f"| tutor={config.LLM_MODEL} | juez={modelo_juez if live and quiere_ragas and not args.sin_ragas else '-'} "
          f"| partes: ragas={quiere_ragas} casos={quiere_casos} costo={quiere_costo}", flush=True)

    meta = {"nombre": f"agente_{args.nombre}", "fecha": datetime.now().isoformat(timespec="seconds"),
            "modo": "live" if live else "offline", "arquitectura": "agente", "modelo_tutor": config.LLM_MODEL,
            "proveedor": config.LLM_PROVIDER, "juez_final": bool(args.juez_final),
            "parametros": {"chunk_tamano": config.CHUNK_TAMANO, "chunk_solape": config.CHUNK_SOLAPE,
                           "top_k": config.K_MAX, "umbral": config.UMBRAL_DISTANCIA, "embeddings": config.MODELO_EMBEDDINGS,
                           "texto_embedding": config.EMBEDDING_TEXTO_MODO, "max_tokens_respuesta": config.MAX_TOKENS_RESPUESTA,
                           "max_chars_fragmento": config.MAX_CHARS, "max_iter": config.MAX_ITER},
            "dataset": {"ruta": str(args.dataset), "n": len(filas),
                        "sha256": hashlib.sha256(Path(args.dataset).read_bytes()).hexdigest()[:16]}}
    partes_md = [f"# Evaluación del agente `{args.nombre}`", "",
                 f"Tutor: `{config.LLM_MODEL}` · {len(filas)} preguntas · modo {'LIVE' if live else 'OFFLINE (no válido)'} · {meta['fecha']}", ""]
    filas_agente: list[dict] = []
    try:
        if quiere_ragas or quiere_costo:
            filas_agente = correr_agente(filas, llm)
            meta["abstencion"] = run_eval.metricas_abstencion(filas_agente)
        if quiere_costo:
            if not live:
                print("\nOFFLINE: se omite la medicion de costo (necesita --live).")
            else:
                fijo = correr_pipeline_fijo(filas, llm)
                meta["costo"] = resumen_costo(fijo, filas_agente)
                partes_md += [md_costo(meta["costo"]), ""]
        if quiere_casos:
            casos = correr_casos(Path(args.casos_ruta), llm, args.limite_casos)
            meta["casos"] = casos
            partes_md += [md_casos(casos), ""]
        if quiere_ragas:
            if not live or args.sin_ragas:
                print("\nSe omite Ragas (necesita --live y no usar --sin-ragas).")
            else:
                _, ragas = run_eval.evaluar_con_ragas(filas_agente, modelo_juez)
                meta["modelo_juez"], meta["ragas"] = modelo_juez, ragas
    except LimiteDeUsoError:
        print(f"\nERROR: {MENSAJE_LIMITE}.\nLa corrida se detuvo; lo ya hecho se guarda como parcial.")
        _guardar_parcial(args.nombre, meta, filas_agente)
        return 3
    except Exception as e:  # noqa: BLE001 - se explica y se detiene, sin reintentos propios
        print(f"\nERROR: {type(e).__name__}: {str(e)[:400]}\nLa corrida se detuvo. Nada se reintento por su cuenta.")
        _guardar_parcial(args.nombre, meta, filas_agente)
        return 2

    ruta = guardar(args.nombre, meta, filas_agente, "\n".join(partes_md), offline=not live)
    if "ragas" in meta and (RES / f"{args.base}.json").exists():
        # compare.comparar lee el JSON recien guardado, por eso se llama despues de guardar
        tabla = compare.comparar(args.base, f"agente_{args.nombre}")
        (RES / f"agente_{args.nombre}.md").write_text("\n".join(partes_md) + "\n\n" + tabla, encoding="utf-8")
        print("\n" + tabla)
    elif "ragas" in meta:
        print(f"\nAVISO: no existe la linea base '{args.base}' en {RES}; no se genero la tabla Avance 2 vs agente.")
    print(f"\nGuardado: {ruta}")
    if "abstencion" in meta:
        print("Abstencion:", json.dumps(meta["abstencion"], ensure_ascii=False))
    if "costo" in meta:
        print("\n" + md_costo(meta["costo"]))
    if live and filas_agente:
        print(f"\nTokens del agente en esta corrida: {sum(f['tokens_entrada'] for f in filas_agente):,} entrada / "
              f"{sum(f['tokens_salida'] for f in filas_agente):,} salida (sin contar casos, pipeline fijo ni juez).")
    return 0


def _guardar_parcial(nombre: str, meta: dict, filas: list[dict]) -> None:
    if filas:
        RES.mkdir(parents=True, exist_ok=True)
        (RES / f"agente_{nombre}.PARCIAL.json").write_text(json.dumps({**meta, "filas": filas}, ensure_ascii=False, indent=2), encoding="utf-8")


def main(argv=None) -> int:
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nombre", default="final", help="nombre de la corrida (archivos agente_<nombre>.*)")
    ap.add_argument("--todo", action="store_true", help="ragas + casos + costo")
    ap.add_argument("--ragas", action="store_true")
    ap.add_argument("--casos", action="store_true")
    ap.add_argument("--costo", action="store_true")
    ap.add_argument("--live", action="store_true", help="llamadas REALES a la API")
    ap.add_argument("--juez-final", action="store_true", help="juez = LLM_JUDGE_MODEL (mas caro; la linea base debe usar el mismo)")
    ap.add_argument("--sin-ragas", action="store_true")
    ap.add_argument("--limite", type=int, default=None, help="solo las primeras N preguntas del dataset")
    ap.add_argument("--limite-casos", type=int, default=None, help="solo los primeros N casos de comportamiento")
    ap.add_argument("--dataset", default=str(run_eval.DATASET_DEFECTO))
    ap.add_argument("--casos-ruta", default=str(CASOS_DEFECTO))
    ap.add_argument("--base", default=BASE_DEFECTO, help="corrida del Avance 2 con la que se compara (mismo juez)")
    return ejecutar(ap.parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
