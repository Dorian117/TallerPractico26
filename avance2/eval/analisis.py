"""Analisis de una corrida (pasos 2.3 y 2.4).

    python -m eval.analisis base

Escribe eval/results/<nombre>.analisis.md, <nombre>.tabla.tex y <nombre>.png, e
imprime un resumen. Todo sale de eval/results/<nombre>.json: ninguna cifra se
calcula a mano ni se inventa.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
RES = Path(__file__).resolve().parent / "results"
METRICAS = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
NOMBRES = {"faithfulness": "Faithfulness", "answer_relevancy": "Answer relevancy",
           "context_precision": "Context precision", "context_recall": "Context recall"}


def cargar(nombre: str) -> dict:
    return json.loads((RES / f"{nombre}.json").read_text(encoding="utf-8"))


def valores(filas, m):
    return np.array([f["ragas"][m] for f in filas if f.get("ragas") and f["ragas"].get(m) is not None], float)


def ic_bootstrap(v: np.ndarray, n: int = 5000, seed: int = 42):
    """IC 95% de la media por bootstrap: con ~18 preguntas es ANCHO, y eso es lo
    que hay que mirar antes de afirmar que una metrica es mejor que otra."""
    if len(v) < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    medias = rng.choice(v, size=(n, len(v)), replace=True).mean(axis=1)
    return tuple(np.percentile(medias, [2.5, 97.5]))


def tabla(datos: dict):
    filas = [f for f in datos["filas"] if f.get("ragas")]
    out = []
    for m in METRICAS:
        v = valores(filas, m)
        lo, hi = ic_bootstrap(v)
        out.append({"metrica": m, "media": float(v.mean()) if len(v) else float("nan"),
                    "std": float(v.std(ddof=1)) if len(v) > 1 else float("nan"),
                    "min": float(v.min()) if len(v) else float("nan"), "n": int(len(v)),
                    "sin_puntaje": len(filas) - len(v), "ic_lo": float(lo), "ic_hi": float(hi)})
    return out


def md_tabla(t) -> str:
    l = ["| Métrica | Media | IC 95 % (bootstrap) | Desv. est. | Mínimo | n | sin puntaje |", "|---|---|---|---|---|---|---|"]
    for r in t:
        l.append(f"| {NOMBRES[r['metrica']]} | {r['media']:.3f} | [{r['ic_lo']:.3f}, {r['ic_hi']:.3f}] | "
                 f"{r['std']:.3f} | {r['min']:.3f} | {r['n']} | {r['sin_puntaje']} |")
    return "\n".join(l)


def tex_tabla(t, nombre: str) -> str:
    l = [r"\begin{table}[ht]", r"\centering", r"\begin{tabular}{lrrrr}", r"\toprule",
         r"Métrica & Media & IC 95\,\% & Desv.\ est. & $n$ \\", r"\midrule"]
    for r in t:
        l.append(f"{NOMBRES[r['metrica']]} & {r['media']:.3f} & [{r['ic_lo']:.3f}, {r['ic_hi']:.3f}] & "
                 f"{r['std']:.3f} & {r['n']} \\\\")
    nombre_tex = nombre.replace("_", r"\_")     # '_' fuera de modo matematico no compila en LaTeX
    l += [r"\bottomrule", r"\end{tabular}",
          rf"\caption{{Métricas de Ragas de la corrida \texttt{{{nombre_tex}}} (preguntas dentro del corpus).}}",
          rf"\label{{tab:ragas-{nombre}}}", r"\end{table}"]
    return "\n".join(l)


def grafico(t, datos, ruta: Path, titulo: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    filas = [f for f in datos["filas"] if f.get("ragas")]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    x = np.arange(len(METRICAS))
    medias = [r["media"] for r in t]
    err = [[r["media"] - r["ic_lo"] for r in t], [r["ic_hi"] - r["media"] for r in t]]
    ax.bar(x, medias, yerr=err, capsize=5, color="#4C78A8", alpha=.85, label="media e IC 95 %")
    rng = np.random.default_rng(1)
    for i, m in enumerate(METRICAS):
        v = valores(filas, m)
        ax.scatter(i + rng.uniform(-.12, .12, len(v)), v, s=14, color="#222", alpha=.55, zorder=3,
                   label="preguntas" if i == 0 else None)
    ax.set_xticks(x, [NOMBRES[m].replace(" ", "\n") for m in METRICAS])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Puntaje (0-1)")
    ax.set_title(titulo, fontsize=10)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(ruta, dpi=160)
    plt.close(fig)


def evidencia(datos, metrica: str, k: int = 4) -> str:
    filas = [f for f in datos["filas"] if f.get("ragas") and f["ragas"].get(metrica) is not None]
    peores = sorted(filas, key=lambda f: f["ragas"][metrica])[:k]
    l = []
    for f in peores:
        top = f["fuentes_recuperadas"]
        esperado = f["fuente_esperada"]
        en_top = any(x["fuente"] == esperado for x in top)
        l += [f"- **{metrica} = {f['ragas'][metrica]:.3f}** · tipo `{f['tipo']}` · «{f['question'][:110]}»",
              f"  - archivo esperado en el top-{len(top)}: **{'sí' if en_top else 'NO'}** "
              f"(recuperó: " + "; ".join(f"{x['fuente'][:22]}…{x['seccion'][-28:]} d={x['distancia']}" for x in top) + ")",
              f"  - otras métricas: " + ", ".join(f"{m}={f['ragas'][m]:.2f}" for m in METRICAS if f['ragas'].get(m) is not None),
              f"  - citas inválidas: {f['citas_invalidas']} · respuesta (inicio): «{f['respuesta'][:260].replace(chr(10), ' ')}…»"]
    return "\n".join(l)


def por_tipo(datos) -> str:
    l = ["| Tipo | n | " + " | ".join(NOMBRES[m] for m in METRICAS) + " |", "|---|---|" + "---|" * len(METRICAS)]
    for tipo in ("definicion", "procedimiento", "ejercicio", "seguimiento"):
        fs = [f for f in datos["filas"] if f["tipo"] == tipo and f.get("ragas")]
        if not fs:
            continue
        cel = []
        for m in METRICAS:
            v = valores(fs, m)
            cel.append(f"{v.mean():.2f}" if len(v) else "–")
        l.append(f"| {tipo} | {len(fs)} | " + " | ".join(cel) + " |")
    return "\n".join(l)


def calibracion(datos) -> tuple[str, dict]:
    """2.3: distancia del mejor fragmento (top-1) dentro vs fuera del corpus."""
    dentro = np.array([f["fuentes_recuperadas"][0]["distancia"] for f in datos["filas"] if f["tipo"] != "fuera_de_corpus"])
    fuera = np.array([f["fuentes_recuperadas"][0]["distancia"] for f in datos["filas"] if f["tipo"] == "fuera_de_corpus"])
    if not len(fuera) or not len(dentro):
        return "Sin datos suficientes para calibrar.", {}
    auc = float(np.mean([(d < o) + .5 * (d == o) for d in dentro for o in fuera]))
    margen = float(fuera.min() - dentro.max())
    filas = ["| umbral | dentro retenidas | fuera rechazadas | exactitud balanceada |", "|---|---|---|---|"]
    mejor = (-1, None)
    for t in sorted(set(np.round(np.concatenate([dentro, fuera]), 4))):
        ret = float((dentro <= t).mean()); rech = float((fuera > t).mean()); ba = (ret + rech) / 2
        filas.append(f"| {t:.4f} | {ret:.0%} | {rech:.0%} | {ba:.2f} |")
        if ba > mejor[0]:
            mejor = (ba, float(t))
    txt = (f"- Top-1 **dentro** del corpus (n={len(dentro)}): min {dentro.min():.4f} · mediana {np.median(dentro):.4f} · max {dentro.max():.4f}\n"
           f"- Top-1 **fuera** del corpus (n={len(fuera)}): min {fuera.min():.4f} · mediana {np.median(fuera):.4f} · max {fuera.max():.4f}\n"
           f"- AUC (prob. de que una pregunta del corpus tenga mejor top-1 que una de fuera): **{auc:.2f}**\n"
           f"- Margen (min fuera − max dentro): **{margen:+.4f}** ({'separadas' if margen > 0 else 'SOLAPADAS: ningun umbral separa sin errores'})\n"
           f"- Mejor umbral por exactitud balanceada: {mejor[1]:.4f} (exactitud {mejor[0]:.2f})\n\n" + "\n".join(filas))
    return txt, {"auc": auc, "margen": margen, "mejor_umbral": mejor[1], "ba": mejor[0]}


def main(nombre: str) -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    datos = cargar(nombre)
    t = tabla(datos)
    peor = min(t, key=lambda r: r["media"])
    cal, calinfo = calibracion(datos)
    ab = datos["abstencion"]
    meta = datos
    md = [f"# Análisis de la corrida `{nombre}`", "",
          f"Tutor: `{meta['modelo_tutor']}` · juez: `{meta.get('modelo_juez')}` (juez final: {meta['juez_final']}) · "
          f"{meta['dataset']['n']} preguntas · parámetros: `{json.dumps(meta['parametros'], ensure_ascii=False)}`", "",
          "## Métricas de Ragas (preguntas dentro del corpus)", "", md_tabla(t), "",
          f"Métrica más baja: **{NOMBRES[peor['metrica']]}** ({peor['media']:.3f}). "
          "Con ~18 preguntas y un LLM como juez los intervalos son anchos: dos métricas (o dos corridas) "
          "solo difieren con seguridad si sus IC no se solapan.", "",
          "### Por tipo de pregunta", "", por_tipo(datos), "",
          f"### Evidencia: las preguntas con menor `{peor['metrica']}`", "", evidencia(datos, peor["metrica"]), "",
          "## Fuera de corpus (métricas propias, aparte de Ragas)", "",
          f"- n = {ab['n_fuera_de_corpus']} · **tasa de abstención correcta = {ab['tasa_abstencion_correcta']}** · "
          f"**tasa de alucinación = {ab['tasa_alucinacion']}**",
          f"- Abstención indebida (dentro del corpus, respondió «no encontré»): {ab['tasa_abstencion_indebida']}",
          f"- Respuestas con citas a fragmentos inexistentes: {ab['respuestas_con_citas_invalidas']}", "",
          "## Calibración del umbral de distancia (2.3)", "", cal, ""]
    (RES / f"{nombre}.analisis.md").write_text("\n".join(md), encoding="utf-8")
    (RES / f"{nombre}.tabla.tex").write_text(tex_tabla(t, nombre), encoding="utf-8")
    grafico(t, datos, RES / f"{nombre}.png", f"Ragas · corrida {nombre} · juez {meta.get('modelo_juez')}")
    print("\n".join(md))
    print(f"\nArchivos: {nombre}.analisis.md · {nombre}.tabla.tex · {nombre}.png")


if __name__ == "__main__":
    main(sys.argv[1])
