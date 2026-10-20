"""Compara dos corridas del MISMO dataset (antes -> despues), paso 2.5.

    python -m eval.compare base iter1

Imprime y guarda eval/results/compare_<antes>_<despues>.{md,png}. Marca lo que
EMPEORO y senala si la diferencia cae dentro del ruido (IC 95 % solapados).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eval.analisis import METRICAS, NOMBRES, RES, cargar, ic_bootstrap, tabla, valores  # noqa: E402


def comparar(a: str, b: str) -> str:
    da, db = cargar(a), cargar(b)
    ta, tb = {r["metrica"]: r for r in tabla(da)}, {r["metrica"]: r for r in tabla(db)}
    advertencias = []
    if da["dataset"]["sha256"] != db["dataset"]["sha256"]:
        advertencias.append("**⚠ Los datasets difieren (distinto sha256): la comparacion no es valida.**")
    if da.get("modelo_juez") != db.get("modelo_juez"):
        advertencias.append(f"**⚠ Jueces distintos ({da.get('modelo_juez')} vs {db.get('modelo_juez')}): las metricas no son comparables.**")
    cambios = {k: (da["parametros"].get(k), db["parametros"].get(k)) for k in da["parametros"]
               if da["parametros"].get(k) != db["parametros"].get(k)}
    if da["modelo_tutor"] != db["modelo_tutor"]:
        cambios["modelo_tutor"] = (da["modelo_tutor"], db["modelo_tutor"])
    l = [f"# Comparación `{a}` → `{b}`", "", *advertencias, "",
         f"**Parámetros que cambiaron:** `{json.dumps(cambios, ensure_ascii=False)}`"
         + ("  (más de uno: no se puede atribuir el efecto)" if len(cambios) > 1 else ""), "",
         "| Métrica | Antes | Después | Δ | ¿Dentro del ruido? | Veredicto |", "|---|---|---|---|---|---|"]
    for m in METRICAS:
        x, y = ta[m], tb[m]
        d = y["media"] - x["media"]
        solapan = not (y["ic_lo"] > x["ic_hi"] or x["ic_lo"] > y["ic_hi"])
        veredicto = "sin cambio claro" if solapan else ("mejoró" if d > 0 else "EMPEORÓ")
        if d < 0 and solapan:
            veredicto = "bajó (dentro del ruido)"
        l.append(f"| {NOMBRES[m]} | {x['media']:.3f} | {y['media']:.3f} | {d:+.3f} | {'sí' if solapan else 'no'} | {veredicto} |")
    aa, ab = da["abstencion"], db["abstencion"]
    l += ["", "## Fuera de corpus", "",
          f"- Abstención correcta: {aa['tasa_abstencion_correcta']} → {ab['tasa_abstencion_correcta']}",
          f"- Alucinación: {aa['tasa_alucinacion']} → {ab['tasa_alucinacion']}",
          f"- Abstención indebida (dentro del corpus): {aa['tasa_abstencion_indebida']} → {ab['tasa_abstencion_indebida']}",
          f"- Respuestas con citas inválidas: {aa['respuestas_con_citas_invalidas']} → {ab['respuestas_con_citas_invalidas']}", ""]
    # diferencias por pregunta (pareadas): mas informativas que las medias
    fa = {f["question"]: f for f in da["filas"] if f.get("ragas")}
    fb = {f["question"]: f for f in db["filas"] if f.get("ragas")}
    comunes = [q for q in fa if q in fb]
    l += [f"## Diferencias pareadas por pregunta (n = {len(comunes)})", "",
          "| Métrica | preguntas que mejoran | empeoran | iguales | Δ medio pareado |", "|---|---|---|---|---|"]
    for m in METRICAS:
        dif = [fb[q]["ragas"][m] - fa[q]["ragas"][m] for q in comunes
               if fa[q]["ragas"].get(m) is not None and fb[q]["ragas"].get(m) is not None]
        dif = np.array(dif)
        l.append(f"| {NOMBRES[m]} | {(dif > .01).sum()} | {(dif < -.01).sum()} | {(abs(dif) <= .01).sum()} | "
                 f"{dif.mean() if len(dif) else float('nan'):+.3f} |")
    txt = "\n".join(l)
    base = RES / f"compare_{a}_{b}"
    base.with_suffix(".md").write_text(txt, encoding="utf-8")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    x = np.arange(len(METRICAS)); w = .38
    for k, (t, etiqueta, color) in enumerate([(ta, a, "#9ecae9"), (tb, b, "#4C78A8")]):
        med = [t[m]["media"] for m in METRICAS]
        err = [[t[m]["media"] - t[m]["ic_lo"] for m in METRICAS], [t[m]["ic_hi"] - t[m]["media"] for m in METRICAS]]
        ax.bar(x + (k - .5) * w, med, w, yerr=err, capsize=4, color=color, label=etiqueta)
    ax.set_xticks(x, [NOMBRES[m].replace(" ", "\n") for m in METRICAS]); ax.set_ylim(0, 1.05)
    ax.set_ylabel("Puntaje (0-1)"); ax.legend(); ax.set_title(f"Antes ({a}) vs después ({b}) · barras: IC 95 %", fontsize=10)
    fig.tight_layout(); fig.savefig(base.with_suffix(".png"), dpi=160); plt.close(fig)
    return txt


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    print(comparar(sys.argv[1], sys.argv[2]))
