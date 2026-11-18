"""Version del diagrama para el documento PDF (docs/diagrama_arquitectura_doc.png): menos texto y letra mayor,
porque se inserta a ~16 cm de ancho. La version detallada es docs/diagrama_arquitectura.png.

    python docs/diagrama_arquitectura_doc.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

VERDE, VERDE_B = "#e3f2e1", "#3d8b37"
NARANJA, NARANJA_B = "#ffe9d2", "#d9741a"
AZUL, AZUL_B = "#e4edf9", "#2d5fa8"
GRIS = "#333333"

fig, ax = plt.subplots(figsize=(11, 7.0))
ax.set_xlim(0, 11)
ax.set_ylim(-1.35, 6.45)
ax.axis("off")


def caja(x, y, w, h, titulo, lineas=(), fondo=AZUL, borde=AZUL_B):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03,rounding_size=0.1", fc=fondo, ec=borde, lw=2.2))
    ax.text(x + w / 2, y + h - 0.3, titulo, ha="center", va="center", fontsize=12.5, fontweight="bold", color="#111")
    for i, l in enumerate(lineas):
        extra = 0.3 if "\n" in titulo else 0.0          # un titulo de dos lineas baja el texto
        ax.text(x + w / 2, y + h - 0.68 - extra - 0.3 * i, l, ha="center", va="center", fontsize=10.2, color=GRIS)


def flecha(p, q, color=GRIS, ls="-", doble=False, cur=0.0):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="<|-|>" if doble else "-|>", mutation_scale=18, lw=2.2, color=color,
                                 linestyle=ls, connectionstyle=f"arc3,rad={cur}"))


def num(x, y, t, color=GRIS):
    ax.text(x, y, t, ha="center", va="center", fontsize=11.5, fontweight="bold", color=color,
            bbox=dict(fc="white", ec=color, pad=1.6, boxstyle="circle,pad=0.25"))


ax.text(5.5, 6.2, "Arquitectura del agente", ha="center", fontsize=15, fontweight="bold")

caja(0.15, 3.0, 2.3, 1.75, "Estudiante", ["escribe en la", "app Streamlit"], AZUL, AZUL_B)
caja(3.35, 2.75, 3.0, 2.25, "Agente", ["bucle manual (SDK)", "el LLM decide,", "el programa ejecuta", "MAX_ITER = 5"], VERDE, VERDE_B)
caja(7.15, 3.95, 3.7, 2.05, "LLM · Anthropic", ["Claude Haiku 5.5  (externo)", "SALE: prompt + pregunta +", "historial + resultados de tools"], NARANJA, NARANJA_B)
caja(0.15, -0.05, 3.3, 2.0, "buscar_en_base", ["K_MAX = 4, MAX_CHARS = 1000", "documento + sección", "Chroma + E5: 100 % local"], VERDE, VERDE_B)
caja(3.7, -0.05, 3.5, 2.0, "calcular_integral\nverificar_antiderivada", ["sympy, proceso aparte", "límite de tiempo 10 s"], VERDE, VERDE_B)
caja(7.55, -0.05, 3.3, 2.0, "buscar_en_web", ["Tavily, sitios fijos", "máx. 3 resultados", "SALE: solo la consulta"], NARANJA, NARANJA_B)

flecha((2.45, 3.95), (3.35, 3.95))
num(2.9, 4.35, "1")
flecha((6.35, 4.75), (7.15, 5.25), NARANJA_B)
num(6.65, 5.35, "2", NARANJA_B)
flecha((7.6, 3.95), (6.35, 3.95), NARANJA_B, cur=-0.0)
num(7.0, 3.55, "3", NARANJA_B)
flecha((3.95, 2.75), (2.4, 1.95), doble=True)
flecha((4.85, 2.75), (5.45, 1.95), doble=True)
flecha((5.8, 2.75), (8.9, 1.95), NARANJA_B, doble=True)
num(4.45, 2.2, "4-5")
flecha((3.35, 3.4), (2.45, 3.4), VERDE_B, ls="--")
num(2.9, 3.1, "6", VERDE_B)

ax.text(0.15, -0.75, "1 pregunta e historial   2 el agente consulta al LLM   3 el LLM responde con texto final o pide una tool\n"
        "4 el programa ejecuta la tool   5 devuelve el resultado al LLM (se repite, máx. 5 llamadas)   6 respuesta con fuentes",
        fontsize=9.3, color=GRIS, va="center", linespacing=1.5)
ax.text(0.15, -1.2, "Verde: dentro del sistema.  Naranja: servicio externo.  Nunca sale: corpus, índice, claves ni registro de uso.",
        fontsize=9.3, color=GRIS, va="center")

out = Path(__file__).with_name("diagrama_arquitectura_doc.png")
fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
print("Guardado:", out)
