"""Genera docs/diagrama_arquitectura.png (flujo pregunta -> LLM -> tool -> resultado -> respuesta).

    python docs/diagrama_arquitectura.py

Verde = ocurre en el sistema (nada sale). Naranja = servicio externo (indica QUE sale hacia alli).
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

VERDE, VERDE_B = "#e3f2e1", "#3d8b37"
NARANJA, NARANJA_B = "#ffe9d2", "#d9741a"
AZUL, AZUL_B = "#e4edf9", "#2d5fa8"
GRIS = "#444444"

fig, ax = plt.subplots(figsize=(16, 10.4))
ax.set_xlim(0, 16)
ax.set_ylim(-2.2, 9.2)
ax.axis("off")


def caja(x, y, w, h, titulo, lineas=(), fondo=AZUL, borde=AZUL_B, tam=11):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.12", fc=fondo, ec=borde, lw=2))
    ax.text(x + w / 2, y + h - 0.28, titulo, ha="center", va="center", fontsize=tam, fontweight="bold", color="#111")
    for i, l in enumerate(lineas):
        ax.text(x + w / 2, y + h - 0.62 - 0.27 * i, l, ha="center", va="center", fontsize=8.6, color=GRIS)


def flecha(p, q, texto="", color=GRIS, dy=0.0, dx=0.0, ls="-", ha="center", cur=0.0):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=16, lw=1.9, color=color, linestyle=ls,
                                 connectionstyle=f"arc3,rad={cur}"))
    if texto:
        ax.text((p[0] + q[0]) / 2 + dx, (p[1] + q[1]) / 2 + dy, texto, ha=ha, va="center", fontsize=8.4, color=color,
                bbox=dict(fc="white", ec="none", pad=1.2, alpha=0.9))


ax.text(8, 8.95, "Arquitectura del agente: pregunta → LLM → tool → resultado → respuesta", ha="center", fontsize=15, fontweight="bold")

# --- columna izquierda: estudiante y app
caja(0.3, 4.0, 2.3, 1.3, "Estudiante", ["escribe una pregunta", "(y ve respuesta,", "herramientas y fuentes)"])
caja(3.2, 3.85, 2.7, 1.6, "App Streamlit", ["app.py (URL pública)", "historial de conversación", "límites: 15 preguntas,", "1 500 caracteres"],
     fondo=VERDE, borde=VERDE_B)
# --- agente
caja(6.7, 3.55, 3.2, 2.2, "Agente  (rag/agent.py)", ["bucle manual con el SDK", "decide el LLM, ejecuta el programa", "MAX_ITER = 5 llamadas al LLM", "registro: logs/uso_tools.jsonl"],
     fondo=VERDE, borde=VERDE_B)
# --- LLM externo
caja(11.1, 6.45, 4.5, 2.0, "LLM · Anthropic (externo)", ["Claude Haiku 5.5, tool calling", "SALE hacia Anthropic:", "system prompt + pregunta + historial", "+ resultados de las tools (fragmentos ≤ 4×1 000 car.,", "extractos web). NUNCA el corpus completo."],
     fondo=NARANJA, borde=NARANJA_B, tam=11.5)

# --- tools
caja(6.3, 0.35, 3.0, 2.15, "buscar_en_base", ["K_MAX = 4 fragmentos", "MAX_CHARS = 1 000 c/u", "documento + sección + origen 'base'", "Chroma + embeddings E5", "100 % local"],
     fondo=VERDE, borde=VERDE_B)
caja(9.55, 0.35, 2.9, 2.15, "calcular_integral\nverificar_antiderivada", ["", "sympy en proceso aparte", "parseo con lista blanca", "tiempo máx. 10 s", "100 % local"],
     fondo=VERDE, borde=VERDE_B, tam=10)
caja(12.7, 0.35, 3.0, 2.15, "buscar_en_web  (Tavily)", ["primero la base (en código)", "sitios fijos, máx. 3 resultados", "SALE solo: consulta ≤ 200 car.", "sin texto de la base"],
     fondo=NARANJA, borde=NARANJA_B)
caja(12.7, 2.95, 3.0, 1.3, "Sitios permitidos", ["tutorial.math.lamar.edu · ocw.mit.edu", "openstax.org · mathworld.wolfram.com"], fondo="#fff6ea", borde=NARANJA_B, tam=10)

# --- datos locales
caja(0.3, 0.35, 5.4, 2.15, "Datos que NO salen del sistema", ["corpus/ (6 apuntes LaTeX) · índice Chroma (chroma_db/)", "embeddings locales (E5) · log de uso de tools", "claves: solo en .env / secrets", "(ANTHROPIC_API_KEY, TAVILY_API_KEY)"],
     fondo=VERDE, borde=VERDE_B)

# --- flechas (numeradas = orden de una pregunta que usa una tool)
def doble(p, q, color=GRIS, cur=0.0):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="<|-|>", mutation_scale=16, lw=1.9, color=color,
                                 connectionstyle=f"arc3,rad={cur}"))


def num(x, y, t, color=GRIS):
    ax.text(x, y, t, ha="center", va="center", fontsize=11, fontweight="bold", color=color,
            bbox=dict(fc="white", ec=color, pad=1.5, boxstyle="circle,pad=0.25"))


flecha((2.6, 4.45), (3.2, 4.45))                                   # estudiante -> app
num(2.9, 4.75, "1")
flecha((5.9, 4.45), (6.7, 4.45))                                   # app -> agente
num(6.3, 4.75, "2")
flecha((6.7, 5.15), (5.9, 5.15), ls="--", color=VERDE_B)           # respuesta final: agente -> app
flecha((3.2, 5.15), (2.6, 5.15), ls="--", color=VERDE_B)           # app -> estudiante
num(6.3, 5.45, "8", VERDE_B)
flecha((9.3, 5.55), (11.4, 6.45), "", NARANJA_B)                    # agente -> LLM
num(10.15, 6.25, "3", NARANJA_B)
flecha((13.2, 6.45), (9.9, 5.15), "", NARANJA_B, cur=-0.25)         # LLM -> agente
num(12.4, 5.55, "4", NARANJA_B)
ax.text(12.9, 5.0, "texto final\no tool_use(nombre, args)", ha="center", fontsize=8.4, color=NARANJA_B)
doble((7.6, 3.55), (7.8, 2.5))                                      # agente <-> buscar_en_base
doble((8.5, 3.55), (10.6, 2.5))                                     # agente <-> calculo
doble((9.9, 3.9), (12.7, 1.9), NARANJA_B, cur=0.15)                 # agente <-> web
num(8.35, 3.05, "5-6")
flecha((14.2, 2.5), (14.2, 2.95), "", NARANJA_B)
flecha((6.7, 3.95), (5.2, 2.5), cur=0.1)
num(5.75, 3.35, "7")

# --- leyenda
ax.add_patch(FancyBboxPatch((0.3, 7.55), 0.35, 0.28, boxstyle="round,pad=0.02", fc=VERDE, ec=VERDE_B, lw=2))
ax.text(0.8, 7.69, "ocurre dentro del sistema", va="center", fontsize=9.5)
ax.add_patch(FancyBboxPatch((0.3, 7.1), 0.35, 0.28, boxstyle="round,pad=0.02", fc=NARANJA, ec=NARANJA_B, lw=2))
ax.text(0.8, 7.24, "servicio externo: se indica qué sale", va="center", fontsize=9.5)
ax.text(0.3, -1.15, "1  El estudiante escribe la pregunta.\n"
        "2  La app envía pregunta + historial al agente.\n"
        "3  El agente consulta al LLM (sale a Anthropic).\n"
        "4  El LLM responde con texto final o pide una tool.",
        fontsize=9.2, color=GRIS, va="center", linespacing=1.6)
ax.text(8.3, -1.15, "5-6  El programa ejecuta la tool y devuelve el resultado al LLM\n"
        "         (se repite, máx. MAX_ITER = 5 llamadas por pregunta).\n"
        "7  Se registra el uso (pregunta, tools, tokens).\n"
        "8  Respuesta final con 'Fuente: documento, sec.' y [web: URL].\n"
        "Un saludo: el LLM responde en el paso 4, sin tool.",
        fontsize=9.2, color=GRIS, va="center", linespacing=1.6)

out = Path(__file__).with_name("diagrama_arquitectura.png")
fig.savefig(out, dpi=170, bbox_inches="tight", facecolor="white")
print("Guardado:", out)
