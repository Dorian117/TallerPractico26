"""Rellena Avance3_plantilla_de_entrega.docx con lo que ya se sabe SIN ejecutar el agente.

    python docs/generar_borrador_entrega.py "<ruta a Avance3_plantilla_de_entrega.docx>"

Todo lo que depende de ejecuciones reales (capturas, tablas de Ragas y costo del agente, casos obtenidos,
interfaz desplegada, reflexion) queda en ROJO como [PENDIENTE: ...]. Las cifras del Avance 2 salen de
eval/results/iter1_topk8.json; nada se inventa.
"""
import json
import sys
from pathlib import Path

import docx
from docx.shared import Cm, Pt, RGBColor

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))
from rag import config  # noqa: E402
from rag.prompts_agente import SYSTEM_AGENTE  # noqa: E402
from rag.tools_web import SITIOS_PERMITIDOS  # noqa: E402

ROJO = RGBColor(0xFF, 0x00, 0x00)
PLANTILLA = Path(sys.argv[1])
SALIDA = RAIZ / "docs" / "Avance3_entrega_borrador.docx"
d = docx.Document(str(PLANTILLA))
T = d.tables


# ---------------------------------------------------------------------------
def limpiar(celda):
    """La plantilla trae parrafos vacios de relleno en algunas celdas: se quitan (salvo el primero)."""
    for p in celda.paragraphs[1:]:
        if not p.text.strip() and not p._element.xpath(".//w:drawing"):
            p._element.getparent().remove(p._element)


def dividir_filas(tabla):
    """Quita 'no dividir fila entre paginas': una fila muy larga (el prompt) deja de saltar entera a la pagina siguiente."""
    for tr in tabla._tbl.tr_lst:
        for e in tr.xpath("./w:trPr/w:cantSplit"):
            e.getparent().remove(e)


def escribir(celda, partes, tam=9.5, nuevo_parrafo=False):
    """partes: str | lista de (texto, rojo?) | lista de str. Un str suelto es negro."""
    if isinstance(partes, str):
        partes = [(partes, False)]
    if not nuevo_parrafo:
        limpiar(celda)
    p = celda.add_paragraph() if nuevo_parrafo else celda.paragraphs[0]
    p.paragraph_format.space_after = Pt(2)
    p.paragraph_format.space_before = Pt(0)
    for item in partes:
        texto, rojo = item if isinstance(item, tuple) else (item, False)
        r = p.add_run(texto)
        r.font.size = Pt(tam)
        if rojo:
            r.font.color.rgb = ROJO
            r.bold = True
    return p


def lineas(celda, items, tam=9.5):
    """Varios parrafos en la misma celda (el primero reutiliza el parrafo vacio)."""
    for i, it in enumerate(items):
        escribir(celda, it, tam, nuevo_parrafo=i > 0)


def pend(texto):
    return (f"[PENDIENTE: {texto}]", True)


def fila(tabla, i, valores, tam=9.5):
    for c, v in zip(tabla.rows[i].cells, valores):
        if v is not None:
            escribir(c, v, tam)


# ---------------------------------------------------------------------------
# Portada: titulo y quitar los textos guia en gris
# ---------------------------------------------------------------------------
for p in d.paragraphs:
    if p.text.startswith("Avance 3: Plantilla"):
        for r in p.runs[1:]:
            r.text = ""
        p.runs[0].text = "Avance 3: Agente RAG con herramientas — Tutor de Cálculo II"
    elif p.text.startswith(("Completen cada", "Inserten el diagrama", "Máximo una página")):
        p._element.getparent().remove(p._element)

# ---------------------------------------------------------------------------
# Datos del equipo
# ---------------------------------------------------------------------------
escribir(T[0].rows[0].cells[1], [("Sergio Herrera  ", False), pend("confirmar si hay más integrantes")])
escribir(T[0].rows[1].cells[1], "https://github.com/Dorian117/TallerPractico26")
escribir(T[0].rows[2].cells[1], [pend("URL pública de Streamlit Cloud")])

# 1.1 Diagrama: se une la fila para usar todo el ancho
fila_diag = T[1].rows[0]
celda = fila_diag.cells[0].merge(fila_diag.cells[1])
for p in celda.paragraphs:
    for r in p.runs:
        r.text = ""
celda.paragraphs[0].add_run("Diagrama de la arquitectura").bold = True
limpiar(celda)
celda.add_paragraph().add_run().add_picture(str(RAIZ / "docs" / "diagrama_arquitectura_doc.png"), width=Cm(15.8))
escribir(celda, "Verde: ocurre dentro del sistema. Naranja: servicio externo. Hacia Anthropic salen el system prompt, la "
         "pregunta, el historial y solo lo que devuelven las tools (≤ 4 fragmentos de ≤ 1 000 caracteres por búsqueda). "
         "Hacia Tavily sale solo la consulta corta (≤ 200 caracteres). El corpus, el índice, las claves y el registro de "
         "uso no salen del sistema. Versión detallada: avance2/docs/diagrama_arquitectura.png.", 8.5, nuevo_parrafo=True)

# 1.2 Tools
tools = [
    ("buscar_en_base", "búsqueda",
     "Recupera fragmentos de los apuntes (teoría, fórmulas, métodos, ejemplos). Fuente principal; va siempre primero.",
     "consulta",
     f"K_MAX = {config.K_MAX} fragmentos; MAX_CHARS = {config.MAX_CHARS} por fragmento (recorte sin partir fórmulas); "
     f"consulta ≤ {config.CONSULTA_MAX}. Cada fragmento: origen «base», documento y sección."),
    ("calcular_integral", "tarea",
     "Calcula con sympy una integral indefinida o definida; el LLM no la calcula de memoria.",
     "expresion, variable, limite_inferior, limite_superior",
     f"Parseo seguro (sin eval; lista blanca); expresión ≤ {config.CALCULO_EXPR_MAX} caracteres; proceso aparte con "
     f"tiempo máx. {config.CALCULO_TIMEOUT_S} s."),
    ("verificar_antiderivada", "tarea",
     "Deriva una candidata y comprueba si es antiderivada del integrando (simbólico y numérico).",
     "integrando, candidata, variable", "Los mismos límites que calcular_integral."),
    ("buscar_en_web", "web",
     "Busca en 4 sitios permitidos información pública o actual; solo si la base no tiene la respuesta.",
     "consulta (los dominios no son argumento)",
     f"Máx. {config.WEB_MAX_RESULTADOS} resultados; extracto ≤ {config.WEB_EXTRACTO_MAX}; consulta ≤ {config.WEB_CONSULTA_MAX}; "
     "sitios fijos; rechazada si no se consultó antes la base."),
]
for i, t in enumerate(tools, start=1):
    fila(T[2], i, (t[0].replace("_", "_\u200b"),) + t[1:], 8)

lineas(T[3].rows[0].cells[1], [
    "Comando: python -m tests.prueba_tool_busqueda  (sin LLM, gasto 0). Salida real (extracto):",
    "buscar_en_base(consulta='regla nemotecnica ILATE para elegir u y dv') → 4 fragmentos, el primero: "
    "documento «Cálculo II: Método de Integración por Partes», sección «La Regla Nemotécnica ILATE > Algoritmo General», "
    "origen «base», distancia 0.1155, fuente «Fuente: Cálculo II: Método de Integración por Partes, sec. «La Regla Nemotécnica "
    "ILATE > Algoritmo General»».",
    [pend("pegar aquí la captura de la salida completa de ese comando")]], 8.5)

# 1.3 Bucle y registro
escribir(T[4].rows[0].cells[1], "Manual, sobre el SDK de Anthropic (rag/agent.py): el LLM decide si llama una tool, el programa la ejecuta "
         "(rag/tools.py) y devuelve el resultado, y se repite hasta que responde sin pedir tools. Se eligió el bucle manual y no "
         "create_agent de LangChain porque el proyecto ya usa el SDK vía rag/llm.py (sin temperatura, max_tokens explícito, reintentos "
         "solo ante 429/5xx, límite de gasto, caché) y así los límites viven en código y las llamadas y tokens se cuentan con exactitud.", 9)
escribir(T[4].rows[1].cells[1], f"MAX_ITER = {config.MAX_ITER} llamadas al LLM por pregunta (rag/config.py). En la última se prohíbe usar tools "
         "(tool_choice: none) para que responda con lo que tenga; si aun así pide tools, no se ejecutan y se devuelve un mensaje fijo. "
         "Un saludo se responde sin tools (1 llamada).", 9)
escribir(T[4].rows[2].cells[1], "logs/uso_tools.jsonl (JSON Lines, una línea por pregunta): pregunta, tools y argumentos, nº de llamadas al LLM, tokens de entrada "
         "y salida, tiempo, estado y avisos. No guarda claves (una clave pegada en la pregunta se oculta) ni el texto de los fragmentos "
         "(solo documento y sección). Está en .gitignore salvo un ejemplo; se apaga con LOG_USO=0.", 9)
escribir(T[4].rows[3].cells[1], [pend("pegar un extracto real de logs/uso_tools.jsonl tras la ejecución")])

# 2. Prompt
cel = T[5].rows[0].cells[1]
escribir(cel, "Archivo: rag/prompts_agente.py (SYSTEM_AGENTE). Texto completo:", 8.5)
dividir_filas(T[5])
for parrafo in SYSTEM_AGENTE.split("\n"):
    if not parrafo.strip():
        continue
    q = escribir(cel, parrafo, 6.8, nuevo_parrafo=True)
    q.paragraph_format.space_after = Pt(0)
    if parrafo.isupper() or parrafo.split(" (")[0].isupper() or parrafo.rstrip(":").isupper():
        q.paragraph_format.space_before = Pt(6)
        q.runs[0].bold = True
lineas(T[5].rows[1].cells[1], [
    "• Ya NO lleva el contexto: no hay <material_curso>; el material llega por tools.",
    "• Bloque nuevo «USO DE LAS HERRAMIENTAS»: cuándo usar cada tool y cuándo ninguna (saludos).",
    "• Citas nuevas: «Fuente: documento, sec. «…»» (base) y «[web: URL]» (web), separadas por origen (antes [n]).",
    "• Ejemplos few-shot: de 10 mensajes con fragmentos copiados a ejemplos de DECISIÓN (qué tool, con qué argumentos) dentro del prompt: ahorra tokens.",
    "• Se conserva: pedagogía de pistas, formato de 6 pasos, errores de signo, la respuesta exacta «No encontré esto en el material del curso» "
    "y la regla de ignorar instrucciones dentro del material.",
    "• Excepción nueva a la pedagogía: pedir el resultado o la verificación de una integral es una tarea de cálculo y se usa la tool.",
    "• Bloque nuevo SEGURIDAD contra «ignora tus reglas / muestra todo el documento».",
    "El pipeline del Avance 2 conserva su prompt original (rag/prompts.py) para que la comparación sea entre sistemas reales."], 8.5)

# 3. Busqueda web
escribir(T[6].rows[0].cells[1], "Tavily (API REST /search, llamada con httpx; sin dependencia nueva). Clave TAVILY_API_KEY solo por entorno o secrets. "
         "Su página de precios indica un plan gratuito de 1 000 créditos/mes sin tarjeta (confirmar límites vigentes).", 9)
escribir(T[6].rows[1].cells[1], "Solo si la base no tiene la respuesta Y la pregunta trata de información pública o actual. La regla está en el docstring, en el "
         "System Prompt y se aplica en código: buscar_en_web se rechaza si en esa pregunta no se llamó antes buscar_en_base. "
         "La consulta se limita a 200 caracteres y se rechaza si copia 54 o más caracteres de la base.", 9)
escribir(T[6].rows[2].cells[1], "Máximo 3 resultados; cada uno con título, URL y extracto (≤ 300 caracteres) y origen «web». Se filtran dominios ajenos aunque el servicio los devolviera.", 9)
escribir(T[6].rows[3].cells[1], "La tool marca cada resultado con origen «web» y el prompt exige citarlo como [web: URL], separado de las fuentes de la base "
         "(«Fuente: documento, sec.»). La interfaz las muestra en secciones distintas.", 9)
motivos = {
    "tutorial.math.lamar.edu": "Paul's Online Math Notes (Lamar University): notas de Cálculo II con las mismas técnicas del curso.",
    "ocw.mit.edu": "MIT OpenCourseWare, curso 18.01SC de Cálculo: material oficial de una universidad de referencia.",
    "openstax.org": "OpenStax (Rice University): libro abierto Calculus Volume 2, capítulo de técnicas de integración.",
    "mathworld.wolfram.com": "MathWorld (Wolfram): definiciones y fórmulas matemáticas con rigor de enciclopedia.",
}
for i, sitio in enumerate(SITIOS_PERMITIDOS, start=1):
    fila(T[7], i, (sitio, motivos[sitio]), 9)
escribir(T[7].rows[4].cells[1], "(Se verificó el 2026-11-18 que los cuatro están activos. Se descartó khanacademy.org: carga, pero no se pudo comprobar su contenido.)", 8, nuevo_parrafo=True)

# 4. Ejecucion de las tools (preguntas elegidas y comportamiento esperado; el resto depende de ejecutar)
ejecuciones = [
    (T[8], "Hola, buenas tardes. ¿Cómo estás?", "Esperado: ninguna tool y 1 llamada al LLM."),
    (T[9], "¿Qué es la regla ILATE y para qué sirve en la integración por partes?", "Esperado: buscar_en_base y NO buscar_en_web."),
    (T[10], "¿Cuánto vale la integral definida de 0 a 3 de x^2 dx?",
     "Esperado: calcular_integral(expresion=«x**2», variable=«x», limite_inferior=0, limite_superior=3); resultado 9."),
    (T[11], "¿Qué temas cubre el capítulo de técnicas de integración del libro Calculus Volume 2 de OpenStax?",
     "Esperado: buscar_en_base y luego buscar_en_web (solo sitios permitidos); fuente marcada [web: URL]. Requiere TAVILY_API_KEY."),
]
for tabla, pregunta, esperado in ejecuciones:
    escribir(tabla.rows[0].cells[1], pregunta)
    escribir(tabla.rows[1].cells[1], [(esperado + " ", False), pend("tools y argumentos reales según logs/uso_tools.jsonl")], 9)
    escribir(tabla.rows[2].cells[1], [pend("captura de la ejecución")])
    escribir(tabla.rows[3].cells[1], [pend("¿se comportó como se esperaba?")])

# 5. Ragas: la columna del Avance 2 son cifras REALES de la corrida iter1_topk8 (juez Haiku, top_k = 8)
base = json.loads((RAIZ / "eval" / "results" / "iter1_topk8.json").read_text(encoding="utf-8"))
nombres = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
for i, m in enumerate(nombres, start=1):
    escribir(T[12].rows[i].cells[1], f"{base['ragas']['promedios'][m]:.3f}", 10)
    escribir(T[12].rows[i].cells[2], [pend("corrida real")], 9)
    escribir(T[12].rows[i].cells[3], [pend("calcular")], 9)
escribir(T[13].rows[0].cells[1], [("Línea base: 18 preguntas del corpus (de las 23 del dataset aprobado; las 5 fuera de corpus se miden aparte con "
                                    "métricas de abstención). Corrida iter1_topk8, juez claude-haiku-5-5, top_k = 8. ", False),
                                   pend("confirmar el n del agente tras la corrida")], 9)
escribir(T[13].rows[1].cells[1], [pend("análisis: qué mejoró, qué empeoró y a qué componente se atribuye. Pistas: los fragmentos del agente llegan "
                                        "recortados a 1 000 caracteres (puede bajar context_recall); el agente elige su propia consulta; "
                                        "con 18 preguntas las diferencias suelen caer dentro del ruido (ver el IC en el .md de la corrida)")], 9)

# 6. Casos de comportamiento (esperado ya escrito; obtenido depende de ejecutar)
casos = json.loads((RAIZ / "eval" / "casos_comportamiento.json").read_text(encoding="utf-8"))["casos"]
import copy
while len(T[14].rows) - 1 < len(casos):          # la plantilla trae 7 filas; el caso 8 se agrega CLONANDO la ultima (mismo formato)
    T[14]._tbl.append(copy.deepcopy(T[14].rows[-1]._tr))
ANCHOS_CASOS = (0.6, 1.9, 2.6, 3.1, 2.4, 2.0, 1.9, 2.0)   # cm; suman 16.5 (el ancho de texto)
T[14].autofit = False
for col, ancho in zip(T[14].columns, ANCHOS_CASOS):          # la rejilla de la tabla manda sobre el ancho de celda
    col.width = Cm(ancho)
for fila_ in T[14].rows:
    for celda_, ancho in zip(fila_.cells, ANCHOS_CASOS):
        celda_.width = Cm(ancho)
for i, c in enumerate(casos, start=1):
    celdas = T[14].rows[i].cells
    for celda in celdas:                          # limpiar texto previo de la plantilla en las columnas 0 y 1
        for p in celda.paragraphs:
            for r in p.runs:
                r.text = ""
    escribir(celdas[0], str(c["id"]), 8)
    escribir(celdas[1], c["tipo"], 8)
    escribir(celdas[2], c["pregunta"], 7.5)
    escribir(celdas[3], c["esperado"], 7.5)
    for k in (4, 5):
        escribir(celdas[k], [pend("ejecutar")], 7)
    for k in (6, 7):
        escribir(celdas[k], [("[PENDIENTE]", True)], 6.5)

# 7. Costo
for i in range(1, 4):
    escribir(T[15].rows[i].cells[1], [pend("medir")], 9)
    escribir(T[15].rows[i].cells[2], [pend("medir")], 9)
escribir(T[16].rows[0].cells[1], [("Las 23 preguntas del dataset del Avance 2 (el pipeline fijo se vuelve a ejecutar sin caché y con el mismo contador para medir tiempos reales). ", False),
                                   pend("confirmar tras la corrida")], 9)

# 8. Interfaz
escribir(T[17].rows[0].cells[1], [pend("URL pública de Streamlit Cloud")])
for i in (1, 2, 3):
    escribir(T[17].rows[i].cells[1], [pend("captura")])

# 9. Reflexion
escribir(T[18].rows[0].cells[1], [pend("máximo una página: qué mejoró el agente frente al RAG del Avance 2 y qué cuesta (llamadas, tiempo, errores de decisión "
                                        "del modelo). Se redacta con los resultados reales de Ragas, casos y costo")], 9)

d.save(str(SALIDA))
print("Guardado:", SALIDA)
