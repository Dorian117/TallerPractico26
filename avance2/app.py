"""Tutor de Calculo II - chat web (Streamlit).

    streamlit run app.py

Avance 3: la app usa el AGENTE con tools (rag/agent.py). El pipeline fijo del Avance 2
(rag/generate.py) sigue en el repo como linea base de la evaluacion.
Las claves (ANTHROPIC_API_KEY, TAVILY_API_KEY) se leen del entorno / .env o de st.secrets; nunca estan en el repo.
"""
from __future__ import annotations

import json
import re
import sys

import streamlit as st

from rag import config
from rag.agent import responder_agente
from rag.generate import Turno
from rag.llm import MENSAJE_LIMITE, LimiteDeUsoError, _secreto, crear_llm

st.set_page_config(page_title="Tutor de Cálculo II", page_icon="∫", layout="centered")

SALUDO = (
    "Hola, soy tu tutor de Cálculo II. Respondo con el **material del curso** (y, solo si hace falta "
    "información pública, con sitios web de matemáticas confiables): antiderivación, simplificación "
    "algebraica, sustitución, integración por partes, integrales trigonométricas y sustitución "
    "trigonométrica. También puedo calcular y verificar integrales. "
    "Cuéntame qué integral estás resolviendo y hasta dónde llegaste."
)


# ---------------------------------------------------------------------------
# Recursos compartidos entre sesiones (se crean una vez por proceso)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Preparando el índice del curso (solo ocurre en el primer arranque)…")
def preparar_indice() -> str:
    from rag import embeddings, ingest
    estado = ingest.asegurar_indice()
    embeddings.embeber_consulta("calentamiento")      # carga el modelo local ahora, no en la 1.ª pregunta
    return estado


@st.cache_resource
def obtener_llm():
    # Sin cache en disco: la app no guarda las preguntas de los estudiantes.
    return crear_llm("tutor", usar_cache=False)


# ---------------------------------------------------------------------------
# Estado de la conversacion
# ---------------------------------------------------------------------------
def reiniciar() -> None:
    st.session_state.mensajes = []        # [{rol, texto, detalle, pie}]
    st.session_state.preguntas = 0
    st.session_state.tokens_entrada = 0
    st.session_state.tokens_salida = 0


if "mensajes" not in st.session_state:
    reiniciar()


def md(texto: str) -> str:
    """Markdown une los saltos de linea simples en un solo renglon: el titulo de cada paso
    ("**4. Integrar**") quedaba pegado a su texto. Un salto simple pasa a salto duro
    (dos espacios + \\n); los parrafos (linea en blanco) no cambian. Es inocuo dentro de $$...$$."""
    return re.sub(r"(?<!\n)\n(?!\n)", "  \n", texto)


def historial() -> list[Turno]:
    return [Turno(m["rol"], m["texto"]) for m in st.session_state.mensajes]


def detalle_de(r) -> dict:
    """Lo que se guarda y se muestra de cada respuesta: herramientas usadas y fuentes separadas por origen."""
    corto = lambda t: t[: config.APP_FUENTE_VISTA] + (" […]" if len(t) > config.APP_FUENTE_VISTA else "")  # noqa: E731
    return {
        "herramientas": [{"nombre": c.nombre, "argumentos": c.argumentos, "segundos": round(c.segundos, 1),
                          "error": c.resultado.get("error") if c.error else None} for c in r.llamadas_tools],
        "base": [{"documento": f["documento"], "seccion": f["seccion"], "pagina": f["pagina"],
                  "distancia": f["distancia"], "vista": corto(f["texto"])} for f in r.fragmentos_base],
        "web": [{"titulo": x["titulo"], "url": x["url"], "extracto": x["extracto"]} for x in r.resultados_web],
    }


def mostrar_detalle(d: dict) -> None:
    if d.get("herramientas"):
        with st.expander(f"Herramientas usadas ({len(d['herramientas'])})"):
            for i, h in enumerate(d["herramientas"], start=1):
                st.markdown(f"**{i}. `{h['nombre']}`** · {h['segundos']} s" + (" · ⚠ error" if h["error"] else ""))
                st.code(json.dumps(h["argumentos"], ensure_ascii=False, indent=2), language="json")
                if h["error"]:
                    st.caption(h["error"])
    elif d.get("herramientas") == []:
        st.caption("Sin herramientas: respondí directamente.")
    if d.get("base") or d.get("web"):
        with st.expander(f"Fuentes (base: {len(d['base'])} · web: {len(d['web'])})"):
            if d["base"]:
                st.markdown("**Base de conocimiento**")
                st.caption("Distancia coseno: 0 = idéntico; menor es más parecido a la consulta del agente.")
                for f in d["base"]:
                    pag = f" · pág. {f['pagina']}" if f["pagina"] else ""
                    st.markdown(f"{f['documento']} › {f['seccion']}{pag} · distancia **{f['distancia']:.4f}**")
                    st.code(f["vista"], language="latex")
            if d["web"]:
                st.markdown("**Web (sitios permitidos)**")
                for x in d["web"]:
                    titulo = re.sub(r"[\[\]]", "", x["titulo"]) or x["url"]
                    st.markdown(f"[{titulo}]({x['url']})")
                    st.caption(x["extracto"])


# ---------------------------------------------------------------------------
# Barra lateral
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Tutor de Cálculo II")
    st.caption(f"Agente con herramientas · modelo `{config.LLM_MODEL}` · máx. {config.K_MAX} fragmentos por búsqueda · "
               f"búsqueda web: {'activa' if _secreto('TAVILY_API_KEY') else 'no configurada'}")
    if st.button("Nueva conversación", use_container_width=True):
        reiniciar()
        st.rerun()
    st.metric("Preguntas en esta conversación", f"{st.session_state.preguntas} / {config.APP_MAX_PREGUNTAS}")
    st.caption(f"Tokens usados: {st.session_state.tokens_entrada:,} entrada · "
               f"{st.session_state.tokens_salida:,} salida")
    with st.expander("Privacidad"):
        st.markdown(
            "- Los apuntes se vectorizan **localmente**; el índice es propio.\n"
            "- Al proveedor del modelo solo se envían: tu pregunta, la conversación, "
            "los fragmentos que devuelven las herramientas y las instrucciones fijas del tutor.\n"
            "- Al buscador web solo sale una consulta corta del agente, y solo si la base no tiene la "
            "respuesta; nunca texto de la base.\n"
            "- Se registra cada uso de herramientas en el servidor (pregunta, herramientas, tokens; sin "
            "texto de los apuntes). El registro no se publica."
        )

# ---------------------------------------------------------------------------
# Cuerpo
# ---------------------------------------------------------------------------
st.title("Tutor de Cálculo II")

if config.LLM_PROVIDER == "anthropic" and not _secreto("ANTHROPIC_API_KEY"):
    st.error("Falta la clave de la API. Defínela como variable de entorno o secret `ANTHROPIC_API_KEY` "
             "(ver README).")
    st.stop()

try:
    preparar_indice()
except Exception as e:  # noqa: BLE001
    print(f"[app] error preparando el índice: {type(e).__name__}", file=sys.stderr)
    st.error("No se pudo preparar el índice del curso. Revisa que la carpeta `corpus/` tenga los apuntes.")
    st.stop()

with st.chat_message("assistant"):
    st.markdown(SALUDO)
for m in st.session_state.mensajes:
    with st.chat_message(m["rol"]):
        st.markdown(md(m["texto"]))
        if m["rol"] == "assistant":
            mostrar_detalle(m.get("detalle", {}))
            if m.get("pie"):
                st.caption(m["pie"])

agotado = st.session_state.preguntas >= config.APP_MAX_PREGUNTAS
if agotado:
    st.info("Llegaste al límite de preguntas de esta conversación. Pulsa **Nueva conversación** para seguir.")

pregunta = st.chat_input("Escribe tu pregunta (admite $LaTeX$)…", max_chars=config.APP_MAX_CARACTERES,
                         disabled=agotado)
if pregunta:
    pregunta = pregunta.strip()
    previo = historial()
    with st.chat_message("user"):
        st.markdown(md(pregunta))
    with st.chat_message("assistant"):
        try:
            with st.spinner("Pensando qué herramientas necesito…"):
                r = responder_agente(pregunta, previo, llm=obtener_llm())
        except LimiteDeUsoError:
            st.error(MENSAJE_LIMITE)
            st.stop()
        except Exception as e:  # noqa: BLE001 - no se muestra el detalle al usuario
            print(f"[app] error al responder: {type(e).__name__}", file=sys.stderr)
            st.error("No pude responder en este momento. Intenta de nuevo en unos segundos.")
            st.stop()
        st.markdown(md(r.texto))
        detalle = detalle_de(r)
        mostrar_detalle(detalle)
        pie = (f"{r.n_llamadas_llm} llamada(s) al modelo · {r.tokens_entrada:,} tokens de entrada · "
               f"{r.tokens_salida:,} de salida · {r.segundos:.1f} s")
        if r.citas_invalidas:
            pie += f" · fuentes citadas que no coinciden con ninguna herramienta: {r.citas_invalidas}"
        for a in r.avisos:
            pie += f" · aviso: {a}"
        st.caption(pie)
    st.session_state.mensajes += [
        {"rol": "user", "texto": pregunta},
        {"rol": "assistant", "texto": r.texto, "detalle": detalle, "pie": pie},
    ]
    st.session_state.preguntas += 1
    st.session_state.tokens_entrada += r.tokens_entrada
    st.session_state.tokens_salida += r.tokens_salida
    st.rerun()      # refresca la barra lateral (contadores) y el estado de chat_input
