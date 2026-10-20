"""Tutor de Calculo II - chat web (Streamlit).

    streamlit run app.py

Reutiliza el pipeline de rag/ sin cambios (condensar -> recuperar -> generar).
La clave de la API se lee del entorno / .env o de st.secrets; nunca esta en el repo.
"""
from __future__ import annotations

import re
import sys

import streamlit as st

from rag import config
from rag.generate import Turno, responder
from rag.llm import MENSAJE_LIMITE, LimiteDeUsoError, _secreto, crear_llm

st.set_page_config(page_title="Tutor de Cálculo II", page_icon="∫", layout="centered")

SALUDO = (
    "Hola, soy tu tutor de Cálculo II. Respondo **solo con el material del curso**: "
    "antiderivación, simplificación algebraica, sustitución, integración por partes, "
    "integrales trigonométricas y sustitución trigonométrica. "
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
    st.session_state.mensajes = []        # [{rol, texto, fuentes, pie}]
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


def fuentes_de(r) -> list[dict]:
    return [{
        "rango": i, "documento": f.documento, "seccion": f.seccion, "pagina": f.pagina,
        "distancia": f.distancia, "vista": f.texto[: config.APP_FUENTE_VISTA]
        + (" […]" if len(f.texto) > config.APP_FUENTE_VISTA else ""),
    } for i, f in enumerate(r.fragmentos, start=1)]


def mostrar_fuentes(fuentes: list[dict]) -> None:
    if not fuentes:
        return
    with st.expander(f"Fuentes ({len(fuentes)} fragmentos del material)"):
        st.caption("Distancia coseno: 0 = idéntico; menor es más parecido a tu pregunta. "
                   "Los números coinciden con las citas [n] de la respuesta.")
        for f in fuentes:
            pag = f" · pág. {f['pagina']}" if f["pagina"] else ""
            st.markdown(f"**[{f['rango']}]** {f['documento']} › {f['seccion']}{pag} · "
                        f"distancia **{f['distancia']:.4f}**")
            st.code(f["vista"], language="latex")


# ---------------------------------------------------------------------------
# Barra lateral
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Tutor de Cálculo II")
    st.caption(f"Modelo: `{config.LLM_MODEL}` · recuperación: {config.APP_TOP_K} fragmentos")
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
            "los fragmentos recuperados y las instrucciones fijas del tutor.\n"
            "- La aplicación no guarda tus preguntas."
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
            mostrar_fuentes(m.get("fuentes", []))
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
            with st.spinner("Buscando en el material y preparando la respuesta…"):
                r = responder(pregunta, previo, top_k=config.APP_TOP_K, llm=obtener_llm())
        except LimiteDeUsoError:
            st.error(MENSAJE_LIMITE)
            st.stop()
        except Exception as e:  # noqa: BLE001 - no se muestra el detalle al usuario
            print(f"[app] error al responder: {type(e).__name__}", file=sys.stderr)
            st.error("No pude responder en este momento. Intenta de nuevo en unos segundos.")
            st.stop()
        st.markdown(md(r.texto))
        fuentes = fuentes_de(r)
        mostrar_fuentes(fuentes)
        pie = f"{r.tokens_entrada:,} tokens de entrada · {r.tokens_salida:,} de salida · {r.segundos:.1f} s"
        if r.citas_invalidas:
            pie += f" · citas a fragmentos inexistentes: {r.citas_invalidas}"
        for a in r.avisos:
            pie += f" · aviso: {a}"
        st.caption(pie)
    st.session_state.mensajes += [
        {"rol": "user", "texto": pregunta},
        {"rol": "assistant", "texto": r.texto, "fuentes": fuentes, "pie": pie},
    ]
    st.session_state.preguntas += 1
    st.session_state.tokens_entrada += r.tokens_entrada
    st.session_state.tokens_salida += r.tokens_salida
    st.rerun()      # refresca la barra lateral (contadores) y el estado de chat_input
