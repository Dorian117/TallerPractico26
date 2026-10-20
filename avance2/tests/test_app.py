"""Tests de la app Streamlit SIN LLM real (LLM falso; gasto 0).

    python -m tests.test_app

Usan streamlit.testing.v1.AppTest sobre app.py y el indice real (embeddings
locales). Comprueban: carga, un turno con fuentes, seguimiento (condensacion),
limite de preguntas, reinicio, mensaje de limite de uso, falta de clave y que
un error interno no filtra detalles al usuario.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ["LLM_PROVIDER"] = "falso"          # antes de importar rag.config
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st  # noqa: E402
from streamlit.testing.v1 import AppTest  # noqa: E402

from rag import config, llm  # noqa: E402

APP = str(Path(__file__).resolve().parent.parent / "app.py")
Q1 = r"¿Cómo se resuelve $\int \sqrt{9-x^2}\,dx$ con sustitución trigonométrica?"


def _app(falso=None) -> AppTest:
    st.cache_resource.clear()
    if falso is not None:
        llm.crear_llm = lambda *a, **k: falso
    return AppTest.from_file(APP, default_timeout=120).run()


def _textos(at) -> str:
    return " ".join(str(m.markdown[0].value) if m.markdown else "" for m in at.chat_message)


def test_carga_sin_excepciones():
    at = _app()
    assert not at.exception, at.exception
    assert at.title[0].value == "Tutor de Cálculo II"
    assert len(at.chat_input) == 1 and not at.chat_input[0].disabled


def test_un_turno_con_fuentes():
    falso = llm.LLMFalso()
    at = _app(falso)
    at.chat_input[0].set_value(Q1).run()
    assert not at.exception, at.exception
    assert at.session_state.preguntas == 1 and len(at.session_state.mensajes) == 2
    assert len(falso.llamadas) == 1, "sin historial no se condensa: una sola llamada"
    assert falso.llamadas[0]["max_tokens"] == config.MAX_TOKENS_RESPUESTA
    fuentes = at.session_state.mensajes[1]["fuentes"]
    assert len(fuentes) == config.APP_TOP_K and all(f["distancia"] >= 0 for f in fuentes)
    assert any("Fuentes" in e.label for e in at.expander)


def test_seguimiento_condensa_y_conserva_historial():
    falso = llm.LLMFalso()
    at = _app(falso)
    at.chat_input[0].set_value(Q1).run()
    at.chat_input[0].set_value("¿Y si fuera $16-x^2$?").run()
    assert not at.exception, at.exception
    assert at.session_state.preguntas == 2 and len(at.session_state.mensajes) == 4
    assert len(falso.llamadas) == 3, "turno 1: 1 llamada; turno 2: condensar + responder"
    assert falso.llamadas[1]["system"] is None, "la condensacion no lleva el system prompt del tutor"


def test_limite_de_preguntas_deshabilita_el_chat_y_nueva_conversacion_reinicia():
    at = _app(llm.LLMFalso())
    at.chat_input[0].set_value(Q1).run()
    at.session_state.preguntas = config.APP_MAX_PREGUNTAS
    at.run()
    assert at.chat_input[0].disabled
    [b for b in at.sidebar.button if b.label == "Nueva conversación"][0].click().run()
    assert at.session_state.preguntas == 0 and at.session_state.mensajes == []
    assert not at.chat_input[0].disabled


def test_saltos_de_linea_simples_se_conservan():
    """Defecto visto en la prueba real: '**4. Integrar**\\nTexto' salia en un solo renglon."""
    falso = llm.LLMFalso(lambda s, m: "**1. Paso**\nTexto del paso [1].\n\nOtro parrafo\n$$x^2\n+1$$")
    at = _app(falso)
    at.chat_input[0].set_value(Q1).run()
    mostrado = at.chat_message[-1].markdown[0].value
    assert "**1. Paso**  \nTexto del paso [1]." in mostrado
    assert "\n\nOtro parrafo  \n$$x^2  \n+1$$" in mostrado, "los parrafos no se tocan"
    assert at.session_state.mensajes[1]["texto"].startswith("**1. Paso**\nTexto"), "el historial guarda el texto original"


def test_limite_de_uso_muestra_el_mensaje_fijo():
    class Agotado(llm.LLMFalso):
        def generar(self, **kw):
            raise llm.LimiteDeUsoError("billing")
    at = _app(Agotado())
    at.chat_input[0].set_value(Q1).run()
    assert [e.value for e in at.error] == [llm.MENSAJE_LIMITE]
    assert at.session_state.preguntas == 0, "una pregunta fallida no cuenta ni se guarda"


def test_error_interno_no_filtra_detalles():
    class Roto(llm.LLMFalso):
        def generar(self, **kw):
            raise RuntimeError("detalle-interno-secreto sk-ant-xxx")
    at = _app(Roto())
    at.chat_input[0].set_value(Q1).run()
    assert len(at.error) == 1
    assert "detalle-interno" not in at.error[0].value and "sk-ant" not in at.error[0].value


def test_sin_clave_muestra_aviso_y_no_construye_el_chat():
    original = (config.LLM_PROVIDER, llm._secreto)
    try:
        config.LLM_PROVIDER = "anthropic"
        llm._secreto = lambda nombre: None
        at = _app()
        assert any("Falta la clave" in e.value for e in at.error)
        assert len(at.chat_input) == 0
    finally:
        config.LLM_PROVIDER, llm._secreto = original


def test_la_clave_no_aparece_en_el_codigo_de_la_app():
    codigo = Path(APP).read_text(encoding="utf-8")
    assert "sk-ant" not in codigo and "AQ." not in codigo


LOCALES = [v for k, v in sorted(globals().items()) if k.startswith("test_")]

if __name__ == "__main__":
    fallos = 0
    for t in LOCALES:
        try:
            t()
            print(f"  OK    {t.__name__}")
        except Exception as e:  # noqa: BLE001
            fallos += 1
            print(f"  FALLO {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(LOCALES) - fallos}/{len(LOCALES)} tests OK  (ningun LLM real)")
    sys.exit(1 if fallos else 0)
