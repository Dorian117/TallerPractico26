"""Tests de la evaluacion. NO usan ningun LLM real (gratis, sin red).

    python -m tests.test_eval

Lo que NO se puede probar sin gastar: que Ragas puntue con el juez real. Eso
se comprueba en la primera corrida --live con --limite pequeno.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import anthropic  # noqa: E402
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage  # noqa: E402
from langchain_core.outputs import ChatGeneration, ChatResult  # noqa: E402

from eval import juez, run_eval  # noqa: E402
from rag import config, llm  # noqa: E402


def _con_cache_temporal(fn):
    def envuelto():
        original = config.CACHE_LLM_DIR
        with tempfile.TemporaryDirectory() as t:
            config.CACHE_LLM_DIR = Path(t)
            try:
                fn(Path(t))
            finally:
                config.CACHE_LLM_DIR = original
    envuelto.__name__ = fn.__name__
    return envuelto


def _chat():
    return juez.ChatAnthropicJuez(model="claude-sonnet-5-5", max_tokens=4096, max_retries=0,
                                  anthropic_api_key="sk-ant-FALSA-no-se-usa")


# ------------------------------------ juez ----------------------------------
def test_juez_nunca_envia_muestreo_aunque_ragas_asigne_temperatura():
    chat = _chat()
    chat.temperature = 0.01          # exactamente lo que hace LangchainLLMWrapper antes de cada llamada
    chat.top_p = 0.1
    payload = chat._get_request_payload([SystemMessage("s"), HumanMessage("h")])
    assert not {"temperature", "top_p", "top_k"} & set(payload), payload.keys()
    assert payload["max_tokens"] == 4096 and payload["model"] == "claude-sonnet-5-5"


def test_run_config_reintentos_solo_429_y_5xx():
    rc = juez.run_config()
    assert rc.max_retries == config.REINTENTOS_API + 1 == 4, "Ragas cuenta intentos totales: 1 + 3 reintentos"
    assert set(rc.exception_types) == {anthropic.RateLimitError, anthropic.InternalServerError}
    assert not issubclass(anthropic.BadRequestError, rc.exception_types), "un 400 (limite de gasto) no se reintenta"
    assert rc.max_workers <= 2


def test_crear_juez_configuracion():
    j = juez.crear_juez("claude-haiku-5-5")
    assert j.bypass_temperature is True and j.langchain_llm.model == "claude-haiku-5-5"
    assert j.langchain_llm.max_retries == 0, "los reintentos los hace Ragas con run_config, no el SDK"


@_con_cache_temporal
def test_juez_cache_en_disco(_dir):
    chat = _chat()
    msgs = [HumanMessage("pregunta del juez")]
    clave = chat._clave(msgs, None)
    assert clave == chat._clave(msgs, None) and clave != chat._clave([HumanMessage("otra")], None)
    assert chat._servido_de_cache(clave) is None
    ia = AIMessage(content="veredicto", usage_metadata={"input_tokens": 5, "output_tokens": 2, "total_tokens": 7})
    chat._guardar(clave, ChatResult(generations=[ChatGeneration(message=ia)]))
    # con la cache sembrada, _agenerate responde SIN tocar la red (la clave API es falsa)
    r = asyncio.run(chat._agenerate(msgs))
    assert r.generations[0].message.content == "veredicto"


# ---------------------------------- abstencion -------------------------------
def test_deteccion_de_abstencion():
    assert run_eval.se_abstiene("No encontré esto en el material del curso. Los fragmentos...")
    assert run_eval.se_abstiene("NO ENCONTRE ESTO EN EL MATERIAL DEL CURSO")
    assert not run_eval.se_abstiene("Tu elección es correcta. Aquí la solución")
    assert not run_eval.se_abstiene("Los fragmentos dicen: no encontré nada raro " + "x" * 400)


# --------------------------- ejecucion de la corrida -------------------------
def _dataset(t: Path) -> Path:
    filas = [
        {"question": "¿Qué sustitución uso con raíz de a^2 - x^2?", "ground_truth": "x = a sen(theta)",
         "tipo": "definicion", "fuente_esperada": "susticion"},
        {"question": "¿Y si fuera con a^2 + x^2?", "ground_truth": "x = a tan(theta)", "tipo": "seguimiento",
         "fuente_esperada": "susticion",
         "historial": [{"role": "user", "content": "¿Qué sustitución uso con raíz de a^2 - x^2?"},
                       {"role": "assistant", "content": "x = a sen(theta) [1]"}]},
        {"question": "¿Quién construyó la Torre Eiffel?", "ground_truth": "No encontré esto en el material del curso",
         "tipo": "fuera_de_corpus", "fuente_esperada": ""},
    ]
    ruta = t / "dataset.jsonl"
    ruta.write_text("\n".join(json.dumps(f, ensure_ascii=False) for f in filas), encoding="utf-8")
    return ruta


def _args(t: Path, **kw):
    base = dict(nombre="prueba", dataset=str(_dataset(t)), limite=None, live=False, juez_final=False,
                sin_ragas=False, top_k=None)
    return argparse.Namespace(**{**base, **kw})


def test_corrida_offline_con_llm_falso():
    original = run_eval.RESULTADOS
    with tempfile.TemporaryDirectory() as t:
        run_eval.RESULTADOS = Path(t) / "results"
        try:
            def resp(system, mensajes):
                ult = mensajes[-1]["content"]
                if "Eiffel" in ult:
                    return "No encontré esto en el material del curso. No respondo de memoria."
                m = __import__("re").search(r"<pregunta_seguimiento>\s*(.*?)\s*</pregunta_seguimiento>", ult, __import__("re").S)
                return m.group(1) if m else "Segun el material [1] y [9]."
            codigo = run_eval.ejecutar(_args(Path(t)), llm=llm.LLMFalso(responder=resp))
            assert codigo == 0
            salida = Path(t) / "results" / "prueba.OFFLINE.json"
            assert salida.exists() and not (Path(t) / "results" / "prueba.json").exists(), "offline nunca se guarda como valido"
            datos = json.loads(salida.read_text(encoding="utf-8"))
            assert datos["modo"] == "offline" and "NO son resultados validos" in datos["advertencia"]
            ab = datos["abstencion"]
            assert ab["tasa_abstencion_correcta"] == 1.0 and ab["tasa_alucinacion"] == 0.0
            assert ab["tasa_abstencion_indebida"] == 0.0
            assert ab["respuestas_con_citas_invalidas"] == 2, "[9] no existe en las respuestas con fragmentos"
            assert (Path(t) / "results" / "prueba.OFFLINE.csv").exists()
            # --limite: solo la primera pregunta
            assert run_eval.ejecutar(_args(Path(t), nombre="lim", limite=1), llm=llm.LLMFalso(responder=resp)) == 0
            assert len(json.loads((Path(t) / "results" / "lim.OFFLINE.json").read_text(encoding="utf-8"))["filas"]) == 1
        finally:
            run_eval.RESULTADOS = original


def test_corrida_se_detiene_con_mensaje_claro():
    original = run_eval.RESULTADOS
    with tempfile.TemporaryDirectory() as t:
        run_eval.RESULTADOS = Path(t) / "results"
        try:
            class Agotado(llm.LLMFalso):
                def generar(self, **kw):
                    raise llm.LimiteDeUsoError("billing")
            assert run_eval.ejecutar(_args(Path(t)), llm=Agotado()) == 3, "limite de uso -> codigo 3"
            class Roto(llm.LLMFalso):
                def generar(self, **kw):
                    raise RuntimeError("fallo inesperado")
            assert run_eval.ejecutar(_args(Path(t)), llm=Roto()) == 2, "otro error -> se detiene, codigo 2"
            assert not (Path(t) / "results").exists(), "una corrida fallida no deja resultados a medias"
            assert run_eval.ejecutar(_args(Path(t), dataset=str(Path(t) / "no_existe.jsonl"))) == 1
        finally:
            run_eval.RESULTADOS = original


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
