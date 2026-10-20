"""Juez de Ragas con Anthropic (wrapper de LangChain).

Por que esta via y no `ragas.llms.llm_factory(provider="anthropic")`:
  - llm_factory usa `instructor`, que por defecto FUERZA `tool_choice` ->
    {"type": "tool"}, y Claude Sonnet 5.5 lo rechaza con 400.
  - llm_factory ademas envia temperatura 0.01 y top_p 0.1, que los modelos 5.5
    rechazan con 400 (solo admiten el valor por defecto).
Las metricas clasicas de Ragas piden el JSON como texto y lo parsean, sin
herramientas forzadas, asi que funcionan con ChatAnthropic.

AVISO: en Ragas 0.4.3 `LangchainLLMWrapper` esta marcado como deprecado (se
quitara en una version futura). Por eso ragas esta FIJADO en 0.4.3.

Reglas de gasto que se mantienen aqui (las mismas que rag/llm.py):
  - Sin temperature / top_p / top_k: Ragas SI asigna `temperature` al objeto
    antes de cada llamada sincrona, asi que no basta con `bypass_temperature`;
    el payload se limpia siempre en `_get_request_payload`.
  - max_tokens explicito. Reintentos: maximo 3, solo 429 y 5xx.
  - Un limite de uso/credito no se reintenta y se convierte en LimiteDeUsoError.
  - Cache en disco por hash de modelo + prompt (reanudar una corrida no cuesta).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_anthropic import ChatAnthropic  # noqa: E402
from langchain_core.embeddings import Embeddings  # noqa: E402
from langchain_core.messages import AIMessage  # noqa: E402
from langchain_core.outputs import ChatGeneration, ChatResult  # noqa: E402

from rag import config  # noqa: E402
from rag.llm import LimiteDeUsoError, RespuestaLLM, _Cache, _secreto, clave_cache, es_limite_de_uso  # noqa: E402

USO = {"llamadas": 0, "desde_cache": 0, "tokens_entrada": 0, "tokens_salida": 0}


def _texto(mensaje) -> str:
    t = getattr(mensaje, "text", None)
    if isinstance(t, str) and t:     # `.text` es propiedad (str); llamarlo como metodo esta deprecado
        return str(t)
    if callable(t):
        t = t()
        if isinstance(t, str) and t:
            return t
    c = mensaje.content
    return c if isinstance(c, str) else "".join(b.get("text", "") for b in c if isinstance(b, dict))


class ChatAnthropicJuez(ChatAnthropic):
    """ChatAnthropic que nunca envia muestreo y que cachea en disco."""

    def __setattr__(self, nombre, valor):
        # Ragas ejecuta `llm.temperature = 0.01` antes de cada llamada sincrona, y
        # langchain-anthropic lanza ValueError si un modelo 5.5 recibe muestreo
        # no predeterminado. Decision del proyecto: el muestreo NO se puede fijar,
        # asi que estas asignaciones se ignoran y siempre rige el valor del modelo.
        if nombre in ("temperature", "top_p", "top_k"):
            return
        super().__setattr__(nombre, valor)

    def _get_request_payload(self, *args, **kwargs):
        payload = super()._get_request_payload(*args, **kwargs)
        for k in ("temperature", "top_p", "top_k"):
            payload.pop(k, None)           # cinturon y tirantes: nunca viaja en la peticion
        return payload

    def _clave(self, messages, stop) -> str:
        msgs = [{"role": m.type, "content": json.loads(json.dumps(m.content, default=str))} for m in messages]
        # max_tokens NO entra en la clave: las respuestas truncadas no se guardan, asi que
        # toda entrada de cache es una respuesta completa y vale con cualquier tope.
        return "juez-" + clave_cache(self.model, json.dumps(stop or []), msgs, 0)

    def _servido_de_cache(self, clave):
        previa = _Cache().leer(clave)
        if previa is None:
            return None
        USO["desde_cache"] += 1
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=previa.texto))])

    def _guardar(self, clave, resultado: ChatResult) -> None:
        msg = resultado.generations[0].message
        uso = getattr(msg, "usage_metadata", None) or {}
        USO["llamadas"] += 1
        USO["tokens_entrada"] += uso.get("input_tokens", 0)
        USO["tokens_salida"] += uso.get("output_tokens", 0)
        parada = (getattr(msg, "response_metadata", {}) or {}).get("stop_reason", "end_turn")
        _Cache().escribir(clave, RespuestaLLM(_texto(msg), self.model, uso.get("input_tokens", 0),
                                              uso.get("output_tokens", 0), parada))

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        clave = self._clave(messages, stop)
        r = self._servido_de_cache(clave)
        if r is None:
            r = super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)
            self._guardar(clave, r)
        return r

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        clave = self._clave(messages, stop)
        r = self._servido_de_cache(clave)
        if r is None:
            r = await super()._agenerate(messages, stop=stop, run_manager=run_manager, **kwargs)
            self._guardar(clave, r)
        return r


def run_config():
    """Reintentos de Ragas alineados con la regla: 3 reintentos, SOLO 429 y 5xx.
    `max_retries` de Ragas cuenta INTENTOS TOTALES, de ahi el +1. Por defecto
    Ragas reintenta cualquier excepcion 10 veces (incluido un limite de gasto)."""
    import anthropic
    from ragas.run_config import RunConfig
    return RunConfig(
        timeout=180,
        max_retries=config.REINTENTOS_API + 1,
        max_wait=30,
        max_workers=2,     # pocos workers: cuota y gasto controlados
        exception_types=(anthropic.RateLimitError, anthropic.InternalServerError),
    )


def crear_juez(modelo: str | None = None):
    """Devuelve el LLM juez para `ragas.evaluate(llm=...)`."""
    # Se importa la clase REAL: `ragas.llms.LangchainLLMWrapper` es un envoltorio de
    # deprecacion (DeprecationHelper) del que no se puede heredar.
    from ragas.llms.base import LangchainLLMWrapper

    chat = ChatAnthropicJuez(
        model=modelo or config.LLM_JUDGE_MODEL,
        max_tokens=config.MAX_TOKENS_JUEZ,
        max_retries=0,                       # los reintentos los hace Ragas con run_config()
        timeout=180,
        anthropic_api_key=_secreto("ANTHROPIC_API_KEY"),
    )

    class JuezRagas(LangchainLLMWrapper):
        limite: LimiteDeUsoError | None = None

        async def agenerate_text(self, *args, **kwargs):
            try:
                return await super().agenerate_text(*args, **kwargs)
            except Exception as e:  # noqa: BLE001
                if es_limite_de_uso(getattr(e, "status_code", None), getattr(e, "type", None),
                                    str(getattr(e, "message", e))):
                    self.limite = LimiteDeUsoError(str(e))   # Ragas lo tragaria como NaN: se anota
                raise

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)   # el aviso de deprecacion ya se documento
        return JuezRagas(chat, run_config=run_config(), bypass_temperature=True, bypass_n=True)


class EmbeddingsLocales(Embeddings):
    """Embeddings E5 locales para answer_relevancy (nada sale de la maquina)."""

    def embed_documents(self, texts):
        from rag import embeddings
        return embeddings.embeber_consultas(texts).tolist()

    def embed_query(self, text):
        from rag import embeddings
        return embeddings.embeber_consulta(text).tolist()


def crear_embeddings():
    from ragas.embeddings import LangchainEmbeddingsWrapper
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        return LangchainEmbeddingsWrapper(EmbeddingsLocales())
