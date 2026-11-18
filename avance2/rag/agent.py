"""Agente con tools (Avance 3): bucle MANUAL sobre el SDK de Anthropic.

Por que manual y no LangChain `create_agent`: el proyecto ya habla con el SDK a traves de
rag/llm.py (sin temperatura, max_tokens explicito, reintentos solo 429/5xx, limite de gasto,
cache). Un bucle propio reutiliza todo eso, deja los limites en codigo y cuenta con exactitud las
llamadas al LLM y los tokens, que es lo que pide el registro de uso y la comparacion de costo.

El bucle:  el LLM decide -> si pide tools, el PROGRAMA las ejecuta (tools.ejecutar) y devuelve los
resultados -> el LLM decide de nuevo. Una respuesta sin tools termina el bucle. Maximo MAX_ITER
llamadas al LLM por pregunta; en la ultima se le prohibe llamar tools para que responda con lo que tenga.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from datetime import datetime

from . import config, tools
from .generate import Turno, _historial_a_mensajes
from .llm import LimiteDeUsoError, crear_llm
from .prompts_agente import MENSAJE_MAX_ITER, SYSTEM_AGENTE


@dataclass
class LlamadaTool:
    nombre: str
    argumentos: dict
    resultado: dict          # lo que recibio el modelo (los fragmentos con su texto recortado)
    segundos: float = 0.0

    @property
    def error(self) -> bool:
        return "error" in self.resultado


@dataclass
class RespuestaAgente:
    texto: str = ""
    llamadas_tools: list[LlamadaTool] = field(default_factory=list)
    n_llamadas_llm: int = 0
    tokens_entrada: int = 0
    tokens_salida: int = 0
    segundos: float = 0.0
    estado: str = "ok"       # ok | max_iter | truncada | rechazada | limite | error
    modelo: str = ""
    avisos: list[str] = field(default_factory=list)
    citas_invalidas: list[str] = field(default_factory=list)

    @property
    def fragmentos_base(self) -> list[dict]:
        return [f for c in self.llamadas_tools if c.nombre == "buscar_en_base" for f in c.resultado.get("fragmentos", [])]

    @property
    def resultados_web(self) -> list[dict]:
        return [r for c in self.llamadas_tools if c.nombre == "buscar_en_web" for r in c.resultado.get("resultados", [])]


# ---------------------------------------------------------------------------
# Citas: lo que el modelo cita debe haber salido de las tools de ESTA pregunta
# ---------------------------------------------------------------------------
def _citas_invalidas(texto: str, llamadas: list[LlamadaTool]) -> list[str]:
    """Lineas 'Fuente: ...' y '[web: URL]' que no coinciden con ningun resultado de las tools.
    Como en el Avance 2, solo comprueba que la fuente EXISTA, no que respalde lo dicho."""
    base = {f["fuente"] for c in llamadas if c.nombre == "buscar_en_base" for f in c.resultado.get("fragmentos", [])}
    web = {r["url"] for c in llamadas if c.nombre == "buscar_en_web" for r in c.resultado.get("resultados", [])}
    limpiar = lambda s: re.sub(r"[*_`\s]+$|^[*_`\s]+", "", s).rstrip(".")  # noqa: E731
    malas = [limpiar(m) for m in re.findall(r"(Fuente:[^\n]+)", texto) if limpiar(m) not in base]
    malas += [u for u in re.findall(r"\[web:\s*([^\]\s]+)\s*\]", texto) if u not in web]
    return malas


# ---------------------------------------------------------------------------
# Registro de uso (JSON Lines)
# ---------------------------------------------------------------------------
def _resumen(c: LlamadaTool) -> dict:
    """Que se guarda de cada tool: nunca el texto de los fragmentos, solo documento y seccion."""
    r = c.resultado
    if c.error:
        return {"error": str(r["error"])[:200]}
    if c.nombre == "buscar_en_base":
        return {"fragmentos": [{"documento": f["documento"], "seccion": f["seccion"], "pagina": f["pagina"]}
                               for f in r["fragmentos"]]}
    if c.nombre == "buscar_en_web":
        return {"resultados": [{"titulo": x["titulo"][:100], "url": x["url"]} for x in r.get("resultados", [])]}
    return {"resultado": str(r.get("resultado") or r.get("resultado_exacto") or r.get("coincide"))[:200]}


# Si un estudiante pega una clave en su pregunta, el registro no la conserva.
_RE_CLAVE = re.compile(r"(?:sk-ant-|tvly-|AIza|AQ\.|hf_|ghp_)[A-Za-z0-9_\-.]{8,}")


def registrar_uso(pregunta: str, r: RespuestaAgente) -> None:
    """Una linea JSON por pregunta en logs/uso_tools.jsonl. Un fallo al escribir (disco de solo
    lectura en la nube, p. ej.) nunca debe romper la respuesta. Sin claves ni texto de fragmentos."""
    if not config.LOG_USO_ACTIVO:
        return
    linea = {
        "ts": datetime.now().isoformat(timespec="seconds"), "pregunta": _RE_CLAVE.sub("[CLAVE-OCULTA]", pregunta), "modelo": r.modelo,
        "tools": [{"nombre": c.nombre, "argumentos": c.argumentos, "segundos": round(c.segundos, 2), **_resumen(c)}
                  for c in r.llamadas_tools],
        "n_llamadas_llm": r.n_llamadas_llm, "tokens_entrada": r.tokens_entrada, "tokens_salida": r.tokens_salida,
        "segundos": round(r.segundos, 2), "estado": r.estado, "avisos": r.avisos,
    }
    try:
        config.LOGS_DIR.mkdir(parents=True, exist_ok=True)
        with (config.LOGS_DIR / "uso_tools.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(linea, ensure_ascii=False) + "\n")
    except OSError:
        pass


# ---------------------------------------------------------------------------
# El bucle
# ---------------------------------------------------------------------------
def responder_agente(pregunta: str, historial: list[Turno] | None = None, llm=None,
                     registrar: bool = True) -> RespuestaAgente:
    """Responde con el agente. Propaga LimiteDeUsoError (como `generate.responder`); cualquier
    otro error tambien se propaga, pero antes queda en el registro con estado='error'."""
    llm = llm or crear_llm("tutor", usar_cache=False)    # la app no guarda preguntas en la cache en disco
    mensajes = _historial_a_mensajes(historial or []) + [{"role": "user", "content": pregunta.strip()}]
    definiciones = tools.definiciones()
    estado_pregunta: dict = {}                            # politica por pregunta (p. ej. "primero la base")
    r = RespuestaAgente(modelo=getattr(llm, "modelo", ""))
    t0 = time.time()
    try:
        for iteracion in range(1, config.MAX_ITER + 1):
            ultima = iteracion == config.MAX_ITER
            llamada = llm.generar_con_tools(system=SYSTEM_AGENTE, mensajes=mensajes, tools=definiciones,
                                            max_tokens=config.MAX_TOKENS_RESPUESTA, sin_tools=ultima)
            r.n_llamadas_llm += 1
            r.tokens_entrada += llamada.tokens_entrada
            r.tokens_salida += llamada.tokens_salida
            pedidas = [b for b in llamada.bloques if b["type"] == "tool_use"]

            if llamada.detenido_por in ("max_tokens", "refusal"):
                r.estado = "truncada" if llamada.detenido_por == "max_tokens" else "rechazada"
                r.avisos.append(f"El modelo se detuvo por {llamada.detenido_por}.")
                r.texto = llamada.texto or MENSAJE_MAX_ITER
                break
            if not pedidas:                               # respuesta final (saludo, o ya con la info de las tools)
                r.texto = llamada.texto
                if not r.texto:
                    r.avisos.append("El modelo no devolvio texto.")
                    r.texto = "No pude generar una respuesta. Intenta reformular la pregunta."
                break
            if ultima:                                    # pidio tools aunque se le prohibio: no se ejecutan
                r.estado, r.texto = "max_iter", MENSAJE_MAX_ITER
                r.avisos.append(f"Se alcanzo MAX_ITER={config.MAX_ITER} sin respuesta final.")
                break

            mensajes.append({"role": "assistant", "content": llamada.bloques})
            resultados = []
            for b in pedidas:
                t1 = time.time()
                salida = tools.ejecutar(b["name"], b["input"], estado_pregunta)
                r.llamadas_tools.append(LlamadaTool(b["name"], b["input"], salida, time.time() - t1))
                bloque = {"type": "tool_result", "tool_use_id": b["id"], "content": json.dumps(salida, ensure_ascii=False)}
                if "error" in salida:
                    bloque["is_error"] = True
                resultados.append(bloque)
            mensajes.append({"role": "user", "content": resultados})
        r.citas_invalidas = _citas_invalidas(r.texto, r.llamadas_tools)
        if r.citas_invalidas:
            r.avisos.append(f"Citas que no coinciden con ninguna tool: {r.citas_invalidas}")
    except LimiteDeUsoError:
        r.estado = "limite"
        raise
    except Exception:
        r.estado = "error"
        raise
    finally:
        r.segundos = time.time() - t0
        if registrar:
            registrar_uso(pregunta, r)
    return r
