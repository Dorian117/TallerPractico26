"""Generacion: condensar -> recuperar -> construir prompt -> LLM.

Lo UNICO que sale de la maquina hacia el proveedor del LLM en todo el pipeline:
  - la pregunta del estudiante (y el historial, tambien en la condensacion),
  - los top_k fragmentos recuperados,
  - el system prompt y los ejemplos few-shot (texto fijo del proyecto).
Nunca el corpus completo, los archivos ni el indice.

Este modulo no conoce al proveedor: usa `llm.crear_llm`. Los errores de
limite de uso (`LimiteDeUsoError`) se propagan; quien llama (CLI, app,
run_eval) decide como mostrarlos.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from . import config, prompts
from .llm import LimiteDeUsoError, crear_llm
from .retrieve import Fragmento, recuperar

MENSAJE_SIN_COBERTURA = (
    "No encontré esto en el material del curso. Los fragmentos disponibles no "
    "tratan esta pregunta, y prefiero no responder de memoria para no darte "
    "algo que no coincida con tus apuntes. Puedo ayudarte con antiderivación, "
    "sustitución, simplificación algebraica, integración por partes, integrales "
    "trigonométricas y sustitución trigonométrica."
)


@dataclass
class Turno:
    rol: str     # "user" | "assistant"
    texto: str


@dataclass
class Respuesta:
    texto: str
    fragmentos: list[Fragmento]
    pregunta_recuperacion: str            # la que se uso para buscar (condensada si hubo historial)
    citas_invalidas: list[int] = field(default_factory=list)
    llamo_modelo: bool = True
    modelo: str = ""
    tokens_entrada: int = 0
    tokens_salida: int = 0
    segundos: float = 0.0
    desde_cache: bool = False
    avisos: list[str] = field(default_factory=list)


def _recortar(texto: str, n: int) -> str:
    return texto if len(texto) <= n else texto[:n].rstrip() + " [...]"


# ---------------------------------------------------------------------------
# Condensacion de la pregunta de seguimiento
# ---------------------------------------------------------------------------
def condensar_pregunta(historial: list[Turno], pregunta: str, llm) -> str:
    """"¿Y si en lugar de 4x^2 fuera 9x^2?" no sirve como consulta: sin el
    historial no dice de que integral habla y la recuperacion fallaria. Se
    reescribe como pregunta autonoma ANTES de buscar. Sin historial no hay
    nada que condensar (y se ahorra una llamada)."""
    if not historial:
        return pregunta
    texto_hist = "\n".join(
        f"{'Estudiante' if t.rol == 'user' else 'Tutor'}: "
        f"{_recortar(t.texto, config.CONDENSACION_RECORTE)}"
        for t in historial[-config.CONDENSACION_TURNOS:]
    )
    r = llm.generar(
        system=None,
        mensajes=[{"role": "user", "content": prompts.PROMPT_CONDENSACION.format(
            historial=texto_hist, pregunta=pregunta.strip())}],
        max_tokens=config.MAX_TOKENS_CONDENSACION,
    )
    lineas = r.texto.strip().splitlines()
    return lineas[0].strip() if lineas else pregunta


def _historial_a_mensajes(historial: list[Turno]) -> list[dict]:
    """Los turnos REALES (no resumidos) de la conversacion. Se guarda solo el
    texto de pregunta y respuesta, sin re-enviar los fragmentos viejos."""
    turnos = historial[-config.HISTORIAL_MAX_TURNOS:]
    while turnos and turnos[0].rol != "user":   # la conversacion debe empezar por el estudiante
        turnos = turnos[1:]
    return [{"role": t.rol, "content": t.texto} for t in turnos]


# ---------------------------------------------------------------------------
# Pipeline completo
# ---------------------------------------------------------------------------
def responder(pregunta: str, historial: list[Turno] | None = None, top_k: int | None = None,
              llm=None) -> Respuesta:
    historial = historial or []
    llm = llm or crear_llm("tutor", usar_cache=False)   # la app no guarda preguntas de estudiantes en disco
    t0 = time.time()
    avisos: list[str] = []

    # 1) condensar (solo si hay historial); la RECUPERACION usa la condensada
    try:
        consulta = condensar_pregunta(historial, pregunta, llm)
    except LimiteDeUsoError:
        raise
    except Exception as e:  # noqa: BLE001
        consulta = pregunta
        avisos.append(f"Condensacion fallida ({type(e).__name__}); se busco con la pregunta original.")

    # 2) recuperar top_k fragmentos
    fragmentos = recuperar(consulta, top_k=top_k)

    # 3) con umbral calibrado y ningun fragmento dentro: no se llama al modelo.
    #    (Inactivo mientras UMBRAL_DISTANCIA sea None; se decide en la Fase 3.)
    if config.UMBRAL_DISTANCIA is not None and not fragmentos:
        return Respuesta(MENSAJE_SIN_COBERTURA, [], consulta, llamo_modelo=False,
                         segundos=time.time() - t0, avisos=avisos)

    # 4) prompt: system aparte + few-shot + historial real + turno actual.
    #    La GENERACION recibe la pregunta ORIGINAL del estudiante, no la condensada.
    mensajes = prompts.armar_mensajes(pregunta, fragmentos, _historial_a_mensajes(historial))
    r = llm.generar(system=prompts.SYSTEM_PROMPT, mensajes=mensajes, max_tokens=config.MAX_TOKENS_RESPUESTA)

    texto = r.texto
    if r.detenido_por == "max_tokens":
        avisos.append("La respuesta se corto por el limite de tokens de salida (max_tokens).")
    if r.detenido_por == "refusal":
        avisos.append("El modelo rechazo la solicitud (stop_reason=refusal).")
    if not texto:
        avisos.append(f"El modelo no devolvio texto (detenido_por={r.detenido_por or 'desconocido'}).")
        texto = "No pude generar una respuesta. Intenta reformular la pregunta."

    return Respuesta(
        texto=texto,
        fragmentos=fragmentos,
        pregunta_recuperacion=consulta,
        citas_invalidas=prompts.citas_invalidas(texto, len(fragmentos)),
        modelo=r.modelo,
        tokens_entrada=r.tokens_entrada,
        tokens_salida=r.tokens_salida,
        segundos=time.time() - t0,
        desde_cache=r.desde_cache,
        avisos=avisos,
    )
