"""Capa de LLM: una interfaz, varios proveedores, y las reglas de gasto.

Reglas que se cumplen AQUI y en ningun otro sitio (el resto del proyecto solo
llama a `generar`):

1. SIN TEMPERATURA. Ninguna llamada envia temperature/top_p/top_k. Los modelos
   Claude 5.5 rechazan con 400 los valores no predeterminados; se usa siempre
   el del modelo.
2. `max_tokens` EXPLICITO en cada llamada (lo exige la API de Anthropic).
3. REINTENTOS: maximo 3, con espera exponencial, SOLO ante 429 y 5xx. Un error
   de limite de gasto/credito NO se reintenta: reintentar no lo arregla y solo
   gasta tiempo. Se convierte en `LimiteDeUsoError`. Cualquier otro error
   (400, 401, 404, red...) tampoco se reintenta y se propaga tal cual.
   El SDK de Anthropic se crea con max_retries=0 para que el unico bucle de
   reintentos sea el de este modulo (por defecto el SDK reintenta tambien 408,
   409 y errores de conexion, que aqui no se quieren).
4. El system prompt va en el parametro `system`, NUNCA dentro de `mensajes`.
   `mensajes` alterna user/assistant y empieza por user.
5. CACHE en disco por hash de (modelo + system + mensajes + max_tokens): la
   misma pregunta con el mismo prompt no se paga dos veces.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

from . import config

MENSAJE_LIMITE = "El servicio alcanzó su límite de uso"

# Se puede sustituir en los tests para no esperar de verdad.
_dormir: Callable[[float], None] = time.sleep


class LimiteDeUsoError(RuntimeError):
    """Limite de gasto / credito agotado. No se reintenta. Muestra MENSAJE_LIMITE."""

    def __init__(self, detalle: str = ""):
        super().__init__(MENSAJE_LIMITE)
        self.detalle = detalle


class SinCacheError(RuntimeError):
    """Modo solo-cache: la peticion no esta en la cache y no se permite llamar a la API."""


@dataclass
class RespuestaLLM:
    texto: str
    modelo: str
    tokens_entrada: int = 0
    tokens_salida: int = 0
    detenido_por: str = ""        # end_turn | max_tokens | refusal | ...
    desde_cache: bool = False


# ---------------------------------------------------------------------------
# Clasificacion de errores y bucle de reintentos
# ---------------------------------------------------------------------------
# Los problemas de facturacion llegan como 402 `billing_error`; en algunos casos
# (credito agotado, limite del workspace) pueden llegar como 400 con un mensaje
# que lo dice. Los textos exactos de ese 400 NO estan verificados: se detectan
# por palabras clave y conviene confirmarlos con un caso real.
_RE_LIMITE = re.compile(
    r"credit balance|usage limit|spend(ing)? limit|billing|insufficient (funds|credit)|"
    r"(limit|quota).{0,40}(reached|exceeded)|(reached|exceeded).{0,40}(limit|quota)",
    re.I,
)


def es_limite_de_uso(status: int | None, tipo: str | None, mensaje: str | None) -> bool:
    return status == 402 or tipo == "billing_error" or (status == 400 and bool(_RE_LIMITE.search(mensaje or "")))


def _espera(e: Exception, intento: int) -> float:
    """Espera exponencial con jitter (1, 2, 4... s, tope 30 s); respeta `retry-after`."""
    cabeceras = getattr(getattr(e, "response", None), "headers", None) or {}
    try:
        sugerida = float(cabeceras.get("retry-after", 0))
    except (TypeError, ValueError):
        sugerida = 0.0
    return min(max(sugerida, 2 ** intento + random.random()), 30.0)


def _con_reintentos(llamada: Callable, clasificar: Callable[[Exception], str]):
    """clasificar(e) devuelve 'limite' | 'reintentar' | 'otro'."""
    for intento in range(config.REINTENTOS_API + 1):
        try:
            return llamada()
        except Exception as e:  # noqa: BLE001 - se clasifica y se re-lanza
            clase = clasificar(e)
            if clase == "limite":
                raise LimiteDeUsoError(str(e)) from e
            if clase != "reintentar" or intento == config.REINTENTOS_API:
                raise
            _dormir(_espera(e, intento))


# ---------------------------------------------------------------------------
# Proveedores
# ---------------------------------------------------------------------------
def _secreto(nombre: str) -> str | None:
    """Clave solo desde el entorno (.env) o los secrets de la plataforma."""
    valor = os.getenv(nombre)
    if valor:
        return valor
    try:   # Streamlit Community Cloud / Hugging Face guardan secrets aqui
        import streamlit as st
        return st.secrets.get(nombre)
    except Exception:  # noqa: BLE001 - sin streamlit o sin secrets
        return None


class LLMAnthropic:
    def __init__(self, modelo: str):
        self.modelo = modelo
        self._cliente = None

    def _sdk(self):
        if self._cliente is None:
            import anthropic
            self._cliente = anthropic.Anthropic(
                api_key=_secreto("ANTHROPIC_API_KEY"),
                max_retries=0,        # los reintentos los hace _con_reintentos (regla 3)
                timeout=180.0,
            )
        return self._cliente

    @staticmethod
    def _clasificar(e: Exception) -> str:
        import anthropic
        if isinstance(e, anthropic.APIStatusError):
            if es_limite_de_uso(e.status_code, getattr(e, "type", None), str(getattr(e, "message", e))):
                return "limite"
            if e.status_code == 429 or e.status_code >= 500:     # 5xx incluye 529 (sobrecarga)
                return "reintentar"
        return "otro"

    def generar(self, *, system: str | None, mensajes: list[dict], max_tokens: int) -> RespuestaLLM:
        # Sin temperature / top_p / top_k (regla 1). `system` solo si hay.
        kwargs = dict(model=self.modelo, max_tokens=max_tokens, messages=mensajes)
        if system:
            kwargs["system"] = system
        r = _con_reintentos(lambda: self._sdk().messages.create(**kwargs), self._clasificar)
        u = r.usage
        entrada = (u.input_tokens or 0) + (getattr(u, "cache_creation_input_tokens", 0) or 0) \
            + (getattr(u, "cache_read_input_tokens", 0) or 0)
        return RespuestaLLM(
            texto="".join(b.text for b in r.content if b.type == "text").strip(),
            modelo=self.modelo, tokens_entrada=entrada, tokens_salida=u.output_tokens or 0,
            detenido_por=r.stop_reason or "",
        )


class LLMGemini:
    """Proveedor ALTERNATIVO, solo conservado (LLM_PROVIDER=gemini + GEMINI_API_KEY,
    ambos comentados en .env). Misma interfaz y tampoco envia temperatura.
    Por decision del proyecto NO se amplia: no detecta limites de gasto, no tiene
    juez de Ragas y no se ha vuelto a probar en vivo tras pasar a Anthropic."""

    def __init__(self, modelo: str):
        self.modelo = modelo
        self._cliente = None

    def _sdk(self):
        if self._cliente is None:
            from google import genai
            self._cliente = genai.Client(api_key=_secreto("GEMINI_API_KEY"))
        return self._cliente

    @staticmethod
    def _clasificar(e: Exception) -> str:
        from google.genai import errors
        if isinstance(e, errors.APIError) and (e.code == 429 or (e.code or 0) >= 500):
            return "reintentar"
        return "otro"

    def generar(self, *, system: str | None, mensajes: list[dict], max_tokens: int) -> RespuestaLLM:
        from google.genai import types
        contenido = [types.Content(role="model" if m["role"] == "assistant" else "user",
                                   parts=[types.Part.from_text(text=m["content"])]) for m in mensajes]
        cfg = types.GenerateContentConfig(system_instruction=system or None, max_output_tokens=max_tokens)
        r = _con_reintentos(
            lambda: self._sdk().models.generate_content(model=self.modelo, contents=contenido, config=cfg),
            self._clasificar)
        u = r.usage_metadata
        return RespuestaLLM(
            texto=(r.text or "").strip(), modelo=self.modelo,
            tokens_entrada=getattr(u, "prompt_token_count", 0) or 0,
            tokens_salida=getattr(u, "candidates_token_count", 0) or 0,
            detenido_por=str(r.candidates[0].finish_reason) if r.candidates else "",
        )


class LLMFalso:
    """LLM de mentira para desarrollo y tests: gratis, instantaneo y determinista.
    Registra cada llamada en `.llamadas` para poder inspeccionar el prompt."""

    def __init__(self, responder: Callable[[str | None, list[dict]], str] | None = None, modelo: str = "falso"):
        self.modelo = modelo
        self.llamadas: list[dict] = []
        self._responder = responder or self._por_defecto

    @staticmethod
    def _por_defecto(system: str | None, mensajes: list[dict]) -> str:
        ultimo = mensajes[-1]["content"]
        m = re.search(r"<pregunta_seguimiento>\s*(.*?)\s*</pregunta_seguimiento>", ultimo, re.S)
        if m:                                   # condensacion: devuelve la pregunta tal cual
            return m.group(1)
        return "Respuesta simulada (LLM falso) apoyada en el material [1]."

    def generar(self, *, system: str | None, mensajes: list[dict], max_tokens: int) -> RespuestaLLM:
        self.llamadas.append({"system": system, "mensajes": mensajes, "max_tokens": max_tokens})
        texto = self._responder(system, mensajes)
        return RespuestaLLM(texto=texto, modelo=self.modelo, tokens_entrada=len(json.dumps(mensajes)) // 4,
                            tokens_salida=len(texto) // 4, detenido_por="end_turn")


# ---------------------------------------------------------------------------
# Cache en disco
# ---------------------------------------------------------------------------
def clave_cache(modelo: str, system: str | None, mensajes: list[dict], max_tokens: int) -> str:
    carga = json.dumps({"modelo": modelo, "system": system or "", "mensajes": mensajes,
                        "max_tokens": max_tokens}, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(carga.encode("utf-8")).hexdigest()


class _Cache:
    def __init__(self, directorio: Path | None = None):
        self.dir = Path(directorio or config.CACHE_LLM_DIR)

    def leer(self, clave: str) -> RespuestaLLM | None:
        ruta = self.dir / f"{clave}.json"
        if not ruta.exists():
            return None
        try:
            datos = json.loads(ruta.read_text(encoding="utf-8"))
            return RespuestaLLM(**{**datos, "desde_cache": True})
        except (OSError, ValueError, TypeError):
            return None     # entrada corrupta: se trata como no existente

    def escribir(self, clave: str, r: RespuestaLLM) -> None:
        # Una respuesta truncada o rechazada no se guarda: la proxima vez se
        # volveria a servir la mala en vez de reintentar.
        if not r.texto or r.detenido_por in ("max_tokens", "refusal"):
            return
        self.dir.mkdir(parents=True, exist_ok=True)
        tmp = self.dir / f"{clave}.tmp"
        tmp.write_text(json.dumps(asdict(r), ensure_ascii=False), encoding="utf-8")
        tmp.replace(self.dir / f"{clave}.json")     # escritura atomica


class LLMConCache:
    """Envuelve un LLM real: si la peticion ya se hizo, se sirve del disco."""

    def __init__(self, base, directorio: Path | None = None):
        self.base, self.modelo, self._cache = base, base.modelo, _Cache(directorio)

    def generar(self, *, system: str | None, mensajes: list[dict], max_tokens: int) -> RespuestaLLM:
        clave = clave_cache(self.modelo, system, mensajes, max_tokens)
        previa = self._cache.leer(clave)
        if previa is not None:
            return previa
        r = self.base.generar(system=system, mensajes=mensajes, max_tokens=max_tokens)
        self._cache.escribir(clave, r)
        return r


class LLMCacheOFalso:
    """Modo de desarrollo/pruebas (sin --live): sirve la cache REAL si existe y,
    si no, un LLM falso. Nunca llama a la API ni escribe en la cache."""

    def __init__(self, modelo: str, directorio: Path | None = None, solo_cache: bool = False):
        self.modelo, self._cache, self._falso = modelo, _Cache(directorio), LLMFalso(modelo=f"falso:{modelo}")
        self.solo_cache = solo_cache

    def generar(self, *, system: str | None, mensajes: list[dict], max_tokens: int) -> RespuestaLLM:
        previa = self._cache.leer(clave_cache(self.modelo, system, mensajes, max_tokens))
        if previa is not None:
            return previa
        if self.solo_cache:
            raise SinCacheError("Peticion no encontrada en la cache y el modo solo-cache no llama a la API.")
        return self._falso.generar(system=system, mensajes=mensajes, max_tokens=max_tokens)


# ---------------------------------------------------------------------------
def crear_llm(rol: str = "tutor", *, live: bool = True, usar_cache: bool = True,
              proveedor: str | None = None, modelo: str | None = None):
    """rol='tutor' usa LLM_MODEL; rol='juez' usa LLM_JUDGE_MODEL.
    live=False  -> cache real o LLM falso (no gasta nada).
    live=True   -> proveedor real, con cache en disco si usar_cache."""
    proveedor = (proveedor or config.LLM_PROVIDER).lower()
    modelo = modelo or (config.LLM_JUDGE_MODEL if rol == "juez" else config.LLM_MODEL)
    if proveedor == "falso":
        return LLMFalso(modelo=modelo)
    if not live:
        return LLMCacheOFalso(modelo)
    clases = {"anthropic": LLMAnthropic, "gemini": LLMGemini}
    if proveedor not in clases:
        raise ValueError(f"LLM_PROVIDER desconocido: {proveedor!r} (anthropic | gemini | falso)")
    base = clases[proveedor](modelo)
    return LLMConCache(base) if usar_cache else base
