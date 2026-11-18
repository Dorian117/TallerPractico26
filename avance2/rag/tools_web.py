"""Tool de busqueda web CONTROLADA (Avance 3), con Tavily por HTTP (httpx, ya instalado: sin dependencia nueva).

La base de conocimiento NUNCA sale del sistema; lo unico que sale hacia el buscador es la consulta corta
del agente. Por eso los controles viven en codigo y no dependen del modelo:
  - sitios permitidos FIJOS (SITIOS_PERMITIDOS), que el modelo no puede cambiar (no es un argumento);
  - maximo WEB_MAX_RESULTADOS resultados, cada uno con titulo, URL y extracto corto;
  - higiene de la consulta: longitud maxima y rechazo si contiene texto copiado de la base;
  - politica "primero la base": tools.ejecutar la rechaza si no se consulto antes buscar_en_base;
  - cualquier fallo del servicio devuelve un mensaje controlado, nunca una excepcion.
"""
from __future__ import annotations

import re
from functools import lru_cache
from urllib.parse import urlparse

import httpx

from . import config
from .llm import _secreto

URL_TAVILY = "https://api.tavily.com/search"

# Dominio -> por que es confiable (verificados activos con contenido de Calculo el 2026-11-18).
SITIOS_PERMITIDOS = {
    "tutorial.math.lamar.edu": "Paul's Online Math Notes (Lamar University): notas de Calculo II con las mismas tecnicas del curso.",
    "ocw.mit.edu": "MIT OpenCourseWare, curso 18.01SC de Calculo: material oficial de una universidad de referencia.",
    "openstax.org": "OpenStax (Rice University), libro de texto abierto Calculus Volume 2: capitulo de tecnicas de integracion.",
    "mathworld.wolfram.com": "MathWorld (Wolfram): definiciones y formulas matematicas con rigor de enciclopedia.",
}
_VENTANA, _PASO = 50, 5     # una consulta con >= 54 caracteres seguidos copiados de la base se bloquea


def _norm(texto: str) -> str:
    return re.sub(r"\s+", " ", texto.lower()).strip()


@lru_cache(maxsize=1)
def _corpus_normalizado() -> str:
    from . import store     # import tardio: abre el indice solo si se usa la web
    return "\n".join(_norm(d) for d in store.obtener().get(include=["documents"])["documents"])


def _copia_de_la_base(consulta: str) -> bool:
    q, corpus = _norm(consulta), _corpus_normalizado()
    return any(q[i:i + _VENTANA] in corpus for i in range(0, max(len(q) - _VENTANA + 1, 1), _PASO) if len(q) >= _VENTANA)


def _llamar_servicio(payload: dict, clave: str) -> httpx.Response:
    return httpx.post(URL_TAVILY, json=payload, headers={"Authorization": f"Bearer {clave}"},
                      timeout=config.WEB_TIMEOUT_S)


def _permitido(url: str) -> bool:
    host = (urlparse(url).hostname or "").lower()
    return any(host == d or host.endswith("." + d) for d in SITIOS_PERMITIDOS)


def buscar_en_web(consulta: str) -> dict:
    """Busca en la web, SOLO en un grupo fijo de sitios de matematicas confiables, información pública o actual.

    Politica 'primero la base': usala unicamente si buscar_en_base NO trajo nada util para la pregunta Y la
    pregunta trata de informacion publica o actual (una version reciente, un dato historico, una definicion
    general fuera del temario). Nunca para teoria que el curso deberia traer. La consulta debe ser corta y
    escrita con tus propias palabras; no pegues texto de los fragmentos de la base. Devuelve como maximo
    unos pocos resultados con titulo, URL y un extracto corto; al usarlos, cita la URL como [web: URL].
    """
    if not isinstance(consulta, str) or not consulta.strip():
        return {"error": "La consulta esta vacia."}
    consulta = re.sub(r"[\x00-\x1f]+", " ", consulta).strip()[: config.WEB_CONSULTA_MAX]
    if _copia_de_la_base(consulta):
        return {"error": "La consulta contiene texto copiado de la base de conocimiento y no se envio al buscador. "
                         "Reescribela con tus palabras, corta y sin pegar fragmentos."}
    clave = _secreto("TAVILY_API_KEY")
    if not clave:
        return {"error": "La busqueda web no esta configurada en este despliegue (falta TAVILY_API_KEY)."}
    payload = {"query": consulta, "include_domains": list(SITIOS_PERMITIDOS), "max_results": config.WEB_MAX_RESULTADOS,
               "search_depth": "basic", "include_answer": False, "include_raw_content": False}
    try:
        resp = _llamar_servicio(payload, clave)
    except httpx.HTTPError as e:
        return {"error": f"El servicio de busqueda web no respondio ({type(e).__name__})."}
    if resp.status_code != 200:
        causa = {401: "credenciales invalidas", 403: "acceso denegado", 429: "limite de uso del servicio"}.get(
            resp.status_code, f"codigo {resp.status_code}")
        return {"error": f"El servicio de busqueda web fallo: {causa}."}
    try:
        crudos = resp.json().get("results", [])
    except ValueError:
        return {"error": "El servicio de busqueda web devolvio una respuesta ilegible."}
    resultados = [{"origen": "web", "titulo": str(r.get("title", ""))[:150], "url": r["url"],
                   "extracto": " ".join(str(r.get("content", "")).split())[: config.WEB_EXTRACTO_MAX]}
                  for r in crudos if isinstance(r, dict) and isinstance(r.get("url"), str) and _permitido(r["url"])]
    resultados = resultados[: config.WEB_MAX_RESULTADOS]     # el limite se aplica aqui aunque el servicio devuelva mas
    out = {"origen": "web", "consulta": consulta, "resultados": resultados}
    if not resultados:
        out["nota"] = "Sin resultados en los sitios permitidos."
    return out
