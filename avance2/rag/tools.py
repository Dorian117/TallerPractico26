"""Tools del agente (Avance 3): buscar en la base, calcular y verificar integrales.

Cada tool es una funcion normal con DOCSTRING (que se envia al modelo como descripcion:
que hace y cuando usarla) y devuelve un dict. Se pueden probar sin LLM llamandolas
directamente (ver tests/prueba_tool_busqueda.py). Los limites (K_MAX, MAX_CHARS,
tiempos, longitudes) estan en config.py y se aplican AQUI, en codigo.
"""
from __future__ import annotations

import inspect
import multiprocessing as mp
import queue

from . import calculo, config
from .chunking import trocear_texto
from .retrieve import recuperar
from .tools_web import buscar_en_web


# ---------------------------------------------------------------------------
# 1. Tool de busqueda en la base
# ---------------------------------------------------------------------------
def _recortar(texto: str, maximo: int) -> str:
    """Recorta a `maximo` caracteres SIN partir una formula: reutiliza el chunker, que
    corta entre parrafos/lineas y trata las formulas como atomos indivisibles."""
    if len(texto) <= maximo:
        return texto
    pieza = trocear_texto(texto, maximo, 0)[0]
    return pieza[:maximo].rstrip() + " [...recortado]" if len(pieza) > maximo else pieza + " [...recortado]"


def _cita(f) -> str:
    """Texto listo para copiar en la respuesta. El corpus es LaTeX (sin paginas): se cita la seccion."""
    donde = f"pág. {f.pagina}" if f.pagina else "sec. «" + " > ".join(f.seccion.split(" > ")[-2:]) + "»"
    return f"Fuente: {f.documento}, {donde}"


def buscar_en_base(consulta: str) -> dict:
    """Busca en los apuntes del curso de Calculo II (la base de conocimiento) los fragmentos mas relevantes.

    Usala siempre que la pregunta trate de teoria, definiciones, formulas, metodos de integracion
    (antiderivacion, sustitucion, simplificacion algebraica, integracion por partes, integrales
    trigonometricas, sustitucion trigonometrica) o ejemplos del curso. Es la fuente PRINCIPAL: se
    consulta antes que cualquier otra. No la uses para saludos ni charla casual. Si la pregunta
    tiene dos temas distintos, llamala una vez por tema con una consulta corta cada vez.
    Devuelve fragmentos con documento, seccion y el texto de cita ('fuente') que debes copiar al citar.
    """
    if not isinstance(consulta, str) or not consulta.strip():
        return {"error": "La consulta esta vacia."}
    consulta = consulta.strip()[: config.CONSULTA_MAX]
    fragmentos = recuperar(consulta, top_k=config.K_MAX)[: config.K_MAX]
    return {
        "origen": "base",
        "consulta": consulta,
        "fragmentos": [{
            "origen": "base", "documento": f.documento, "seccion": f.seccion, "pagina": f.pagina,
            "fuente": _cita(f), "distancia": round(f.distancia, 4),
            "texto": _recortar(f.texto, config.MAX_CHARS),
        } for f in fragmentos],
    }


# ---------------------------------------------------------------------------
# 2. Tools de tarea del dominio (calculo exacto con sympy, en un proceso con tiempo maximo)
# ---------------------------------------------------------------------------
def _con_limite(accion: str, kwargs: dict) -> dict:
    """Ejecuta calculo.trabajo en un proceso aparte (spawn: funciona igual en Windows y Linux)
    y lo termina si pasa de CALCULO_TIMEOUT_S: sympy puede quedarse horas en una integral dificil."""
    ctx = mp.get_context("spawn")
    cola = ctx.Queue()
    proceso = ctx.Process(target=calculo.trabajo, args=(cola, accion, kwargs), daemon=True)
    proceso.start()
    try:
        return cola.get(timeout=config.CALCULO_TIMEOUT_S)
    except queue.Empty:
        return {"error": f"El calculo supero el tiempo maximo ({config.CALCULO_TIMEOUT_S} s). "
                         "Prueba con una expresion mas simple."}
    finally:
        if proceso.is_alive():
            proceso.terminate()
        proceso.join(2)


def calcular_integral(expresion: str, variable: str = "x", limite_inferior=None, limite_superior=None) -> dict:
    """Calcula de forma SIMBOLICA (sympy) una integral indefinida o definida. No la calcules de memoria.

    Usala cuando el estudiante pida el resultado o el valor de una integral concreta, o cuando
    necesites confirmar un resultado antes de darlo. Extrae del texto de la pregunta: la
    'expresion' del integrando en sintaxis simple (x**2, sqrt(x), sin(x), x*exp(x); multiplicacion
    siempre con *), la 'variable' y, solo si es definida, ambos limites (numeros o constantes como pi/2).
    Indefinida: devuelve la antiderivada (+ C). Definida: devuelve el valor exacto y el numerico.
    """
    return _con_limite("integral", {"expresion": expresion, "variable": variable,
                                    "limite_inferior": limite_inferior, "limite_superior": limite_superior})


def verificar_antiderivada(integrando: str, candidata: str, variable: str = "x") -> dict:
    """Verifica si una funcion 'candidata' es una antiderivada del 'integrando': deriva la candidata y compara.

    Usala cuando el estudiante propone un resultado para una integral y quiere saber si es correcto, o
    para comprobar tu propia respuesta. Extrae del texto el 'integrando', la 'candidata' (puedes incluir
    + C) y la 'variable', en sintaxis simple (x**2, sin(x), x*exp(x); multiplicacion siempre con *).
    Devuelve si coincide y la derivada de la candidata.
    """
    return _con_limite("verificar", {"integrando": integrando, "candidata": candidata, "variable": variable})


# ---------------------------------------------------------------------------
# 3. Registro: esquemas para la API y despacho
# ---------------------------------------------------------------------------
_STR = {"type": "string"}
_NUM_O_STR = {"anyOf": [{"type": "string"}, {"type": "number"}]}
_ESQUEMAS = {
    "buscar_en_base": ({"consulta": {**_STR, "description": "Consulta corta (palabras clave o una pregunta breve) sobre UN solo tema."}},
                       ["consulta"]),
    "calcular_integral": ({"expresion": {**_STR, "description": "Integrando en sintaxis simple, p. ej. x*exp(x)."},
                           "variable": {**_STR, "description": "Variable de integracion (por defecto x)."},
                           "limite_inferior": {**_NUM_O_STR, "description": "Limite inferior (solo si es definida), p. ej. 0."},
                           "limite_superior": {**_NUM_O_STR, "description": "Limite superior (solo si es definida), p. ej. pi/2."}},
                          ["expresion"]),
    "verificar_antiderivada": ({"integrando": {**_STR, "description": "Funcion que se integra."},
                                "candidata": {**_STR, "description": "Funcion propuesta como antiderivada (puede incluir + C)."},
                                "variable": {**_STR, "description": "Variable (por defecto x)."}},
                               ["integrando", "candidata"]),
    # sin argumento de dominios: la lista de sitios permitidos esta fija en tools_web.py
    "buscar_en_web": ({"consulta": {**_STR, "description": "Consulta corta con tus propias palabras (sin pegar texto de la base)."}},
                      ["consulta"]),
}
_EJECUTORES = {"buscar_en_base": buscar_en_base, "calcular_integral": calcular_integral,
               "verificar_antiderivada": verificar_antiderivada, "buscar_en_web": buscar_en_web}


def definicion(fn, propiedades: dict, requeridos: list[str]) -> dict:
    """Formato de tool de la API de Anthropic: el docstring es la descripcion."""
    return {"name": fn.__name__, "description": inspect.getdoc(fn),
            "input_schema": {"type": "object", "properties": propiedades, "required": requeridos}}


def definiciones() -> list[dict]:
    return [definicion(_EJECUTORES[n], p, r) for n, (p, r) in _ESQUEMAS.items()]


def ejecutar(nombre: str, argumentos, estado: dict) -> dict:
    """Ejecuta una tool pedida por el modelo. Nunca lanza: un fallo vuelve como {"error": ...}
    para que el modelo lo lea. Solo se aceptan los argumentos declarados en el esquema.
    `estado` es el diccionario de UNA pregunta (lo crea agent.py)."""
    if nombre not in _EJECUTORES:
        return {"error": f"Tool desconocida: {nombre!r}."}
    if not isinstance(argumentos, dict):
        return {"error": "Los argumentos deben ser un objeto JSON."}
    permitidos = set(_ESQUEMAS[nombre][0])
    if set(argumentos) - permitidos:
        return {"error": f"Argumentos no permitidos: {sorted(set(argumentos) - permitidos)}."}
    if nombre == "buscar_en_web" and not estado.get("base_consultada"):   # politica "primero la base", en codigo
        return {"error": "Primero consulta la base con buscar_en_base; la web solo se usa si la base no tiene la respuesta."}
    try:
        resultado = _EJECUTORES[nombre](**argumentos)
    except TypeError as e:
        return {"error": f"Argumentos invalidos para {nombre}: {e}"}
    except Exception as e:  # noqa: BLE001 - una tool rota no debe tumbar el agente
        return {"error": f"La tool {nombre} fallo ({type(e).__name__})."}
    if nombre == "buscar_en_base" and "error" not in resultado:
        estado["base_consultada"] = True
    return resultado
