"""Calculo simbolico SEGURO con sympy (lo usan las tools de tarea del agente).

El texto viene del estudiante a traves del LLM, asi que NUNCA se hace `eval` sobre
el. La defensa son tres capas:
  1. Lista blanca de caracteres y de nombres (solo variables, constantes y funciones
     matematicas conocidas): sin atributos, sin guiones bajos, sin builtins.
  2. `parse_expr` con transformaciones restringidas y un `global_dict` propio, con
     `__builtins__` vacio.
  3. Quien llama (tools.py) ejecuta esto en un PROCESO aparte con tiempo maximo.

Todas las funciones publicas devuelven un dict; un fallo es {"error": "..."} con un
mensaje que el agente pueda usar para corregir la llamada.
"""
from __future__ import annotations

import re

import sympy as sp
from sympy.parsing.sympy_parser import convert_xor, implicit_multiplication, parse_expr, standard_transformations

from . import config

VARIABLES = tuple("xyztuvwsr")
_FUNCIONES = {
    "sin": sp.sin, "cos": sp.cos, "tan": sp.tan, "cot": sp.cot, "sec": sp.sec, "csc": sp.csc,
    "asin": sp.asin, "acos": sp.acos, "atan": sp.atan, "arcsin": sp.asin, "arccos": sp.acos, "arctan": sp.atan,
    "sinh": sp.sinh, "cosh": sp.cosh, "tanh": sp.tanh,
    "exp": sp.exp, "log": sp.log, "ln": sp.log, "sqrt": sp.sqrt, "abs": sp.Abs, "Abs": sp.Abs,
}
_CONSTANTES = {"pi": sp.pi, "e": sp.E, "E": sp.E}
_TRANSFORMACIONES = standard_transformations + (convert_xor, implicit_multiplication)
_SOLO_CARACTERES = re.compile(r"^[A-Za-z0-9_+\-*/^().\s]+$")
_AYUDA = ("Usa sintaxis simple: x**2, sqrt(x), sin(x), cos(x), tan(x), sec(x), exp(x), log(x), "
          "atan(x), pi, e; multiplicacion siempre con * (por ejemplo x*exp(x)). Sin LaTeX.")


def _error(msg: str) -> dict:
    return {"error": msg}


def _validar(texto, permitidos: set[str]) -> str | None:
    """Devuelve un mensaje de error o None si el texto es aceptable."""
    if not isinstance(texto, str) or not texto.strip():
        return "La expresion esta vacia o no es texto."
    if len(texto) > config.CALCULO_EXPR_MAX:
        return f"La expresion supera {config.CALCULO_EXPR_MAX} caracteres."
    if not _SOLO_CARACTERES.match(texto):
        return f"Caracteres no permitidos. {_AYUDA}"
    if re.search(r"(?<!\d)\.|\.(?!\d)", texto):
        return "El punto solo se admite en decimales (por ejemplo 0.5)."
    if re.search(r"\d{8,}", texto):
        return "Numero demasiado grande."
    # Torres de potencias (9**9**9) y exceso de potencias: bombas de memoria/tiempo.
    if re.search(r"(\*\*|\^)\s*[A-Za-z0-9_.]+\s*(\*\*|\^)", texto) or len(re.findall(r"\*\*|\^", texto)) > 6:
        return "Potencias encadenadas o demasiadas potencias; usa parentesis y simplifica."
    desconocidos = sorted({n for n in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", texto) if n not in permitidos})
    if desconocidos:
        return f"Nombres no permitidos: {', '.join(desconocidos)}. {_AYUDA}"
    return None


def _entorno(extra: tuple[str, ...] = ()) -> tuple[set[str], dict]:
    locales = {**_FUNCIONES, **_CONSTANTES, **{v: sp.Symbol(v) for v in (*VARIABLES, *extra)}}
    return set(locales), locales


def _parsear(texto, extra: tuple[str, ...] = ()):
    permitidos, locales = _entorno(extra)
    fallo = _validar(texto, permitidos)
    if fallo:
        raise ValueError(fallo)
    glob = {"__builtins__": {}, "Integer": sp.Integer, "Float": sp.Float, "Rational": sp.Rational, "Symbol": sp.Symbol}
    try:
        expr = parse_expr(texto.strip(), local_dict=locales, global_dict=glob, transformations=_TRANSFORMACIONES)
    except Exception as e:  # noqa: BLE001 - cualquier fallo de sintaxis se explica, no se propaga
        raise ValueError(f"No pude interpretar '{texto.strip()[:60]}' ({type(e).__name__}). {_AYUDA}") from e
    if not isinstance(expr, sp.Basic) or expr.has(sp.zoo, sp.nan):
        raise ValueError(f"La expresion '{texto.strip()[:60]}' no es valida o no esta definida.")
    return expr


def _variable(nombre) -> sp.Symbol:
    if not isinstance(nombre, str) or nombre.strip() not in VARIABLES:
        raise ValueError(f"Variable invalida {nombre!r}: usa una letra entre {', '.join(VARIABLES)}.")
    return sp.Symbol(nombre.strip())


def integral(expresion, variable="x", limite_inferior=None, limite_superior=None) -> dict:
    try:
        var = _variable(variable)
        f = _parsear(expresion)
        if (limite_inferior is None) != (limite_superior is None):
            return _error("Para una integral definida da AMBOS limites; para la indefinida, ninguno.")
        if limite_inferior is None:
            F = sp.integrate(f, var)
            if F.has(sp.Integral):
                return _error("sympy no encontro una antiderivada en forma cerrada para esta expresion.")
            return {"tipo": "indefinida", "integrando": str(f), "variable": str(var),
                    "resultado": f"{F} + C", "latex": f"{sp.latex(F)} + C"}
        a, b = _parsear(str(limite_inferior)), _parsear(str(limite_superior))
        if a.free_symbols or b.free_symbols:
            return _error("Los limites deben ser numeros o constantes (por ejemplo 0, 1, pi/2), sin variables.")
        valor = sp.integrate(f, (var, a, b))
        if valor.has(sp.Integral):
            return _error("sympy no pudo evaluar la integral definida en forma cerrada.")
        if valor.has(sp.zoo, sp.oo, -sp.oo, sp.nan):
            return _error("La integral diverge o no esta definida en ese intervalo (integrales impropias no soportadas).")
        return {"tipo": "definida", "integrando": str(f), "variable": str(var), "limite_inferior": str(a),
                "limite_superior": str(b), "resultado_exacto": str(valor), "latex": sp.latex(valor),
                "valor_numerico": str(sp.N(valor, 10))}
    except ValueError as e:
        return _error(str(e))


_PUNTOS = (0.37, 0.55, 0.8, 1.1, 1.3, 1.9, 2.4)   # puntos de comprobacion numerica


def verificar(integrando, candidata, variable="x") -> dict:
    """Deriva la candidata y comprueba que sea el integrando: primero simbolico, luego numerico."""
    try:
        var = _variable(variable)
        f = _parsear(integrando)
        F = _parsear(candidata, extra=("C",))
        d = sp.diff(F, var)
        if sp.simplify(d - f) == 0:
            return {"coincide": True, "metodo": "simbolico", "derivada_de_la_candidata": str(d)}
        validos = fallos = 0
        for p in _PUNTOS:
            try:
                v = complex(sp.N((d - f).subs(var, p)))
            except (TypeError, ValueError, ZeroDivisionError):
                continue            # punto fuera del dominio
            validos += 1
            fallos += abs(v) > 1e-7
        if validos >= 4 and fallos == 0:
            return {"coincide": True, "metodo": "numerico", "derivada_de_la_candidata": str(d),
                    "nota": f"Coincide en {validos} puntos de prueba (no se pudo simplificar a 0 simbolicamente)."}
        return {"coincide": False, "derivada_de_la_candidata": str(d),
                "diferencia_con_el_integrando": str(sp.simplify(d - f))}
    except ValueError as e:
        return _error(str(e))


_ACCIONES = {"integral": integral, "verificar": verificar}


def trabajo(cola, accion: str, kwargs: dict) -> None:
    """Punto de entrada del PROCESO hijo: nunca lanza, siempre deja un dict en la cola."""
    try:
        try:   # Linux: tope de memoria del hijo. shortcut: en Windows no existe `resource`, solo rige el timeout.
            import resource
            resource.setrlimit(resource.RLIMIT_AS, (1_500_000_000, 1_500_000_000))
        except (ImportError, ValueError, OSError):
            pass
        cola.put(_ACCIONES[accion](**kwargs))
    except BaseException as e:  # noqa: BLE001
        cola.put({"error": f"Fallo interno del calculo ({type(e).__name__})."})
