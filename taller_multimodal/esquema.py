"""Esquema de extracción (parte 1.2), validación y comparación con la verdad de terreno.

Dominio: ejercicios de integrales de Cálculo II resueltos paso a paso en una imagen.
Esta lógica NO llama a la API, así que se puede probar sin conexión (test_local.py).
"""
from __future__ import annotations

import json
import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

TIPOS = ["definida", "indefinida"]
TECNICAS = ["sustitucion", "partes", "sustitucion_trigonometrica", "fracciones_parciales", "directa", "otra"]


class Extraccion(BaseModel):
    """Los 5 campos que debe devolver el modelo. `extra=forbid` rechaza campos inventados."""

    model_config = ConfigDict(extra="forbid")

    enunciado: str = Field(min_length=3)
    tipo_integral: Literal["definida", "indefinida"]
    tecnica: Literal["sustitucion", "partes", "sustitucion_trigonometrica",
                     "fracciones_parciales", "directa", "otra"]
    resultado_final: str = Field(min_length=1)
    numero_pasos: int = Field(ge=0, le=30)


CAMPOS = list(Extraccion.model_fields)

# Tabla para el informe (esquema: campo, tipo, descripción)
DESCRIPCION_ESQUEMA = [
    ("enunciado", "texto", "Integral planteada al inicio, en texto o LaTeX."),
    ("tipo_integral", "enum: definida | indefinida", "Definida si tiene límites de integración."),
    ("tecnica", "enum: " + " | ".join(TECNICAS), "Técnica principal que usa el desarrollo."),
    ("resultado_final", "texto", "Resultado escrito al final del último paso."),
    ("numero_pasos", "entero (0–30)", "Cantidad de pasos numerados (\"Paso n\") escritos en la imagen."),
]

PROMPT_EXTRACCION = (
    "Eres un asistente que extrae datos de imágenes de ejercicios de integrales de Cálculo II. "
    "Lee la imagen y responde ÚNICAMENTE con un objeto JSON válido, sin texto adicional ni bloques "
    "de código, con exactamente estas claves:\n"
    '- "enunciado": string con la integral planteada al inicio (texto o LaTeX).\n'
    '- "tipo_integral": "definida" (tiene límites) o "indefinida".\n'
    '- "tecnica": una de ' + json.dumps(TECNICAS) + ".\n"
    '- "resultado_final": string con el resultado escrito al final del último paso.\n'
    '- "numero_pasos": entero con la cantidad de pasos numerados ("Paso n") que aparecen.\n'
    "No inventes datos que no se vean en la imagen."
)


# ---------------------------------------------------------------- validación
def extraer_json(texto: str) -> object:
    """Convierte la respuesta del modelo en objeto Python (tolera ```json ... ```)."""
    t = texto.strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t, flags=re.IGNORECASE).strip()
    if not t.startswith("{"):  # a veces hay texto antes/después: tomamos el primer {...}
        m = re.search(r"\{.*\}", t, flags=re.DOTALL)
        if m:
            t = m.group(0)
    return json.loads(t)


def validar(texto: str) -> tuple[Extraccion | None, str | None]:
    """Devuelve (objeto, None) si cumple el esquema o (None, motivo) si no. Nunca lanza."""
    try:
        datos = extraer_json(texto)
    except json.JSONDecodeError as e:
        return None, f"No es JSON válido: {e.msg} (línea {e.lineno}, columna {e.colno})"
    try:
        return Extraccion.model_validate(datos), None
    except ValidationError as e:
        partes = [f"{'.'.join(str(x) for x in err['loc']) or '(raíz)'}: {err['msg']}" for err in e.errors()]
        return None, "El JSON no cumple el esquema -> " + "; ".join(partes)


# --------------------------------------------------------------- comparación
def normalizar(s: str) -> str:
    """Normaliza notación matemática para comparar texto ignorando formato LaTeX/Unicode."""
    s = str(s).lower()
    for a, b in {"−": "-", "–": "-", "∫": r"\int", "√": r"\sqrt", "²": "^2", "³": "^3",
                 "θ": "theta", "·": "", "×": "", "\\dfrac": "\\frac", "\\tfrac": "\\frac"}.items():
        s = s.replace(a, b)
    for _ in range(3):  # fracciones anidadas
        s = re.sub(r"\\frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}", r"(\1)/(\2)", s)
        s = re.sub(r"\\sqrt\s*\{([^{}]*)\}", r"sqrt(\1)", s)
    s = re.sub(r"\\(left|right|big|,|!|;|\s)", "", s)
    s = re.sub(r"[\s\\${}()\[\]|*]", "", s)
    return s


def comparar(extraido: Extraccion, real: dict) -> list[dict]:
    """Compara campo por campo. Devuelve filas {campo, real, extraido, correcto}."""
    filas = []
    for campo in CAMPOS:
        v_real = real[campo]
        v_ext = getattr(extraido, campo)
        if campo in ("enunciado", "resultado_final"):
            ok = normalizar(v_real) == normalizar(v_ext)
        else:
            ok = v_real == v_ext
        filas.append({"campo": campo, "real": v_real, "extraido": v_ext, "correcto": ok})
    return filas


def filas_invalidas(real: dict, motivo: str) -> list[dict]:
    """Si el JSON no es válido, todos los campos cuentan como no acertados."""
    return [{"campo": c, "real": real[c], "extraido": f"(sin dato: {motivo})", "correcto": False} for c in CAMPOS]
