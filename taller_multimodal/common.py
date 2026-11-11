"""Utilidades compartidas del taller multimodal (visión + audio).

Todo el taller usa un solo proveedor (Groq) y una sola clave: GROQ_API_KEY.
La clave NUNCA va en el código: se lee de variables de entorno o de un
archivo .env (que está en .gitignore).
"""
from __future__ import annotations

import base64
import json
import mimetypes
import os
import re
import sys
from pathlib import Path

from dotenv import load_dotenv

# La consola de Windows usa cp1252 por defecto: forzamos UTF-8 para tildes y símbolos.
for _flujo in (sys.stdout, sys.stderr):
    try:
        _flujo.reconfigure(encoding="utf-8")
    except Exception:  # pragma: no cover
        pass

BASE = Path(__file__).resolve().parent
IMAGENES = BASE / "imagenes"
AUDIOS = BASE / "audios"
RESULTADOS = BASE / "resultados"
RESULTADOS.mkdir(exist_ok=True)

# Se cargan primero el .env del taller y luego el de la raíz del repositorio.
load_dotenv(BASE / ".env")
load_dotenv(BASE.parent / ".env")

# Modelos (se pueden cambiar con variables de entorno, sin tocar el código).
MODELO_VISION = os.getenv("GROQ_VISION_MODEL", "qwen/qwen3.8-27b")
MODELO_AUDIO = os.getenv("GROQ_AUDIO_MODEL", "whisper-large-v3-turbo")
MODELO_TEXTO = os.getenv("GROQ_TEXT_MODEL", MODELO_VISION)

# Límites documentados para visión (enunciado del taller).
MAX_IMAGENES = 3
MAX_BYTES_SOLICITUD = 20 * 1024 * 1024  # 20 MB por solicitud


def obtener_cliente():
    """Crea el cliente de Groq o termina con un mensaje claro si falta la clave."""
    from groq import Groq

    clave = os.getenv("GROQ_API_KEY")
    if not clave:
        sys.exit(
            "Falta GROQ_API_KEY. Créala gratis en https://console.groq.com/keys y "
            "pégala en un archivo .env (copia .env.example). No la subas a git."
        )
    return Groq(api_key=clave)


# ---------------------------------------------------------------- imágenes
def imagen_a_data_url(ruta: Path | str, mime_forzado: str | None = None) -> str:
    """Lee una imagen local y la devuelve como data URL base64.

    `mime_forzado` existe solo para el caso de error provocado de la parte 1.4
    (declarar un tipo de imagen distinto al real).
    """
    ruta = Path(ruta)
    if not ruta.is_absolute() and not ruta.exists():
        ruta = IMAGENES / ruta
    datos = ruta.read_bytes()
    mime = mime_forzado or mimetypes.guess_type(ruta.name)[0] or "image/png"
    b64 = base64.b64encode(datos).decode("ascii")
    return f"data:{mime};base64,{b64}"


def bloques_imagen(rutas: list[Path | str]) -> list[dict]:
    return [{"type": "image_url", "image_url": {"url": imagen_a_data_url(r)}} for r in rutas]


_THINK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def limpiar_respuesta(texto: str | None) -> str:
    """Quita bloques <think>…</think> que algunos modelos de razonamiento incluyen."""
    return _THINK.sub("", texto or "").strip()


def preguntar_con_imagenes(
    cliente,
    rutas: list[Path | str],
    pregunta: str,
    sistema: str | None = None,
    json_mode: bool = False,
    modelo: str | None = None,
    max_tokens: int = 2000,
) -> str:
    """Envía una pregunta + 1..3 imágenes al modelo de visión y devuelve el texto."""
    mensajes: list[dict] = []
    if sistema:
        mensajes.append({"role": "system", "content": sistema})
    contenido = [{"type": "text", "text": pregunta}] + bloques_imagen(rutas)
    mensajes.append({"role": "user", "content": contenido})

    args = dict(
        model=modelo or MODELO_VISION,
        messages=mensajes,
        temperature=0,
        max_completion_tokens=max_tokens,
    )
    if json_mode:
        args["response_format"] = {"type": "json_object"}
    try:
        r = cliente.chat.completions.create(**args)
    except Exception as e:
        # Algunos modelos no admiten el modo JSON: reintentamos sin él (el JSON
        # se pide igualmente en el prompt y se valida en el código).
        if json_mode and getattr(e, "status_code", None) == 400:
            print("   (el modelo rechazó response_format=json_object; reintento sin él)")
            args.pop("response_format")
            r = cliente.chat.completions.create(**args)
        else:
            raise
    return limpiar_respuesta(r.choices[0].message.content)


def preguntar_texto(cliente, pregunta: str, sistema: str | None = None,
                    modelo: str | None = None, max_tokens: int = 1200) -> str:
    mensajes: list[dict] = []
    if sistema:
        mensajes.append({"role": "system", "content": sistema})
    mensajes.append({"role": "user", "content": pregunta})
    r = cliente.chat.completions.create(
        model=modelo or MODELO_TEXTO,
        messages=mensajes,
        temperature=0,
        max_completion_tokens=max_tokens,
    )
    return limpiar_respuesta(r.choices[0].message.content)


# --------------------------------------------------------------- resultados
def guardar_json(nombre: str, objeto) -> Path:
    ruta = RESULTADOS / f"{nombre}.json"
    ruta.write_text(json.dumps(objeto, ensure_ascii=False, indent=2), encoding="utf-8")
    return ruta


def guardar_md(nombre: str, texto: str) -> Path:
    ruta = RESULTADOS / f"{nombre}.md"
    ruta.write_text(texto, encoding="utf-8")
    return ruta


def celda(valor) -> str:
    """Escapa un valor para una celda de tabla Markdown."""
    return str(valor).replace("|", "\\|").replace("\n", " ")
