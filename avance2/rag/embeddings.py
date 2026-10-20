"""Embeddings 100% locales con sentence-transformers.

Privacidad: el modelo se descarga UNA vez desde Hugging Face (solo pesos; no
se envia ningun dato) y despues todo el calculo ocurre en tu CPU. Ni los
documentos ni las consultas salen de la maquina para ser vectorizados.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np

from . import config


@lru_cache(maxsize=1)
def cargar_modelo():
    # Import dentro de la funcion: importar torch tarda unos segundos y no
    # hace falta para modulos que solo leen/trocean texto.
    from sentence_transformers import SentenceTransformer

    # Primero SOLO desde la cache local: sin red, sin comprobar el Hub (mas
    # rapido y, ademas, no sale ni una peticion). Solo si el modelo no esta
    # descargado todavia se contacta con Hugging Face.
    try:
        modelo = SentenceTransformer(config.MODELO_EMBEDDINGS, device="cpu", local_files_only=True)
    except OSError:
        modelo = SentenceTransformer(config.MODELO_EMBEDDINGS, device="cpu")
    # Fijado explicitamente: si el modelo cambiara su tope por defecto, el
    # truncado dejaria de ser el que medimos al trocear.
    modelo.max_seq_length = config.EMBEDDING_MAX_TOKENS
    return modelo


def contar_tokens(texto: str) -> int:
    """Tokens que ocupara `texto` para el modelo (incluye <s> y </s>)."""
    return len(cargar_modelo().tokenizer.encode(texto, add_special_tokens=True))


def _embeber(textos: list[str], batch_size: int) -> np.ndarray:
    # normalize_embeddings=True -> vectores de norma 1. Asi la similitud coseno
    # es un producto punto y la distancia coseno de Chroma (1 - coseno) queda
    # en [0, 2], con 0 = identico.
    return cargar_modelo().encode(
        textos, batch_size=batch_size, normalize_embeddings=True,
        show_progress_bar=False, convert_to_numpy=True,
    ).astype(np.float32)


def embeber_pasajes(textos: list[str], batch_size: int = 16) -> np.ndarray:
    """E5 exige el prefijo 'passage: ' en los documentos que se indexan."""
    return _embeber([config.PREFIJO_PASAJE + t for t in textos], batch_size)


def embeber_consultas(consultas: list[str]) -> np.ndarray:
    """Varias preguntas a la vez (las usa la metrica answer_relevancy de Ragas,
    que compara preguntas con preguntas: ambas lados llevan 'query: ')."""
    return _embeber([config.PREFIJO_CONSULTA + t for t in consultas], 16)


def embeber_consulta(consulta: str) -> np.ndarray:
    """... y 'query: ' en las preguntas. Mezclarlos degrada la recuperacion."""
    return _embeber([config.PREFIJO_CONSULTA + consulta], 1)[0]
