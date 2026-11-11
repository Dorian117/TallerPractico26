"""Métrica simple para evaluar transcripciones: WER (tasa de error de palabras).

WER = (sustituciones + inserciones + eliminaciones) / palabras de la referencia.
0 % = transcripción idéntica. Sin dependencias externas.
"""
from __future__ import annotations

import re
import unicodedata


def _palabras(texto: str) -> list[str]:
    t = unicodedata.normalize("NFC", texto.lower())
    t = re.sub(r"[¿?¡!.,;:\"'()\[\]]", " ", t)
    return t.split()


def wer(referencia: str, hipotesis: str) -> float:
    ref, hip = _palabras(referencia), _palabras(hipotesis)
    if not ref:
        return 0.0 if not hip else 1.0
    # distancia de edición por palabras (programación dinámica)
    prev = list(range(len(hip) + 1))
    for i, r in enumerate(ref, 1):
        act = [i] + [0] * len(hip)
        for j, h in enumerate(hip, 1):
            act[j] = min(prev[j] + 1, act[j - 1] + 1, prev[j - 1] + (r != h))
        prev = act
    return prev[-1] / len(ref)
