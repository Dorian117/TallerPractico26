"""Prueba de la tool `buscar_en_base` SIN LLM (sin API, gasto 0): la llama directamente.

    python -m tests.prueba_tool_busqueda
    python -m tests.prueba_tool_busqueda "tu consulta"

Imprime la salida tal como la recibiria el modelo (JSON) y comprueba los limites del codigo.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from rag import config, tools  # noqa: E402

CONSULTAS = ["regla nemotecnica ILATE para elegir u y dv", "caso sqrt(a^2 - x^2) en sustitucion trigonometrica"]


def main(consultas: list[str]) -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    print(f"Limites en config.py: K_MAX={config.K_MAX} fragmentos, MAX_CHARS={config.MAX_CHARS} caracteres cada uno\n")
    print("Docstring que ve el modelo:\n" + tools.definiciones()[0]["description"] + "\n")
    for q in consultas:
        salida = tools.buscar_en_base(q)
        print("=" * 100 + f"\nbuscar_en_base(consulta={q!r})\n" + json.dumps(salida, ensure_ascii=False, indent=2))
        frs = salida["fragmentos"]
        assert 0 < len(frs) <= config.K_MAX, "supera K_MAX"
        for f in frs:
            assert len(f["texto"]) <= config.MAX_CHARS + len(" [...recortado]"), "fragmento demasiado largo"
            assert f["origen"] == "base" and f["documento"] and f["seccion"], "falta origen/documento/seccion"
    print("\nOK: K_MAX, MAX_CHARS, origen='base', documento y seccion presentes en todos los fragmentos.")


if __name__ == "__main__":
    main(sys.argv[1:] or CONSULTAS)
