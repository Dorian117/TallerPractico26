"""Tutor por consola.

    python -m rag.cli            # chat interactivo ('nueva' reinicia, 'salir' termina)
    python -m rag.cli --demo     # demo guionizada de 3 turnos (con seguimiento y fuera de corpus)
"""
from __future__ import annotations

import argparse
import sys

from . import config
from .generate import Respuesta, Turno, responder
from .llm import MENSAJE_LIMITE, LimiteDeUsoError, crear_llm

DEMO = [
    r"Intenté resolver $\int \frac{dx}{\sqrt{4x^2+1}}$ con $x = \frac{\tan\theta}{2}$ pero no sé cómo "
    r"seguir. ¿Me muestras la solución completa?",
    r"¿Y si en lugar de $4x^2$ fuera $9x^2$?",
    r"¿Cómo calculo el volumen de un sólido de revolución por el método de discos?",
]


def mostrar(r: Respuesta) -> None:
    if r.pregunta_recuperacion:
        print(f"\n  [consulta usada para buscar] {r.pregunta_recuperacion}")
    print("  [fragmentos recuperados]")
    for i, f in enumerate(r.fragmentos, start=1):
        print(f"    [{i}] dist={f.distancia:.4f} {f.documento[:32]} > {f.seccion[-58:]}")
    print(f"\nTutor:\n{r.texto}")
    pie = (f"\n  [modelo={r.modelo or config.LLM_MODEL} | tokens entrada={r.tokens_entrada} "
           f"salida={r.tokens_salida} | {r.segundos:.1f}s | llamo_modelo={r.llamo_modelo} "
           f"| desde_cache={r.desde_cache}]")
    print(pie)
    if r.citas_invalidas:
        print(f"  [ATENCION: citas a fragmentos inexistentes: {r.citas_invalidas}]")
    for a in r.avisos:
        print(f"  [aviso] {a}")


def conversar(mensajes: list[str] | None) -> None:
    historial: list[Turno] = []
    pendientes = list(mensajes or [])
    # La demo guionizada usa cache en disco (repetirla no vuelve a gastar);
    # el chat interactivo no guarda las preguntas en disco.
    llm = crear_llm("tutor", usar_cache=mensajes is not None)
    while True:
        if mensajes is not None:
            if not pendientes:
                return
            pregunta = pendientes.pop(0)
            print(f"\n{'=' * 90}\nEstudiante: {pregunta}")
        else:
            try:
                pregunta = input("\nEstudiante: ").strip()
            except (EOFError, KeyboardInterrupt):
                return
            if not pregunta:
                continue
            if pregunta.lower() == "salir":
                return
            if pregunta.lower() == "nueva":
                historial.clear()
                print("(conversacion reiniciada)")
                continue
        try:
            r = responder(pregunta, historial, llm=llm)
        except LimiteDeUsoError:
            print(f"\n  [ERROR] {MENSAJE_LIMITE}.")
            if mensajes is not None:      # demo guionizada: no se sigue con el historial roto
                return
            continue
        except Exception as e:  # noqa: BLE001 - clave invalida, red, servicio caido...
            # Un fallo de la API no debe tumbar la sesion ni contaminar el
            # historial: el turno fallido no se guarda.
            print(f"\n  [ERROR] {type(e).__name__}: {str(e)[:220]}\n  (el turno no se guardo en el historial)")
            if mensajes is not None:
                return
            continue
        mostrar(r)
        historial += [Turno("user", pregunta), Turno("assistant", r.texto)]


def main() -> None:
    # la consola de Windows no es UTF-8 por defecto; line_buffering muestra cada turno al momento
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true")
    args = ap.parse_args()
    print("TUTOR DE CALCULO II (RAG) | 'salir' para terminar, 'nueva' para reiniciar" if not args.demo
          else "DEMO: 3 turnos (solucion completa -> seguimiento -> fuera de corpus)")
    conversar(DEMO if args.demo else None)


if __name__ == "__main__":
    main()
