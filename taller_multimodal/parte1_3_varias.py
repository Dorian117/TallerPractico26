"""Parte 1.3: varias imágenes (2 o 3) en una misma solicitud.

Por defecto compara la solución correcta del Ejercicio 2 con la de otro estudiante,
que tiene un error de signo, y pide encontrarlo.

Uso:
    python parte1_3_varias.py
    python parte1_3_varias.py --imagenes ej2_partes.png ej5_error_signo.png ej4_tabla_casos.png
"""
import argparse
from datetime import datetime

import common as c

PREGUNTA_DEFECTO = (
    "Las dos imágenes resuelven la misma integral por partes. Compara las soluciones: "
    "¿cuál está bien y cuál tiene un error? Indica en qué paso está el error y cómo se corrige."
)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--imagenes", nargs="+", default=["ej2_partes.png", "ej5_error_signo.png"])
    ap.add_argument("--pregunta", default=PREGUNTA_DEFECTO)
    args = ap.parse_args()

    if not 2 <= len(args.imagenes) <= c.MAX_IMAGENES:
        raise SystemExit(f"Envía entre 2 y {c.MAX_IMAGENES} imágenes (límite del modelo).")
    rutas = [c.IMAGENES / n for n in args.imagenes]
    for r in rutas:
        if not r.exists():
            raise SystemExit(f"No existe la imagen: {r}")
    peso = sum(r.stat().st_size for r in rutas)
    print(f"Modelo: {c.MODELO_VISION}\nImágenes: {[r.name for r in rutas]} ({peso/1024:.0f} KB)\n"
          f"Pregunta: {args.pregunta}\n")

    cliente = c.obtener_cliente()
    respuesta = c.preguntar_con_imagenes(
        cliente, rutas, args.pregunta,
        sistema="Eres un tutor de Cálculo II. Responde en español de forma breve y concreta.")
    print("Respuesta del modelo:\n" + respuesta)

    c.guardar_json("parte1_3", {"fecha": datetime.now().isoformat(timespec="seconds"),
                                "modelo": c.MODELO_VISION, "imagenes": args.imagenes,
                                "pregunta": args.pregunta, "respuesta": respuesta})
    c.guardar_md("parte1_3", f"""## 1.3 Varias imágenes en una solicitud

- **Imágenes enviadas:** {', '.join('`'+n+'`' for n in args.imagenes)}
- **Pregunta:** {args.pregunta}

**Respuesta del modelo ({c.MODELO_VISION}):**

{respuesta}
""")
    print("\nGuardado en resultados/parte1_3.json y .md")


if __name__ == "__main__":
    main()
