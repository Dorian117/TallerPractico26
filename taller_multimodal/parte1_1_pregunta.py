"""Parte 1.1: pregunta sobre UNA imagen local (base64), como en multimodal.py.

Uso:
    python parte1_1_pregunta.py
    python parte1_1_pregunta.py --imagen ej2_partes.png --pregunta "¿Qué técnica se usa y está bien aplicada?"
"""
import argparse
from datetime import datetime

import common as c

PREGUNTA_DEFECTO = (
    "¿Qué sustitución trigonométrica debo usar para integrar una expresión que contiene "
    "√(9 − x²)? Justifica con la tabla de la imagen."
)
SISTEMA = ("Eres un tutor de Cálculo II. Responde en español, de forma breve y precisa, "
           "basándote en lo que se ve en la imagen.")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--imagen", default="ej4_tabla_casos.png")
    ap.add_argument("--pregunta", default=PREGUNTA_DEFECTO)
    args = ap.parse_args()

    ruta = c.IMAGENES / args.imagen
    if not ruta.exists():
        raise SystemExit(f"No existe la imagen: {ruta}")

    cliente = c.obtener_cliente()
    print(f"Modelo: {c.MODELO_VISION}\nImagen: {ruta.name}\nPregunta: {args.pregunta}\n")
    respuesta = c.preguntar_con_imagenes(cliente, [ruta], args.pregunta, sistema=SISTEMA)
    print("Respuesta del modelo:\n" + respuesta)

    c.guardar_json("parte1_1", {"fecha": datetime.now().isoformat(timespec="seconds"),
                                "modelo": c.MODELO_VISION, "imagen": ruta.name,
                                "pregunta": args.pregunta, "respuesta": respuesta})
    c.guardar_md("parte1_1", f"""## 1.1 Pregunta sobre una imagen

- **Imagen:** `{ruta.name}`
- **Pregunta:** {args.pregunta}

**Respuesta del modelo ({c.MODELO_VISION}):**

{respuesta}
""")
    print("\nGuardado en resultados/parte1_1.json y .md")


if __name__ == "__main__":
    main()
