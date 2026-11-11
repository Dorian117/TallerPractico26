"""Parte 1.4: errores provocados a propósito, para ver el mensaje real de la API.

Casos (cada uno se ejecuta dentro de try/except y se registra lo que devuelva la API):
  imagenes  -> 4 imágenes en una solicitud (el límite es 3).
  mime      -> un PNG declarado como image/jpeg (tipo mal declarado).
  sinvision -> una imagen enviada a un modelo de solo texto.

Uso:
    python parte1_4_error.py                 # ejecuta los tres
    python parte1_4_error.py --caso imagenes
    python parte1_4_error.py --modelo-sin-vision llama-3.1-8b-instant

OJO: algunos proveedores toleran ciertos casos (p. ej. el tipo mal declarado). Si un caso NO
falla, eso también es un resultado: regístralo y escoge para el informe uno que sí falle.
"""
import argparse
from datetime import datetime

import common as c

CASOS = ["imagenes", "mime", "sinvision"]


def describir_error(exc: Exception) -> dict:
    return {
        "tipo": type(exc).__name__,
        "status_code": getattr(exc, "status_code", None),
        "mensaje": str(exc),
        "cuerpo": getattr(exc, "body", None),
    }


def ejecutar(caso: str, cliente, modelo_sin_vision: str) -> dict:
    imgs = ["ej1_sust_trig.png", "ej2_partes.png", "ej3_definida_sust.png", "ej4_tabla_casos.png"]
    pregunta = "Describe brevemente la imagen."
    try:
        if caso == "imagenes":
            bloques = c.bloques_imagen([c.IMAGENES / n for n in imgs])  # 4 > 3
            msj = [{"role": "user", "content": [{"type": "text", "text": pregunta}] + bloques}]
            modelo = c.MODELO_VISION
        elif caso == "mime":
            url = c.imagen_a_data_url(c.IMAGENES / imgs[0], mime_forzado="image/jpeg")  # es PNG
            msj = [{"role": "user", "content": [{"type": "text", "text": pregunta},
                                               {"type": "image_url", "image_url": {"url": url}}]}]
            modelo = c.MODELO_VISION
        elif caso == "sinvision":
            msj = [{"role": "user", "content": [{"type": "text", "text": pregunta}] +
                    c.bloques_imagen([c.IMAGENES / imgs[0]])}]
            modelo = modelo_sin_vision
        else:
            raise ValueError(caso)
        r = cliente.chat.completions.create(model=modelo, messages=msj, max_completion_tokens=200)
        return {"caso": caso, "modelo": modelo, "fallo": False,
                "respuesta": c.limpiar_respuesta(r.choices[0].message.content)[:300]}
    except Exception as exc:  # noqa: BLE001  (aquí queremos capturar el error de la API)
        return {"caso": caso, "modelo": locals().get("modelo", "?"), "fallo": True, **describir_error(exc)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--caso", choices=CASOS + ["todos"], default="todos")
    ap.add_argument("--modelo-sin-vision", default="llama-3.1-8b-instant")
    args = ap.parse_args()

    cliente = c.obtener_cliente()
    casos = CASOS if args.caso == "todos" else [args.caso]
    resultados = []
    for caso in casos:
        print(f"\n== Caso: {caso}")
        res = ejecutar(caso, cliente, args.modelo_sin_vision)
        resultados.append(res)
        if res["fallo"]:
            print(f"   ERROR {res['status_code']} ({res['tipo']}):\n   {res['mensaje']}")
            if res["status_code"] == 404:
                print("   PISTA: ese modelo no existe/no está disponible. Ejecuta `python listar_modelos.py` "
                      "y repite con --caso sinvision --modelo-sin-vision <id de un modelo solo texto>.")
        else:
            print("   La API NO devolvió error. Respuesta:", res["respuesta"])

    c.guardar_json("parte1_4", {"fecha": datetime.now().isoformat(timespec="seconds"),
                                "resultados": resultados})
    md = ["## 1.4 Error provocado", ""]
    explicaciones = {
        "imagenes": "Se enviaron 4 imágenes; el modelo admite máximo 3 por solicitud.",
        "mime": "Se declaró image/jpeg para un archivo PNG (tipo de imagen mal declarado).",
        "sinvision": "Se envió una imagen a un modelo que solo acepta texto.",
    }
    for r in resultados:
        md += [f"### Caso `{r['caso']}`", f"- **Qué se hizo:** {explicaciones[r['caso']]}",
               f"- **Modelo:** `{r['modelo']}`"]
        if r["fallo"]:
            md += [f"- **Código HTTP:** {r['status_code']}", "- **Mensaje de error:**", "",
                   "```", r["mensaje"], "```", "- **Causa:** (explícala con tus palabras)", ""]
        else:
            md += ["- La API no devolvió error en este caso.", ""]
    c.guardar_md("parte1_4", "\n".join(md))
    print("\nGuardado en resultados/parte1_4.json y .md")


if __name__ == "__main__":
    main()
