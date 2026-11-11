"""Parte 1.2: extracción estructurada (JSON validado) de 3 imágenes y % de aciertos.

Uso:
    python parte1_2_extraccion.py

Para cada imagen de imagenes/verdad_terreno.json:
  1. pide al modelo un JSON con los 5 campos del esquema (esquema.py);
  2. valida el JSON con pydantic: si no cumple, informa del motivo y CONTINÚA;
  3. compara campo por campo con el valor real y calcula el porcentaje de aciertos.
"""
import json
from datetime import datetime

import common as c
import esquema as e


def main():
    verdad = json.loads((c.IMAGENES / "verdad_terreno.json").read_text(encoding="utf-8"))
    verdad = {k: v for k, v in verdad.items() if not k.startswith("_")}
    cliente = c.obtener_cliente()
    print(f"Modelo: {c.MODELO_VISION}\n")

    filas_total, detalle = [], []
    for nombre, real in verdad.items():
        ruta = c.IMAGENES / nombre
        print(f"== {nombre}")
        bruto = c.preguntar_con_imagenes(
            cliente, [ruta], "Extrae los datos de este ejercicio.",
            sistema=e.PROMPT_EXTRACCION, json_mode=True)
        print("   respuesta cruda:", bruto.replace("\n", " ")[:300])
        objeto, motivo = e.validar(bruto)
        if objeto is None:
            print("   JSON NO VÁLIDO ->", motivo)
            filas = e.filas_invalidas(real, motivo)
        else:
            print("   JSON válido.")
            filas = e.comparar(objeto, real)
        for f in filas:
            f["imagen"] = nombre
            print(f"   {'✔' if f['correcto'] else '✘'} {f['campo']}: real={f['real']!r}  extraído={f['extraido']!r}")
        filas_total += filas
        detalle.append({"imagen": nombre, "respuesta_cruda": bruto, "valido": objeto is not None,
                        "motivo_invalido": motivo})

    aciertos = sum(f["correcto"] for f in filas_total)
    total = len(filas_total)
    pct = 100 * aciertos / total if total else 0
    print(f"\nPorcentaje de aciertos: {aciertos} de {total} campos = {pct:.1f} %")

    c.guardar_json("parte1_2", {"fecha": datetime.now().isoformat(timespec="seconds"),
                                "modelo": c.MODELO_VISION, "aciertos": aciertos, "total": total,
                                "porcentaje": round(pct, 1), "filas": filas_total, "detalle": detalle})

    md = ["## 1.2 Extracción estructurada", "", "### Esquema", "",
          "| Campo | Tipo | Descripción |", "|---|---|---|"]
    md += [f"| `{a}` | {c.celda(b)} | {c.celda(d)} |" for a, b, d in e.DESCRIPCION_ESQUEMA]
    md += ["", "### Tabla de aciertos", "",
           "| Imagen | Campo | Valor real | Valor extraído | ¿Correcto? |", "|---|---|---|---|---|"]
    md += [f"| {f['imagen']} | {f['campo']} | {c.celda(f['real'])} | {c.celda(f['extraido'])} | "
           f"{'Sí' if f['correcto'] else 'No'} |" for f in filas_total]
    md += ["", f"**Porcentaje de aciertos:** {aciertos} de {total} campos = {pct:.1f} %", "",
           "> Nota: `enunciado` y `resultado_final` se comparan tras normalizar la notación "
           "(LaTeX/Unicode). Revisa a mano los \"No\" de esos dos campos: pueden ser equivalentes.", ""]
    c.guardar_md("parte1_2", "\n".join(md))
    print("Guardado en resultados/parte1_2.json y .md")


if __name__ == "__main__":
    main()
