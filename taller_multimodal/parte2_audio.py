"""Parte 2: audio -> transcripción (Whisper) -> respuesta del modelo de texto.

2.1  Transcribe cada audio de audios/guion.json con Whisper y mide el WER contra lo que se dijo.
2.2  Pasa la transcripción al modelo para que responda como tutor de Cálculo II.

Uso:
    python parte2_audio.py                  # 2.1 y 2.2
    python parte2_audio.py --solo-transcribir
    python parte2_audio.py --sin-prompt     # sin vocabulario técnico (para comparar el efecto)
    python parte2_audio.py --modelo whisper-large-v3
"""
import argparse
import json
import wave
from datetime import datetime
from pathlib import Path

import common as c
import metricas

EXTENSIONES = [".m4a", ".mp3", ".wav", ".ogg", ".webm", ".mp4", ".mpeg", ".flac"]
MAX_BYTES_AUDIO = 25 * 1024 * 1024  # 25 MB (plan gratuito)

# Vocabulario del dominio: se pasa como `prompt` a Whisper para sesgar la transcripción.
VOCABULARIO = ("Cálculo II: integración por partes, regla ILATE, sustitución trigonométrica, "
               "antiderivada, integral definida, integral indefinida, teorema fundamental del cálculo, "
               "secante, tangente, theta, logaritmo natural.")

SISTEMA_TUTOR = ("Eres un tutor de Cálculo II. Responde en español, de forma clara y breve "
                 "(máximo 150 palabras), con las fórmulas esenciales. Si la pregunta parece mal "
                 "transcrita, di qué entendiste.")


def localizar(nombre: str) -> Path | None:
    ruta = c.AUDIOS / nombre
    if ruta.exists():
        return ruta
    for ext in EXTENSIONES:  # mismo nombre con otra extensión
        alt = ruta.with_suffix(ext)
        if alt.exists():
            return alt
    return None


def duracion(ruta: Path, declarada):
    """Duración en segundos: wav con la librería estándar, otros con mutagen si está instalado."""
    try:
        if ruta.suffix.lower() == ".wav":
            with wave.open(str(ruta)) as w:
                return round(w.getnframes() / w.getframerate(), 1)
        from mutagen import File
        m = File(str(ruta))
        if m is not None and m.info is not None:
            return round(m.info.length, 1)
    except Exception:  # noqa: BLE001
        pass
    return declarada


def transcribir(cliente, ruta: Path, modelo: str, usar_prompt: bool) -> str:
    kwargs = dict(model=modelo, language="es", temperature=0.0, response_format="json")
    if usar_prompt:
        kwargs["prompt"] = VOCABULARIO
    r = cliente.audio.transcriptions.create(file=(ruta.name, ruta.read_bytes()), **kwargs)
    return r.text.strip()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--modelo", default=c.MODELO_AUDIO)
    ap.add_argument("--sin-prompt", action="store_true")
    ap.add_argument("--solo-transcribir", action="store_true")
    args = ap.parse_args()

    guion = json.loads((c.AUDIOS / "guion.json").read_text(encoding="utf-8"))["audios"]
    cliente = c.obtener_cliente()
    print(f"Whisper: {args.modelo} | texto: {c.MODELO_TEXTO} | "
          f"vocabulario: {'no' if args.sin_prompt else 'sí'}\n")

    filas = []
    for item in guion:
        ruta = localizar(item["archivo"])
        print(f"== {item['archivo']} ({item['hablante']}, {item['condicion']})")
        if ruta is None:
            print("   (no existe el archivo de audio: grábalo y guárdalo en audios/)\n")
            continue
        if ruta.stat().st_size > MAX_BYTES_AUDIO:
            print("   (supera 25 MB: comprímelo o recórtalo)\n")
            continue
        dur = duracion(ruta, item.get("duracion_s"))
        texto = transcribir(cliente, ruta, args.modelo, not args.sin_prompt)
        error = metricas.wer(item["dijo"], texto)
        print(f"   dijo:          {item['dijo']}\n   transcripción: {texto}\n   WER: {error:.0%}")
        fila = {"archivo": ruta.name, "hablante": item["hablante"], "condicion": item["condicion"],
                "duracion_s": dur, "dijo": item["dijo"], "transcripcion": texto, "wer": round(error, 3)}
        if not args.solo_transcribir:
            respuesta = c.preguntar_texto(cliente, texto, sistema=SISTEMA_TUTOR)
            fila["respuesta"] = respuesta
            print("   respuesta:", respuesta.replace("\n", " ")[:400])
        filas.append(fila)
        print()

    if not filas:
        raise SystemExit("No se procesó ningún audio. Graba los archivos de audios/guion.json.")

    wer_medio = sum(f["wer"] for f in filas) / len(filas)
    print(f"WER promedio: {wer_medio:.1%} en {len(filas)} audios")
    sufijo = "_sinprompt" if args.sin_prompt else ""
    c.guardar_json("parte2" + sufijo, {"fecha": datetime.now().isoformat(timespec="seconds"),
                                        "modelo_audio": args.modelo, "modelo_texto": c.MODELO_TEXTO,
                                        "con_vocabulario": not args.sin_prompt,
                                        "wer_promedio": round(wer_medio, 3), "audios": filas})

    md = ["## 2.1 Transcripción desde archivo", "",
          "| Audio | Hablante | Condición | Duración (s) | Lo que se dijo | Transcripción | WER |",
          "|---|---|---|---|---|---|---|"]
    md += [f"| {f['archivo']} | {f['hablante']} | {f['condicion']} | {f['duracion_s']} | "
           f"{c.celda(f['dijo'])} | {c.celda(f['transcripcion'])} | {f['wer']:.0%} |" for f in filas]
    md += ["", f"**WER promedio:** {wer_medio:.1%}", ""]
    if not args.solo_transcribir:
        md += ["## 2.2 Respuesta del modelo a partir de la transcripción", "",
               "| Audio | Transcripción enviada al modelo | Respuesta del modelo | ¿Correcta y pertinente? (Sí/No) |",
               "|---|---|---|---|"]
        md += [f"| {f['archivo']} | {c.celda(f['transcripcion'])} | {c.celda(f['respuesta'])} | (revisar) |"
               for f in filas]
        md.append("")
    c.guardar_md("parte2" + sufijo, "\n".join(md))
    print(f"Guardado en resultados/parte2{sufijo}.json y .md")


if __name__ == "__main__":
    main()
