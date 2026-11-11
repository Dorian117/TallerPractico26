"""Pruebas SIN conexión ni clave: validan la lógica propia del taller con un cliente simulado.

    python test_local.py

Comprueba: validación del JSON (válido, malformado, fuera de esquema), normalización de
notación, WER, y que parte1_2 y parte2 corren de principio a fin con respuestas simuladas.
No gasta nada de la API y no sustituye la ejecución real de los scripts.
"""
import json
import shutil
import tempfile
import wave
from pathlib import Path
from types import SimpleNamespace

import common as c
import esquema as e
import metricas

VERDE, ROJO, FIN = "\033[92m", "\033[91m", "\033[0m"
n_ok = n_mal = 0


def comprobar(nombre, condicion):
    global n_ok, n_mal
    if condicion:
        n_ok += 1
        print(f"  {VERDE}✔{FIN} {nombre}")
    else:
        n_mal += 1
        print(f"  {ROJO}✘ {nombre}{FIN}")


def cliente_falso(respuestas_chat, textos_audio=None):
    """Cliente con la misma forma que el de Groq, que devuelve respuestas predefinidas."""
    cola_chat, cola_audio = list(respuestas_chat), list(textos_audio or [])

    def chat(**kwargs):
        msg = SimpleNamespace(content=cola_chat.pop(0))
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)])

    def audio(**kwargs):
        return SimpleNamespace(text=cola_audio.pop(0))

    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=chat)),
                           audio=SimpleNamespace(transcriptions=SimpleNamespace(create=audio)))


def main():
    print("1) Validación del esquema")
    bueno = json.dumps({"enunciado": r"\int x e^x dx", "tipo_integral": "indefinida", "tecnica": "partes",
                        "resultado_final": "x e^x - e^x + C", "numero_pasos": 4})
    obj, motivo = e.validar(bueno)
    comprobar("JSON válido se acepta", obj is not None and motivo is None)
    obj, motivo = e.validar("```json\n" + bueno + "\n```")
    comprobar("tolera bloque ```json```", obj is not None)
    obj, motivo = e.validar("Claro: " + bueno)
    comprobar("tolera texto antes del JSON", obj is not None)
    obj, motivo = e.validar("{esto no es json}")
    comprobar("JSON malformado se informa, no falla", obj is None and "JSON válido" in motivo)
    obj, motivo = e.validar(json.dumps({"enunciado": "x", "tipo_integral": "rara"}))
    comprobar("fuera de esquema se informa, no falla", obj is None and "esquema" in motivo)
    obj, motivo = e.validar(bueno[:-1] + ', "extra": 1}')
    comprobar("campo inventado se rechaza", obj is None)

    print("2) Normalización y comparación")
    comprobar("LaTeX vs Unicode (enunciado)",
              e.normalizar(r"\int \frac{dx}{\sqrt{4x^2+1}}") == e.normalizar("∫ dx / √(4x²+1)"))
    comprobar("resultado con \\frac vs 1/2",
              e.normalizar(r"\frac{1}{2}\ln|\sqrt{4x^2+1}+2x|+C") == e.normalizar("(1/2) ln|√(4x²+1) + 2x| + C"))
    comprobar("resultado distinto no coincide", e.normalizar("x e^x + e^x + C") != e.normalizar("x e^x - e^x + C"))

    verdad = json.loads((c.IMAGENES / "verdad_terreno.json").read_text(encoding="utf-8"))
    real = verdad["ej2_partes.png"]
    exacto, _ = e.validar(json.dumps({k: real[k] for k in e.CAMPOS}))
    comprobar("comparar(): extracción idéntica = 5/5", sum(f["correcto"] for f in e.comparar(exacto, real)) == 5)
    peor, _ = e.validar(json.dumps({**{k: real[k] for k in e.CAMPOS}, "tecnica": "directa", "numero_pasos": 3}))
    comprobar("comparar(): 2 campos mal = 3/5", sum(f["correcto"] for f in e.comparar(peor, real)) == 3)

    print("3) WER")
    comprobar("WER idéntico = 0", metricas.wer("hola mundo", "Hola, mundo") == 0)
    comprobar("WER 1 de 4 palabras = 0.25", abs(metricas.wer("a b c d", "a b x d") - 0.25) < 1e-9)

    print("4) parte1_2 de principio a fin con cliente simulado")
    import parte1_2_extraccion as p12
    respuestas = [json.dumps({k: verdad[n][k] for k in e.CAMPOS}) for n in
                  [k for k in verdad if not k.startswith("_")]]
    respuestas[1] = "esto no es JSON"  # una respuesta inválida: el programa debe informar y seguir
    original = c.obtener_cliente
    c.obtener_cliente = lambda: cliente_falso(respuestas)
    c.RESULTADOS = Path(tempfile.mkdtemp())
    try:
        p12.main()
        datos = json.loads((c.RESULTADOS / "parte1_2.json").read_text(encoding="utf-8"))
        comprobar("1 imagen inválida + 2 perfectas = 10/15 campos", datos["aciertos"] == 10 and datos["total"] == 15)
        comprobar("se genera la tabla Markdown", (c.RESULTADOS / "parte1_2.md").exists())
    finally:
        c.obtener_cliente = original

    print("5) parte2 de principio a fin con audio y cliente simulados")
    import parte2_audio as p2
    carpeta = Path(tempfile.mkdtemp())
    with wave.open(str(carpeta / "a1.wav"), "wb") as w:
        w.setnchannels(1), w.setsampwidth(2), w.setframerate(8000)
        w.writeframes(b"\x00\x00" * 8000 * 12)  # 12 s de silencio
    (carpeta / "guion.json").write_text(json.dumps({"audios": [
        {"archivo": "a1.wav", "hablante": "A", "condicion": "sin ruido", "duracion_s": None, "dijo": "integral por partes"},
        {"archivo": "falta.m4a", "hablante": "B", "condicion": "con ruido", "duracion_s": None, "dijo": "x"}]}),
        encoding="utf-8")
    c.AUDIOS = carpeta
    c.obtener_cliente = lambda: cliente_falso(["Respuesta de prueba"], ["integral por partes"])
    import sys
    argv = sys.argv
    sys.argv = ["parte2_audio.py"]
    try:
        p2.main()
        datos = json.loads((c.RESULTADOS / "parte2.json").read_text(encoding="utf-8"))
        comprobar("procesa el audio existente y omite el que falta", len(datos["audios"]) == 1)
        comprobar("duración del wav = 12 s", datos["audios"][0]["duracion_s"] == 12.0)
        comprobar("WER 0 y hay respuesta", datos["audios"][0]["wer"] == 0 and datos["audios"][0]["respuesta"])
    finally:
        sys.argv = argv
        c.obtener_cliente = original
        shutil.rmtree(carpeta, ignore_errors=True)

    print(f"\nResultado: {n_ok} correctas, {n_mal} fallidas")
    raise SystemExit(1 if n_mal else 0)


if __name__ == "__main__":
    main()
