# Taller: entrada multimodal y audio (dominio: Tutor de Cálculo II)

Visión sobre imágenes de ejercicios de integrales y transcripción de audio con preguntas de Cálculo II.
Todo usa **Groq** con una sola clave. Ejecutar desde esta carpeta (`taller_multimodal/`).

## Instalación

```powershell
# Windows (PowerShell)
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env      # y completa GROQ_API_KEY
```

```bash
# Linux / macOS
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && cp .env.example .env
```

## Variables de entorno

| Variable | Obligatoria | Descripción |
|---|---|---|
| `GROQ_API_KEY` | **Sí** | Clave gratuita de <https://console.groq.com/keys>. Va solo en `.env` (ignorado por git) |
| `GROQ_VISION_MODEL` | No | Modelo de visión (por defecto `qwen/qwen3.8-27b`) |
| `GROQ_AUDIO_MODEL` | No | Modelo de transcripción (por defecto `whisper-large-v3-turbo`) |
| `GROQ_TEXT_MODEL` | No | Modelo que responde en 2.2 (por defecto, el de visión) |

Los modelos y sus límites (3 imágenes y 20 MB por solicitud; audio de hasta 25 MB) conviene
revisarlos en la documentación de Groq: cambian con el tiempo.

## Cómo ejecutar cada parte

| Punto | Comando | Qué hace | Salida |
|---|---|---|---|
| 1.1 | `python parte1_1_pregunta.py` | Pregunta sobre una imagen (base64) | `resultados/parte1_1.md` |
| 1.2 | `python parte1_2_extraccion.py` | Extrae 5 campos de 3 imágenes en JSON, valida el esquema y calcula el % de aciertos | `resultados/parte1_2.md` |
| 1.3 | `python parte1_3_varias.py` | Compara 2 imágenes en una solicitud | `resultados/parte1_3.md` |
| 1.4 | `python parte1_4_error.py` | Provoca 3 errores y registra el mensaje de la API | `resultados/parte1_4.md` |
| 2.1 y 2.2 | `python parte2_audio.py` | Audio → transcripción (Whisper) → respuesta del modelo; calcula el WER | `resultados/parte2.md` |

Cada script guarda además un `.json` con todo el detalle. Las tablas `.md` están listas para
pasarlas al informe. Opciones útiles: `parte2_audio.py --sin-prompt` (sin vocabulario técnico, para
comparar), `--solo-transcribir`, `--modelo whisper-large-v3`.

Sin conexión ni clave se pueden probar la validación del JSON y el resto de la lógica:
`python test_local.py`.

## Contenido

```
imagenes/         5 imágenes del dominio + verdad_terreno.json (valores reales para 1.2)
audios/           guion.json con lo que hay que decir en cada audio + los 4 audios grabados
resultados/       salidas de cada ejecución (json y md)
esquema.py        esquema de extracción (pydantic), validación y comparación
common.py         cliente, codificación base64 de imágenes, utilidades
metricas.py       WER para evaluar transcripciones
generar_imagenes.py   (opcional) regenera las imágenes de ejemplo
```

## Esquema de extracción (1.2)

`enunciado`, `tipo_integral` (definida | indefinida), `tecnica` (sustitucion | partes |
sustitucion_trigonometrica | fracciones_parciales | directa | otra), `resultado_final`, `numero_pasos`.
Si el modelo responde un JSON que no cumple el esquema, el programa lo informa y sigue con la
siguiente imagen (cuenta como 0 aciertos).

## Seguridad

La clave nunca está en el código: se lee de `GROQ_API_KEY`. `.env` está en el `.gitignore` de la raíz.
Antes de subir, comprueba: `git check-ignore -v .env` y `git log -p | findstr /i "gsk_"` (no debe aparecer nada).
