import os
from pathlib import Path
from dotenv import load_dotenv
from google import genai
from google.genai import types

# 1. Cargar las variables de entorno desde el archivo .env
load_dotenv()

# 2. Inicializar el cliente oficial de Gemini
api_key = os.getenv("GEMINI_API_KEY")


client = genai.Client(api_key=api_key)
MODELO = "gemini-3-flash-preview"


def procesar_articulo(texto: str, tarea: str) -> str:
    tarea_normalizada = tarea.strip().lower()

    if tarea_normalizada == "resumir":
        instruccion_tarea = "Elabora un resumen ejecutivo claro, conciso y de alto impacto del siguiente texto:"
    elif tarea_normalizada == "profesionalizar":
        instruccion_tarea = "Edita y reescribe el siguiente texto para que tenga un estilo formal, técnico y de alto estándar editorial:"
    else:
        raise ValueError("Tarea no válida. Las opciones permitidas son 'resumir' o 'profesionalizar'.")

    # Configuración de la IA como Editor Editorial
    config = types.GenerateContentConfig(
        system_instruction=(
            "Eres un Editor Editorial de prestigio con amplia trayectoria en publicaciones "
            "académicas, científicas y corporativas de primer nivel. Tu lenguaje es impecable, "
            "riguroso y formal."
        ),
        temperature=0.3,
    )

    prompt = f"{instruccion_tarea}\n\nTexto original:\n{texto}"

    response = client.models.generate_content(
        model=MODELO,
        contents=prompt,
        config=config,
    )

    return response.text.strip()

    print("--- PROCESADOR DE TEXTOS INTELIGENTE ---")
texto_usuario = input("\nIngresa el texto o artículo que deseas procesar:\n> ")
tarea_usuario = input("\n¿Qué tarea deseas realizar? (escribe 'resumir' o 'profesionalizar'):\n> ")

try:
    # Llamada directa a la función con lo que escribió el usuario
    resultado = procesar_articulo(texto=texto_usuario, tarea=tarea_usuario)
    
    print("\n--- RESULTADO DE LA EDICIÓN ---")
    print(resultado)

except ValueError as error:
    print(f"\n[Error de entrada]: {error}")



       
