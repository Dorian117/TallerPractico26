import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

# 1. Cargar las variables de entorno desde el archivo .env
load_dotenv()

# 2. Obtener y validar la API Key
api_key = os.getenv("GEMINI_API_KEY")

# 3. Inicializar el cliente oficial de Gemini
client = genai.Client(api_key=api_key)
MODELO = "gemini-3-flash-preview"

# 4. Configurar el rol del clasificador: respuesta de una sola palabra
config_clasificador = types.GenerateContentConfig(
    system_instruction=(
        "Clasifica el sentimiento de resenas de libros. Responde "
        "UNICAMENTE con una palabra: POSITIVO, NEUTRAL o NEGATIVO. "
        "No agregues explicaciones ni puntuacion adicional."
    ),
    temperature=0.0,
)

# 5. Few-Shot: 3 ejemplos (shots) de entrada -> salida esperada,
#    construidos como turnos de conversacion user -> model. Este
#    historial se reutiliza en cada clasificacion del bucle.
ejemplos_few_shot = [
    types.Content(
        role="user",
        parts=[types.Part.from_text(
            text="Una obra maestra, no pude soltarlo ni un segundo, el "
                 "desarrollo de los personajes es brillante."
        )],
    ),
    types.Content(
        role="model",
        parts=[types.Part.from_text(text="POSITIVO")],
    ),
    types.Content(
        role="user",
        parts=[types.Part.from_text(
            text="Esta bien escrito pero la trama es generica, ni me "
                 "encanto ni me decepciono."
        )],
    ),
    types.Content(
        role="model",
        parts=[types.Part.from_text(text="NEUTRAL")],
    ),
    types.Content(
        role="user",
        parts=[types.Part.from_text(
            text="Perdi el tiempo. Los dialogos son forzados y el ritmo "
                 "es insoportablemente lento."
        )],
    ),
    types.Content(
        role="model",
        parts=[types.Part.from_text(text="NEGATIVO")],
    ),
]


def clasificar_resena(texto_resena: str) -> str:
    # 6. Construir el turno del usuario con la resena a evaluar
    turno_resena = types.Content(
        role="user",
        parts=[types.Part.from_text(text=texto_resena)],
    )

    # 7. Contenido completo enviado al modelo: ejemplos + resena nueva
    contenido_completo = ejemplos_few_shot + [turno_resena]

    # 8. Realizar la peticion sincrona al modelo
    response = client.models.generate_content(
        model=MODELO,
        contents=contenido_completo,
        config=config_clasificador,
    )

    return response.text.strip()


# 9. Presentacion de la herramienta por consola
print("=" * 65)
print("CLASIFICADOR DE SENTIMIENTOS - RESENAS DE LIBROS")
print("Escribe la resena a clasificar, o 'salir' para terminar.")
print("=" * 65)

# 10. Bucle interactivo: se ingresa la resena por consola hasta que
#     el usuario decida terminar
while True:
    try:
        resena_usuario = input("\nResena del libro: ").strip()

        # 11. Validar entrada vacia antes de llamar al modelo
        if not resena_usuario:
            print("Debes ingresar una resena para clasificar.")
            continue

        # 12. Condicion de salida del bucle
        if resena_usuario.lower() == "salir":
            print("\nCerrando el clasificador de sentimientos...")
            break

        # 13. Clasificar la resena ingresada y mostrar el resultado
        sentimiento = clasificar_resena(resena_usuario)
        print(f"\nSentimiento clasificado: {sentimiento}")

    except (KeyboardInterrupt, EOFError):
        print("\n\nCerrando el clasificador de sentimientos...")
        break
    