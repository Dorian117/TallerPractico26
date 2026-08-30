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
#    construidos como turnos de conversacion user -> model
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

# 6. Resena a clasificar (turno final del usuario, la que pide el taller)
resena_a_evaluar = types.Content(
    role="user",
    parts=[types.Part.from_text(
        text="Este libro empezo bien pero el final fue muy apresurado y "
             "decepcionante."
    )],
)

# 7. Construir el contenido completo: ejemplos + resena nueva
contenido_completo = ejemplos_few_shot + [resena_a_evaluar]

# 8. Realizar la peticion sincrona al modelo
response = client.models.generate_content(
    model=MODELO,
    contents=contenido_completo,
    config=config_clasificador,
)

# 9. Mostrar el resultado
print("Sentimiento clasificado:", response.text.strip())