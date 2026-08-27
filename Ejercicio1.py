import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

# 1. Cargar las variables de entorno desde el archivo .env
load_dotenv()

# 2. Obtener y validar la API Key
API_KEY = os.getenv("GENAI_API_KEY")


# 3. Inicializar el cliente oficial de Gemini
client = genai.Client(api_key=API_KEY)

# 4. Definir el prompt con la restricción de longitud solicitada
prompt = (
    "Explica qué es la 'Inferencia en IA' en menos de 50 palabras de forma "
    "clara y directa."
)

# 5. Realizar la petición síncrona al modelo
response = client.models.generate_content(
    model="gemini-3-flash-preview",
    contents=prompt,
)

# 6. Mostrar el resultado
print("Respuesta de Gemini:\n")
print(response.text.strip())