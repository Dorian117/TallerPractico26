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

# 4. Configurar la Persona y el Tono mediante system_instruction:
config_gerente = types.GenerateContentConfig(
    system_instruction=(
        "Eres un Gerente de Finanzas amable pero firme. Tu objetivo es "
        "recuperar pagos pendientes manteniendo una relación comercial "
        "cordial con el cliente, sin sonar agresivo ni amenazante."
    ),
    temperature=0.4,
)

# 5. Datos variables del cliente, separados del resto de la instrucción
#    mediante el delimitador ### para evitar ambigüedad con el modelo
datos_cliente = """
###
Nombre: Carlos Ramirez
Factura pendiente: FAC-2024-0891
Monto adeudado: $1.250.000 COP
Fecha de vencimiento: 15 de agosto de 2026
Dias de mora: 14
###
"""

# 6. Construir el prompt maestro: Tarea + Formato + Contexto delimitado
prompt = (
    "Redacta un correo de cobranza dirigido al cliente cuyos datos se "
    "entregan a continuacion, delimitados por ###.\n\n"
    "El correo debe:\n"
    "1. Recordar el saldo pendiente de forma respetuosa.\n"
    "2. Solicitar el pago dentro de los proximos 5 dias habiles.\n"
    "3. Ofrecer un canal de contacto para resolver dudas.\n"
    "4. Finalizar con un resumen en formato de tabla markdown con las "
    "columnas: Concepto, Fecha de vencimiento y Monto adeudado.\n\n"
    f"{datos_cliente}"
)

# 7. Realizar la peticion sincrona al modelo
response = client.models.generate_content(
    model=MODELO,
    contents=prompt,
    config=config_gerente,
)

# 8. Mostrar el resultado
print("Correo generado por el Gerente de Finanzas:\n")
print(response.text.strip())