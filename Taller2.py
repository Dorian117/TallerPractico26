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

# 4. Configurar el rol y la logica condicional (Si-Entonces) mediante
#    system_instruction. El propio modelo decide la rama a ejecutar
#    basandose unicamente en el contenido del texto recibido.
config_triaje = types.GenerateContentConfig(
    system_instruction=(
        'Eres un asistente de triaje de correos electronicos de soporte. '
        'Se te proporcionara un texto delimitado por triple comillas """.\n\n'
        "Sigue esta logica de forma estricta y evalua las condiciones en "
        "orden, deteniendote en la primera que se cumpla:\n"
        "1. SI el texto contiene una queja sobre un pago o factura:\n"
        "   - Clasificalo como 'URGENTE-FINANZAS'.\n"
        "   - Extrae el numero de factura si existe.\n"
        "2. SI NO, si el texto es una duda tecnica general:\n"
        "   - Clasificalo como 'SOPORTE-ESTANDAR'.\n"
        "   - Responde: 'Gracias, un tecnico lo revisara'.\n"
        "3. SI NO es ninguna de las anteriores:\n"
        "   - Responde simplemente: 'Categoria no identificada'.\n\n"
        "No expliques tu razonamiento, entrega solo el resultado final."
    ),
    temperature=0.0,
)


def clasificar_correo(texto: str) -> str:
    # 5. Delimitar el texto de entrada con comillas triples, tal como
    #    lo espera la logica descrita en el system_instruction
    prompt = f'"""\n{texto.strip()}\n"""'

    # 6. Realizar la peticion sincrona al modelo
    response = client.models.generate_content(
        model=MODELO,
        contents=prompt,
        config=config_triaje,
    )

    return response.text.strip()


# 7. Texto del usuario a clasificar (caso del taller: queja de factura)
texto_usuario = (
    "Hola, mi factura #4502 tiene un cargo doble que no reconozco. Ayuda."
)

# 8. Ejecutar la clasificacion y mostrar el resultado
resultado = clasificar_correo(texto_usuario)
print("--- FILTRO DE SOPORTE TECNICO ---\n")
print(resultado)