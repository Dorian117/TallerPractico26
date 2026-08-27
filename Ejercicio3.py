import os
from pathlib import Path
from dotenv import load_dotenv
from google import genai
from google.genai import types

# 1. Cargar las credenciales de entorno
load_dotenv()

# 2. Inicializar el cliente oficial de Gemini
api_key = os.getenv("GEMINI_API_KEY")

client = genai.Client(api_key=api_key)
MODELO = "gemini-3.5-flash-lite"

# 3. Configurar el rol de vendedor amable mediante system_instruction
config_vendedor = types.GenerateContentConfig(
    system_instruction=(
        "Eres un asesor de ventas amable, cordial y muy servicial en una tienda especializada "
        "en tecnología y componentes de computación. Tu objetivo es orientar a los clientes con "
        "entusiasmo, detallando especificaciones técnicas de forma clara y comprensible, y recomendando "
        "la mejor opción de compra."
    ),
    temperature=0.4,
)

# 4. Contexto Few-Shot: Pre-cargar el historial con 2 ejemplos (user -> model)
historial_few_shot = [
    # Ejemplo 1
    types.Content(
        role="user",
        parts=[types.Part.from_text(text="Hola, ¿qué características tiene la tarjeta gráfica AMD Radeon RX 7600?")]
    ),
    types.Content(
        role="model",
        parts=[types.Part.from_text(
            text=(
                "¡Hola! Con mucho gusto te cuento. La AMD Radeon RX 7600 cuenta con 8 GB de memoria GDDR6, "
                "interfaz de 128 bits, 2048 Stream Processors y soporte para PCIe 4.0. Es una opción excelente "
                "para jugar en 1080p en calidad ultra con gran rendimiento térmico y bajo consumo (165W). "
                "¿Te gustaría que revisemos si es compatible con tu fuente de poder actual?"
            )
        )]
    ),
    # Ejemplo 2
    types.Content(
        role="user",
        parts=[types.Part.from_text(text="¿Qué procesador me recomiendas para diseño y juegos sin gastar demasiado?")]
    ),
    types.Content(
        role="model",
        parts=[types.Part.from_text(
            text=(
                "¡Un gusto saludarte! Te recomiendo ampliamente el AMD Ryzen 5 5600X. Tiene 6 núcleos y 12 hilos, "
                "frecuencia base de 3.7 GHz (hasta 4.6 GHz Turbo), 32 MB de caché L3 y solo 65W de TDP en socket AM4. "
                "Maneja multitarea pesada, edición y juegos con mucha soltura a un precio muy equilibrado. "
                "¿Tienes ya una tarjeta madre AM4 o necesitas que armemos el combo completo?"
            )
        )]
    ),
]

# 5. Crear la sesión de chat con el historial pre-cargado
chat = client.chats.create(
    model=MODELO,
    history=historial_few_shot,
    config=config_vendedor,
)

# 6. Bucle de conversación interactiva
print("=" * 65)
print("🏪 ASISTENTE DE VENTAS TECH - TIENDA DE TECNOLOGÍA")
print("Escribe tu consulta o 'finalizar' para terminar.")
print("=" * 65 + "\n")

while True:
    try:
        mensaje_usuario = input("Cliente: ").strip()

        # Validar entrada vacía
        if not mensaje_usuario:
            continue

        # Condición de salida
        if mensaje_usuario.lower() == "finalizar":
            print("\nVendedor: ¡Muchas gracias por tu visita! Esperamos verte pronto de nuevo. ¡Que tengas un gran día!")
            break

        # Enviar mensaje al chat (el SDK mantiene y acumula el historial automáticamente)
        respuesta = chat.send_message(mensaje_usuario)
        print(f"\nVendedor: {respuesta.text.strip()}\n")

    except (KeyboardInterrupt, EOFError):
        print("\n\nCerrando sesión de chat...")
        break