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
#    Gerente de Finanzas amable pero firme
config_gerente = types.GenerateContentConfig(
    system_instruction=(
        "Eres un Gerente de Finanzas amable pero firme. Tu objetivo es "
        "recuperar pagos pendientes manteniendo una relacion comercial "
        "cordial con el cliente, sin sonar agresivo ni amenazante."
    ),
    temperature=0.4,
)


def solicitar_datos_cliente() -> dict:
    """
    5. Recolecta los datos del cliente por consola y repite la
       solicitud mediante un bucle while hasta que el usuario
       confirme que los datos son correctos.
    """
    datos_confirmados = False
    datos_cliente = {}

    while not datos_confirmados:
        print("\nPor favor ingresa los siguientes datos del cliente:\n")

        datos_cliente["nombre"] = input("Nombre del cliente: ").strip()
        datos_cliente["factura"] = input("Numero de factura: ").strip()
        datos_cliente["monto"] = input("Monto adeudado (ej. $1.250.000 COP): ").strip()
        datos_cliente["fecha_vencimiento"] = input("Fecha de vencimiento: ").strip()
        datos_cliente["dias_mora"] = input("Dias de mora: ").strip()

        # 6. Mostrar un resumen de lo ingresado antes de continuar
        print("\n--- RESUMEN DE LOS DATOS INGRESADOS ---")
        for clave, valor in datos_cliente.items():
            print(f"{clave.capitalize()}: {valor}")

        confirmacion = input(
            "\n¿Los datos son correctos y deseas continuar? (si/no): "
        ).strip().lower()

        # 7. El while solo se rompe cuando el usuario confirma con "si"
        if confirmacion == "si":
            datos_confirmados = True
        else:
            print("\nVolvamos a ingresar los datos.\n")

    return datos_cliente


# 8. Mensaje inicial de presentacion de la herramienta
print("=" * 70)
print(
    "Soy una herramienta basada en IA encargada de generar correos de "
    "cobranza para clientes con pagos pendientes.\n"
    "Indicame los siguientes datos para continuar mi tarea."
)
print("=" * 70)

# 9. Ejecutar la recoleccion de datos (bloquea hasta la confirmacion)
datos_cliente = solicitar_datos_cliente()

# 10. Construir el bloque de datos delimitado por ### con lo ingresado
#     por el usuario, para evitar ambiguedad con el modelo
bloque_datos = f"""
###
Nombre: {datos_cliente['nombre']}
Factura pendiente: {datos_cliente['factura']}
Monto adeudado: {datos_cliente['monto']}
Fecha de vencimiento: {datos_cliente['fecha_vencimiento']}
Dias de mora: {datos_cliente['dias_mora']}
###
"""

# 11. Construir el prompt maestro: Tarea + Formato + Contexto delimitado
prompt = (
    "Redacta un correo de cobranza dirigido al cliente cuyos datos se "
    "entregan a continuacion, delimitados por ###.\n\n"
    "El correo debe:\n"
    "1. Recordar el saldo pendiente de forma respetuosa.\n"
    "2. Solicitar el pago dentro de los proximos 5 dias habiles.\n"
    "3. Ofrecer un canal de contacto para resolver dudas.\n"
    "4. Finalizar con un resumen en formato de tabla markdown con las "
    "columnas: Concepto, Fecha de vencimiento y Monto adeudado.\n\n"
    f"{bloque_datos}"
)

# 12. Realizar la peticion sincrona al modelo
response = client.models.generate_content(
    model=MODELO,
    contents=prompt,
    config=config_gerente,
)

# 13. Mostrar el resultado
print("\nCorreo generado por el Gerente de Finanzas:\n")
print(response.text.strip())