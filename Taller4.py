import os
import json
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

# 4. Configurar el rol del evaluador academico. Se fuerza la salida a
#    JSON puro mediante response_mime_type, evitando texto adicional
#    o bloques de markdown que rompan el parseo posterior.
config_evaluador = types.GenerateContentConfig(
    system_instruction=(
        "Eres un evaluador academico experto en retroalimentacion de "
        "ensayos universitarios. Evalua el ensayo recibido bajo tres "
        "criterios: Ortografia, Coherencia y Argumentacion, cada uno "
        "sobre 5 puntos. Calcula 'nota_final' como el promedio de los "
        "tres criterios. Responde UNICAMENTE con un objeto JSON valido "
        "con la forma exacta: "
        '{"nota_final": <numero>, "comentarios": {"ortografia": "<texto>", '
        '"coherencia": "<texto>", "argumentacion": "<texto>"}}'
    ),
    temperature=0.3,
    response_mime_type="application/json",
)


def leer_ensayo_por_consola() -> str:
    """
    5. Lee el texto del ensayo por consola linea a linea, ya que un
       ensayo de mas de 100 palabras no cabe comodamente en un solo
       input(). La lectura termina cuando el usuario escribe FIN
       en una linea sola.
    """
    print("\nEscribe o pega el texto del ensayo.")
    print("Cuando termines, escribe FIN en una linea nueva y presiona Enter.\n")

    lineas_ensayo = []
    while True:
        linea = input()
        if linea.strip().upper() == "FIN":
            break
        lineas_ensayo.append(linea)

    return "\n".join(lineas_ensayo).strip()


def evaluar_ensayo(texto: str) -> dict:
    # 6. Condicional en Python: valida la longitud ANTES de llamar al
    #    modelo, evitando gastar una peticion en un ensayo insuficiente
    cantidad_palabras = len(texto.split())

    if cantidad_palabras < 100:
        return {
            "nota_final": None,
            "comentarios": (
                f"El ensayo es demasiado corto para evaluar "
                f"({cantidad_palabras} palabras). Se requieren al menos "
                f"100 palabras."
            ),
        }

    # 7. Si cumple la longitud minima, se envia al modelo para evaluacion,
    #    delimitando el ensayo con comillas triples
    prompt = f'"""\n{texto}\n"""'

    response = client.models.generate_content(
        model=MODELO,
        contents=prompt,
        config=config_evaluador,
    )

    # 8. Parsear la respuesta JSON entregada por el modelo
    return json.loads(response.text.strip())


# 9. Presentacion de la herramienta por consola
print("=" * 65)
print("EVALUADOR ACADEMICO DE ENSAYOS")
print("=" * 65)

# 10. Bucle interactivo: se puede evaluar mas de un ensayo en la misma
#     ejecucion, hasta que el usuario decida terminar
while True:
    try:
        ensayo_usuario = leer_ensayo_por_consola()

        # 11. Validar entrada vacia antes de evaluar
        if not ensayo_usuario:
            print("No ingresaste ningun texto.")
        else:
            resultado = evaluar_ensayo(ensayo_usuario)
            print("\n--- RESULTADO DE LA EVALUACION ---")
            print(json.dumps(resultado, indent=2, ensure_ascii=False))

        continuar = input(
            "\n¿Deseas evaluar otro ensayo? (si/no): "
        ).strip().lower()

        # 12. Condicion de salida del bucle
        if continuar != "si":
            print("\nCerrando el evaluador academico...")
            break

    except json.JSONDecodeError as error:
        print(f"\n[Error al parsear JSON]: {error}")

    except (KeyboardInterrupt, EOFError):
        print("\n\nCerrando el evaluador academico...")
        break