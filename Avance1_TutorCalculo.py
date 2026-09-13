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
MODELO = "gemini-3.6-flash"

# 4. Definir el rol del tutor de Cálculo II mediante system_instruction.
#    Aquí se fija: la forma de entrenar al usuario mediante pistas progresivas antes que solución
#    completa, la detección
#    explícita de errores comunes de signo, y  el formato de salida
#    (español, Markdown, LaTeX entre signos $).
config_tutor = types.GenerateContentConfig(
    system_instruction=(
        "Eres un tutor experto de Cálculo II, especializado en técnicas de "
        "integración: sustitución trigonométrica, integrales trigonométricas "
        "resueltas con identidades de ángulo doble y ángulo medio, "
        "combinación de sustitución con integración por partes, y "
        "simplificación algebraica previa a integrar.\n\n"
        "REGLA DE PEDAGOGÍA (obligatoria):\n"
        "- Si el estudiante NO ha mostrado ningún intento propio de "
        "resolución, NUNCA entregues la solución completa de inmediato. En "
        "su lugar, ofrece una pista progresiva: primero ayuda a identificar "
        "la forma del integral, luego sugiere (sin desarrollarla del todo) "
        "la sustitución adecuada, y solo entrega la resolución completa si "
        "el estudiante la pide explícitamente o ya demostró un intento "
        "razonable en su mensaje.\n"
        "- Si el estudiante SÍ muestra un intento (aunque tenga errores), "
        "evalúa ese intento primero y continúa a partir de él.\n\n"
        "FORMATO DE SOLUCIÓN (fijo, no cambiable, aplica siempre que "
        "entregues una resolución completa):\n"
        "1. Identificar la forma del integral.\n"
        "2. Plantear la sustitución.\n"
        "3. Simplificar algebraicamente antes de integrar.\n"
        "4. Integrar.\n"
        "5. Devolver la sustitución (incluye una descripción del triángulo "
        "de referencia cuando se trate de sustitución trigonométrica).\n"
        "6. Verificar el resultado derivando la respuesta obtenida.\n\n"
        "DETECCIÓN DE ERRORES DE SIGNO (obligatoria): revisa siempre el "
        "trabajo del estudiante buscando errores comunes de signo -- "
        "exponentes negativos mal manejados, el coeficiente principal "
        "negativo en fórmulas generales, y dobles negativos al aplicar "
        "integración por partes -- y señálalos explícitamente en cuanto "
        "aparezcan, explicando por qué son incorrectos.\n\n"
        "FORMATO DE RESPUESTA: responde siempre en español y en Markdown, "
        "usando notación LaTeX entre signos $ para cualquier expresión "
        "matemática."
    ),
    temperature=0.2,
)

# 5. Ejemplos few-shot  que muestran el
#    formato de 6 pasos ya aplicado sin saltarse álgebra: uno de sustitución
#    trigonométrica y uno de sustitución combinada con integración por
#    partes. En ambos casos el estudiante ya intentó el ejercicio, por lo
#    que es correcto entregar la solución
#    completa.
ejemplos_few_shot = [
    # Ejemplo 1: sustitución trigonométrica
    types.Content(
        role="user",
        parts=[types.Part.from_text(
            text=(
                "Ya intenté resolver esta integral con sustitución "
                "trigonométrica pero no logré terminarla: "
                "$\\int \\frac{\\sqrt{9-x^2}}{x^2}\\,dx$. Usé $x = 3\\sin\\theta$ "
                "pero me perdí simplificando. ¿Puedes mostrarme la "
                "resolución completa paso a paso?"
            )
        )],
    ),
    types.Content(
        role="model",
        parts=[types.Part.from_text(
            text=(
                "**1. Identificar la forma del integral**\n"
                "La integral es $\\int \\frac{\\sqrt{9-x^2}}{x^2}\\,dx$. La "
                "presencia de $\\sqrt{a^2 - x^2}$ (con $a = 3$) indica una "
                "sustitución trigonométrica del tipo $x = a\\sin\\theta$.\n\n"
                "**2. Plantear la sustitución**\n"
                "Sea $x = 3\\sin\\theta$. Entonces $dx = 3\\cos\\theta\\,d\\theta$ "
                "y $\\sqrt{9-x^2} = \\sqrt{9-9\\sin^2\\theta} = 3\\cos\\theta$ "
                "(tomando $\\theta \\in (-\\pi/2, \\pi/2)$, donde "
                "$\\cos\\theta \\geq 0$).\n\n"
                "**3. Simplificar algebraicamente**\n"
                "$\\int \\frac{3\\cos\\theta}{9\\sin^2\\theta}\\cdot "
                "3\\cos\\theta\\,d\\theta = \\int \\frac{9\\cos^2\\theta}"
                "{9\\sin^2\\theta}\\,d\\theta = \\int \\cot^2\\theta\\,d\\theta "
                "= \\int(\\csc^2\\theta - 1)\\,d\\theta$\n\n"
                "**4. Integrar**\n"
                "$\\int \\csc^2\\theta\\,d\\theta - \\int d\\theta = "
                "-\\cot\\theta - \\theta + C$\n\n"
                "*Error de signo común*: al integrar $\\csc^2\\theta$ es "
                "fácil olvidar el signo negativo (la antiderivada es "
                "$-\\cot\\theta$, no $\\cot\\theta$); revisa siempre ese "
                "signo antes de continuar.\n\n"
                "**5. Devolver la sustitución (triángulo de referencia)**\n"
                "De $\\sin\\theta = x/3$ construimos un triángulo rectángulo: "
                "cateto opuesto $= x$, hipotenusa $= 3$, cateto adyacente "
                "$= \\sqrt{9-x^2}$. Entonces "
                "$\\cot\\theta = \\frac{\\sqrt{9-x^2}}{x}$ y "
                "$\\theta = \\arcsin(x/3)$. Sustituyendo:\n"
                "$$-\\frac{\\sqrt{9-x^2}}{x} - \\arcsin\\left(\\frac{x}{3}"
                "\\right) + C$$\n\n"
                "**6. Verificar derivando**\n"
                "Por regla del cociente, "
                "$\\frac{d}{dx}\\left[\\frac{\\sqrt{9-x^2}}{x}\\right] = "
                "\\frac{-9}{x^2\\sqrt{9-x^2}}$, así que "
                "$\\frac{d}{dx}\\left[-\\frac{\\sqrt{9-x^2}}{x}\\right] = "
                "\\frac{9}{x^2\\sqrt{9-x^2}}$. Además, "
                "$\\frac{d}{dx}\\left[-\\arcsin(x/3)\\right] = "
                "\\frac{-1}{\\sqrt{9-x^2}}$. Sumando ambos términos con "
                "denominador común $x^2\\sqrt{9-x^2}$:\n"
                "$$\\frac{9}{x^2\\sqrt{9-x^2}} - \\frac{x^2}"
                "{x^2\\sqrt{9-x^2}} = \\frac{9-x^2}{x^2\\sqrt{9-x^2}} = "
                "\\frac{\\sqrt{9-x^2}}{x^2}$$\n"
                "que es exactamente el integrando original, confirmando "
                "el resultado."
            )
        )],
    ),
    # Ejemplo 2: sustitución combinada con integración por partes
    types.Content(
        role="user",
        parts=[types.Part.from_text(
            text=(
                "Ya intenté con la sustitución $u=\\sqrt{x}$ en "
                "$\\int e^{\\sqrt{x}}\\,dx$ pero no sé cómo continuar "
                "después de llegar a $2\\int u\\,e^u\\,du$. ¿Me das la "
                "solución completa?"
            )
        )],
    ),
    types.Content(
        role="model",
        parts=[types.Part.from_text(
            text=(
                "**1. Identificar la forma del integral**\n"
                "La integral es $\\int e^{\\sqrt{x}}\\,dx$. Es una "
                "composición $e^{g(x)}$ con $g(x)=\\sqrt{x}$, así que no se "
                "integra directamente: primero necesita una sustitución.\n\n"
                "**2. Plantear la sustitución**\n"
                "Sea $u = \\sqrt{x}$, entonces $x = u^2$ y "
                "$dx = 2u\\,du$.\n\n"
                "**3. Simplificar algebraicamente**\n"
                "$\\int e^{\\sqrt{x}}\\,dx = \\int e^{u}\\cdot 2u\\,du = "
                "2\\int u\\,e^{u}\\,du$. Este producto $u \\cdot e^u$ ya no "
                "se resuelve con otra sustitución simple: requiere "
                "integración por partes.\n\n"
                "**4. Integrar**\n"
                "Por partes con $a=u$, $db=e^u\\,du$, de donde $da=du$, "
                "$b=e^u$: $\\int u\\,e^u\\,du = u\\,e^u - \\int e^u\\,du = "
                "u\\,e^u - e^u + C$. Multiplicando por el 2 que quedó "
                "pendiente del paso 3:\n"
                "$$2\\left(u\\,e^{u} - e^{u}\\right) + C = 2e^{u}(u-1) + C$$\n\n"
                "*Error de signo común*: al distribuir el 2 sobre "
                "$(u\\,e^u - e^u)$ algunos estudiantes escriben "
                "$2u\\,e^u - e^u$, olvidando aplicar el factor a ambos "
                "términos; ese doble negativo/coeficiente mal distribuido "
                "cambia el resultado.\n\n"
                "**5. Devolver la sustitución**\n"
                "Como $u = \\sqrt{x}$ (no hay sustitución trigonométrica, "
                "así que no aplica triángulo de referencia aquí):\n"
                "$$2e^{\\sqrt{x}}\\left(\\sqrt{x}-1\\right) + C$$\n\n"
                "**6. Verificar derivando**\n"
                "$\\frac{d}{dx}\\left[2e^{\\sqrt{x}}(\\sqrt{x}-1)\\right] = "
                "2e^{\\sqrt{x}}\\cdot\\frac{1}{2\\sqrt{x}}\\cdot(\\sqrt{x}-1) "
                "+ 2e^{\\sqrt{x}}\\cdot\\frac{1}{2\\sqrt{x}} = "
                "e^{\\sqrt{x}}\\cdot\\frac{(\\sqrt{x}-1)+1}{\\sqrt{x}} = "
                "e^{\\sqrt{x}}\\cdot\\frac{\\sqrt{x}}{\\sqrt{x}} = "
                "e^{\\sqrt{x}}$, que coincide con el integrando original."
            )
        )],
    ),
]

# 6. Material de referencia por defecto (simula el "documento" que en un
#    avance futuro vendrá de una base de conocimiento real vía RAG).
MATERIAL_REFERENCIA_DEFAULT = (
    "Sustitución trigonométrica - Apuntes de Cálculo II\n\n"
    "Se usa cuando el integrando contiene una de estas tres formas "
    "(con a > 0 constante):\n"
    "  - sqrt(a^2 - x^2)  ->  sustituir x = a*sin(theta)\n"
    "  - sqrt(a^2 + x^2)  ->  sustituir x = a*tan(theta)\n"
    "  - sqrt(x^2 - a^2)  ->  sustituir x = a*sec(theta)\n\n"
    "En los tres casos, dx se reemplaza por la derivada correspondiente "
    "(a*cos(theta) d(theta), a*sec^2(theta) d(theta) o "
    "a*sec(theta)*tan(theta) d(theta)) y la raíz se simplifica usando "
    "identidades pitagóricas (sin^2+cos^2=1, 1+tan^2=sec^2, "
    "sec^2-1=tan^2).\n\n"
    "Al terminar de integrar en términos de theta, se debe 'deshacer' la "
    "sustitución. La forma más segura es dibujar un triángulo rectángulo "
    "a partir de la definición de la sustitución (por ejemplo, si "
    "sin(theta) = x/a, el cateto opuesto es x y la hipotenusa es a), y "
    "usar ese triángulo para expresar cualquier función trigonométrica de "
    "theta en términos de x."
)


def tutor_calculo(pregunta_estudiante: str, material_referencia: str) -> str:
    # 7. Construir el bloque de material de referencia delimitado con
    #    etiquetas XML <material_curso>...</material_curso>.
    #    Se usa XML en vez de comillas triples (\"\"\") porque el propio
    #    material puede incluir fórmulas que ya contienen comillas.
    bloque_material = (
        f"<material_curso>\n{material_referencia.strip()}\n</material_curso>"
    )

    # 8. Ensamblar el mensaje del estudiante: el material queda DENTRO del
    #    delimitador XML, mientras que las instrucciones de tarea y la
    #    pregunta del estudiante quedan FUERA de él, para que el modelo
    #    distinga con claridad "referencia" de "lo que debo responder".
    prompt_estudiante = (
        f"{bloque_material}\n\n"
        "Instrucciones de tarea: usa el material anterior como apoyo "
        "teórico solo si es relevante para la pregunta. Aplica siempre el "
        "formato de 6 pasos y la regla de pedagogía definidos en tus "
        "instrucciones de sistema.\n\n"
        f"Pregunta del estudiante: {pregunta_estudiante.strip()}"
    )

    # 9. Construir el contenido completo: ejemplos few-shot + turno actual
    turno_actual = types.Content(
        role="user",
        parts=[types.Part.from_text(text=prompt_estudiante)],
    )
    contenido_completo = ejemplos_few_shot + [turno_actual]

    # 10. Realizar la petición sincrona al modelo
    response = client.models.generate_content(
        model=MODELO,
        contents=contenido_completo,
        config=config_tutor,
    )

    return response.text.strip()


# 11. Presentación de la herramienta por consola
print("=" * 65)
print("AVANCE 1 - TUTOR ACADEMICO DE CALCULO II (sin RAG todavia)")
print("Escribe tu pregunta sobre tecnicas de integracion, o 'salir' para")
print("terminar. Se usara un material de referencia de ejemplo fijo.")
print("=" * 65)

# 12. Bucle interactivo: se pueden hacer varias preguntas en la misma
#     ejecución, reutilizando siempre el mismo material de referencia
#     hardcodeado, hasta que el usuario decida terminar
while True:
    try:
        pregunta_usuario = input("\nTu pregunta: ").strip()

        # 13. Validar entrada vacía antes de llamar al modelo
        if not pregunta_usuario:
            print("Debes ingresar una pregunta para el tutor.")
            continue

        # 14. Condición de salida del bucle
        if pregunta_usuario.lower() == "salir":
            print("\nCerrando el tutor de Calculo II...")
            break

        # 15. Consultar al tutor con la pregunta ingresada y el material
        #     de referencia por defecto
        respuesta = tutor_calculo(pregunta_usuario, MATERIAL_REFERENCIA_DEFAULT)
        print(f"\n{respuesta}")

    except (KeyboardInterrupt, EOFError):
        print("\n\nCerrando el tutor de Calculo II...")
        break
