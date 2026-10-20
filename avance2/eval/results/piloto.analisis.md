# Análisis de la corrida `piloto`

Tutor: `claude-haiku-5-5` · juez: `claude-haiku-5-5` (juez final: False) · 3 preguntas · parámetros: `{"chunk_tamano": 1800, "chunk_solape": 200, "top_k": 4, "umbral": null, "embeddings": "intfloat/multilingual-e5-small", "texto_embedding": "limpio", "max_tokens_respuesta": 8000}`

## Métricas de Ragas (preguntas dentro del corpus)

| Métrica | Media | IC 95 % (bootstrap) | Desv. est. | Mínimo | n | sin puntaje |
|---|---|---|---|---|---|---|
| Faithfulness | 0.833 | [0.824, 0.842] | 0.009 | 0.824 | 3 | 0 |
| Answer relevancy | 0.962 | [0.935, 0.991] | 0.028 | 0.935 | 3 | 0 |
| Context precision | 0.833 | [0.583, 1.000] | 0.220 | 0.583 | 3 | 0 |
| Context recall | 0.889 | [0.667, 1.000] | 0.192 | 0.667 | 3 | 0 |

Métrica más baja: **Faithfulness** (0.833). Con ~18 preguntas y un LLM como juez los intervalos son anchos: dos métricas (o dos corridas) solo difieren con seguridad si sus IC no se solapan.

### Por tipo de pregunta

| Tipo | n | Faithfulness | Answer relevancy | Context precision | Context recall |
|---|---|---|---|---|---|
| definicion | 3 | 0.83 | 0.96 | 0.83 | 0.89 |

### Evidencia: las preguntas con menor `faithfulness`

- **faithfulness = 0.824** · tipo `definicion` · «¿Cuál es la regla de la potencia para integrar x^n y qué condición debe cumplir n?»
  - archivo esperado en el top-4: **sí** (recuperó: c_digo_latex_de_antide…a de Antiderivadas Generales d=0.0887; c_digo_latex_de_antide…eglas Básicas de Integración d=0.1335; c_digo_latex_de_integr…Método Tabular d=0.1381; c_digo_latex_de_simpli…o Paso a Paso > Ejercicio c) d=0.1383)
  - otras métricas: faithfulness=0.82, answer_relevancy=0.99, context_precision=0.92, context_recall=0.67
  - citas inválidas: [] · respuesta (inicio): «La regla de la potencia está en los fragmentos [1] y [2]:  $$\int x^{n}\,dx = \frac{x^{n+1}}{n+1} + C$$  Para integrar una potencia de $x$, se suma una unidad al exponente y se divide por el nuevo exponente [2].  **Condición:** la fórmula exige $n \neq -1$ [1]…»
- **faithfulness = 0.833** · tipo `definicion` · «¿En qué consiste el método de sustitución y cuándo se puede aplicar?»
  - archivo esperado en el top-4: **sí** (recuperó: c_digo_latex_de_sustit…Introduccion d=0.1422; c_digo_latex_de_sustit…alizar un Cambio de Variable d=0.1563; c_digo_latex_de_sustit…ión de una Función Compuesta d=0.1564; c_digo_latex_de_sustic…Introduccion d=0.1599)
  - otras métricas: faithfulness=0.83, answer_relevancy=0.94, context_precision=0.58, context_recall=1.00
  - citas inválidas: [] · respuesta (inicio): «El método de sustitución consiste en cambiar la variable de integración por una función de $x$ para transformar la integral en una más sencilla. Según el material, se apoya en la regla de la cadena a la inversa [3].  **Idea central**  Si $u = g(x)$, entonces $…»
- **faithfulness = 0.842** · tipo `definicion` · «¿Qué es una antiderivada (o primitiva) de una función y por qué aparece la constante C?»
  - archivo esperado en el top-4: **sí** (recuperó: c_digo_latex_de_antide…Actividad Inicial d=0.1023; c_digo_latex_de_antide…Actividad Inicial d=0.1111; c_digo_latex_de_sustit…ión de una Función Compuesta d=0.1201; c_digo_latex_de_antide…a Antiderivación: Cinemática d=0.1235)
  - otras métricas: faithfulness=0.84, answer_relevancy=0.96, context_precision=1.00, context_recall=1.00
  - citas inválidas: [] · respuesta (inicio): «Una **antiderivada** (o **primitiva**) de $f(x)$ sobre un intervalo $I$ es una función $F(x)$ tal que $F'(x) = f(x)$ para todo $x \in I$ [1].  **Ejemplo:** $F(x) = -\cos(x)$ es una antiderivada de $f(x) = \sin(x)$, porque $F'(x) = -1\cdot(-\sin(x)) = \sin(x) =…»

## Fuera de corpus (métricas propias, aparte de Ragas)

- n = 0 · **tasa de abstención correcta = None** · **tasa de alucinación = None**
- Abstención indebida (dentro del corpus, respondió «no encontré»): 0.0
- Respuestas con citas a fragmentos inexistentes: 0

## Calibración del umbral de distancia (2.3)

Sin datos suficientes para calibrar.
