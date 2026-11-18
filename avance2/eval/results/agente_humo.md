# Evaluación del agente `humo`

Tutor: `claude-haiku-5-5` · 1 preguntas · modo LIVE · 2026-11-18T18:31:29

## Casos de comportamiento

| # | Tipo | Esperado | Obtenido (inicio) | Tools usadas | Llam. LLM | ¿Cumple? |
|---|---|---|---|---|---|---|
| 1 | Saludo o conversación casual | Responde el saludo sin llamar ninguna tool, con una sola llamada al LLM. | ¡Hola, buenas tardes! Estoy bien, gracias por preguntar. Soy tu tutor de Cálculo II y puedo ayudarte con técnicas de integración: antiderivación, sustitución, s | (ninguna) | 1 | Sí |
| 2 | Pregunta compuesta (dos temas) | Busca en la base más de una vez (una por tema), cubre ambos temas y cita con 'Fuente: documento, sec.'. | ## Regla ILATE en integración por partes La integración por partes se basa en $\int u\,dv = uv - \int v\,du$. El problema es elegir qué parte del integrando ser | buscar_en_base, buscar_en_base | 2 | Sí |
| 3 | Tarea del dominio | Usa calcular_integral con la expresión x^2, variable x y límites 0 y 3 extraídos de la pregunta; responde 9 (valor que devuelve la tool). | El valor de la integral definida es $$\int_0^3 x^2\,dx = 9$$ Esto coincide con el cálculo por la regla de Barrow: una antiderivada de $x^2$ es $\frac{x^3}{3}$,  | calcular_integral | 2 | Sí |

**3 de 3 casos ejecutados cumplen la comprobación automática**.
La valoración de la redacción la hace el estudiante.
