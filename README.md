# IAKonrad
Proyecto de Inteligencia artificial que emplea los modelos de respuesta de Google IA mediante Api, siendo configurados con instrucciones especificas para la correcta resolucion de solicitudes generas por comando y chat a este agente.

Para el correcto funcionamiento del sistema, es necesario contar con un entorno virtual, el cual, para windows, activaremos de la siguiente forma: python -m venv env. El cual al ejecutar nos permitira iniciar el proyecto (Si no tienes con las carpetas necesarias, solo con los archivos de python).

Para este momento, ya cuentas correctamente con el archivo de requisitos y tambien con la carpeta del entorno configurada, pero tu maquina no lo tiene aun para poder ejecutar, e spor eso que usaremos ese comando, ahora es importante que actualizes esto en tu maquina con el siguiente comando: .\env\Scripts\activate y posteriormente con: pip install -r requirements.txt

Es importante que crees un archivo de nombre: .env y dentro de el, añadas lo siguiente: GEMINI_API_KEY="Tu key" (la encuentras en: https://aistudio.google.com/api-keys)

De esta forma todo funcionara correctamente, por ultimo, ejecutar cada modelo mediant comando, los cuales son:  python .\Ejercicio1.py; python .\Ejercicio2.py; python .\Ejercicio3.py

Mediante este, doy finalidad al instructivo solicitado :D

Proyecto desarrollado usando modelo de IA para la generacion del codigo base, configuracion y detalles especificos y onfigurables, realizados por Sergio Herrera.

## Avance 1 — Asistente RAG

Este avance corresponde al proyecto "Asistente Experto basado en RAG y Agentes" (enfoque: Tutor Académico Personalizado, dominio: técnicas de integración de Cálculo II), implementado en `Avance1_TutorCalculo.py`.
- System prompting: el `system_instruction` define al modelo como tutor de Cálculo II, fija la pedagogía de pistas progresivas antes que solución completa, el formato de 6 pasos (identificar forma → plantear sustitución → simplificar → integrar → devolver la sustitución con triángulo de referencia → verificar derivando), la detección explícita de errores comunes de signo, y el formato de salida (español, Markdown, LaTeX con `$`).
- Few-shot prompting: la lista `ejemplos_few_shot` (sección `# 5.`) contiene dos turnos completos `user`/`model` construidos con `types.Content`, uno de sustitución trigonométrica y otro de sustitución combinada con integración por partes, cada uno mostrando el formato de 6 pasos completo sin saltar álgebra.
- Estrategia de delimitadores: la función `tutor_calculo`  envuelve el material de referencia en un bloque `<material_curso>...</material_curso>` en formato XML — en vez de comillas triples — precisamente porque el material de un curso de matemáticas puede incluir fórmulas o texto que ya contenga comillas simples, dobles o triples, lo que rompería un delimitador basado en comillas; las instrucciones de tarea y la pregunta del estudiante se ubican fuera de ese bloque para que el modelo distinga con claridad la referencia del encargo a resolver.
