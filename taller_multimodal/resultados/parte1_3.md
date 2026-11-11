## 1.3 Varias imágenes en una solicitud

- **Imágenes enviadas:** `ej2_partes.png`, `ej5_error_signo.png`
- **Pregunta:** Las dos imágenes resuelven la misma integral por partes. Compara las soluciones: ¿cuál está bien y cuál tiene un error? Indica en qué paso está el error y cómo se corrige.

**Respuesta del modelo (qwen/qwen3.8-27b):**

**Solución correcta:** La primera imagen.

**Solución con error:** La segunda imagen (resuelta por otro estudiante).

**Error:** En el **Paso 4**.
El estudiante escribe: $= xe^x + e^x + C$.
El error es el signo. Debe ser un signo **menos** ($-$), no más ($+$).

**Corrección:**
La fórmula de integración por partes es $\int u \, dv = uv - \int v \, du$.
En el Paso 3 se tiene correctamente: $xe^x - \int e^x \, dx$.
Al integrar $\int e^x \, dx$, se obtiene $e^x$.
Por lo tanto, el Paso 4 correcto es:
$$xe^x - e^x + C$$

El estudiante de la segunda imagen olvidó el signo negativo que viene de la fórmula de integración por partes.
