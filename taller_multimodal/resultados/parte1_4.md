## 1.4 Error provocado

### Caso `imagenes`
- **Qué se hizo:** Se enviaron 4 imágenes; el modelo admite máximo 3 por solicitud.
- **Modelo:** `qwen/qwen3.8-27b`
- **Código HTTP:** 400
- **Mensaje de error:**

```
Error code: 400 - {'error': {'message': 'Too many images provided.  This model supports up to 3 images', 'type': 'invalid_request_error'}}
```
- **Causa:** (explícala con tus palabras)
