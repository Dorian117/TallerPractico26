"""Lista los modelos disponibles para tu clave de Groq (útil si un id da 404 model_not_found)."""
import common as c

cliente = c.obtener_cliente()
for m in sorted(cliente.models.list().data, key=lambda m: m.id):
    print(m.id)
