"""Tests de la capa LLM. Por defecto NO usan ningun LLM real (gratis, sin red).

    python -m tests.test_llm            # solo tests locales (cliente simulado / LLM falso)
    python -m tests.test_llm --live     # ademas, 1 llamada minima real al modelo del tutor

Se escribieron sin pytest para no anadir una dependencia.
"""
from __future__ import annotations

import re
import sys
import tempfile
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import anthropic  # noqa: E402

from rag import config, generate, llm, prompts  # noqa: E402
from rag.retrieve import Fragmento  # noqa: E402

try:
    import httpx2 as httpx   # anthropic 1.x usa httpx2
except ImportError:          # pragma: no cover
    import httpx


# --------------------------- utilidades de test -----------------------------
def _resp(status: int):
    return httpx.Response(status, request=httpx.Request("POST", "https://api.anthropic.com/v1/messages"))


def _error(clase, status: int, mensaje: str = "boom"):
    return clase(mensaje, response=_resp(status), body={"type": "error", "error": {"type": "x", "message": mensaje}})


class _ClienteSimulado:
    """Imita anthropic.Anthropic().messages.create y registra los kwargs."""

    def __init__(self, resultados):
        self.resultados = list(resultados)     # excepciones o True (=exito)
        self.kwargs: list[dict] = []
        self.messages = types.SimpleNamespace(create=self._create)

    def _create(self, **kw):
        self.kwargs.append(kw)
        r = self.resultados.pop(0)
        if isinstance(r, Exception):
            raise r
        return types.SimpleNamespace(
            content=[types.SimpleNamespace(type="text", text="hola")],
            usage=types.SimpleNamespace(input_tokens=10, output_tokens=2,
                                        cache_creation_input_tokens=0, cache_read_input_tokens=0),
            stop_reason="end_turn")


def _llm_con(resultados) -> tuple[llm.LLMAnthropic, _ClienteSimulado]:
    l = llm.LLMAnthropic("claude-haiku-5-5")
    l._cliente = _ClienteSimulado(resultados)
    return l, l._cliente


MSG = [{"role": "user", "content": "hola"}]
_FRAGS = [Fragmento("texto uno", "a.txt", "Doc A", "Sec 1", 0, "teoria", "id1", 0.1, 1),
          Fragmento("texto dos", "b.txt", "Doc B", "Sec 2", 0, "teoria", "id2", 0.2, 2)]


# ------------------------------- los tests ----------------------------------
def test_parametros_enviados():
    l, c = _llm_con([True])
    l.generar(system="SYS", mensajes=MSG, max_tokens=123)
    kw = c.kwargs[0]
    assert kw["max_tokens"] == 123, "max_tokens explicito"
    assert kw["system"] == "SYS", "system va en su parametro"
    assert all(m["role"] in ("user", "assistant") for m in kw["messages"]), "system NO va en mensajes"
    for prohibido in ("temperature", "top_p", "top_k"):
        assert prohibido not in kw, f"no debe enviarse {prohibido}"
    # sin system: el parametro no se envia (no None)
    l, c = _llm_con([True])
    l.generar(system=None, mensajes=MSG, max_tokens=5)
    assert "system" not in c.kwargs[0]


def test_ninguna_llamada_usa_temperatura_en_el_codigo():
    # `top_k` solo se vigila en llm.py: en el resto es el top_k de RECUPERACION de fragmentos.
    muestreo = re.compile(r"(?<![A-Za-z_])temperature\s*=|\btop_p\s*=")
    for ruta in list((config.RAIZ / "rag").glob("*.py")) + list((config.RAIZ / "eval").glob("*.py")):
        for n, linea in enumerate(ruta.read_text(encoding="utf-8").splitlines(), 1):
            codigo = linea.split("#")[0]
            assert not muestreo.search(codigo), f"{ruta.name}:{n} fija temperature/top_p: {linea.strip()}"
            if ruta.name == "llm.py":
                assert not re.search(r"\btop_k\s*=", codigo), f"llm.py:{n} fija top_k"


def test_mensajes_alternan_y_empiezan_por_user():
    for historial in ([], [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]):
        m = prompts.armar_mensajes("pregunta", _FRAGS, historial)
        roles = [x["role"] for x in m]
        assert roles[0] == "user" and roles[-1] == "user"
        assert all(a != b for a, b in zip(roles, roles[1:])), "roles alternados"
        assert "system" not in roles
    assert len(prompts.EJEMPLOS_FEW_SHOT) == 10
    assert "<material_curso>" in m[-1]["content"] and "Pregunta del estudiante" in m[-1]["content"]


def test_reintenta_429_y_5xx_con_maximo_3():
    llm._dormir = lambda s: None   # sin esperas reales
    for cls, status in ((anthropic.RateLimitError, 429), (anthropic.InternalServerError, 500),
                        (anthropic.InternalServerError, 529)):
        l, c = _llm_con([_error(cls, status), _error(cls, status), True])
        assert l.generar(system=None, mensajes=MSG, max_tokens=5).texto == "hola"
        assert len(c.kwargs) == 3, f"{status}: 2 fallos + 1 exito"
        # siempre falla: 1 intento + 3 reintentos = 4 llamadas y se propaga
        l, c = _llm_con([_error(cls, status)] * 10)
        try:
            l.generar(system=None, mensajes=MSG, max_tokens=5)
            raise AssertionError("debio propagar el error")
        except cls:
            pass
        assert len(c.kwargs) == 1 + config.REINTENTOS_API == 4


def test_no_reintenta_otros_errores():
    llm._dormir = lambda s: None
    casos = [(anthropic.BadRequestError, 400, "messages: roles must alternate"),
             (anthropic.AuthenticationError, 401, "invalid x-api-key"),
             (anthropic.NotFoundError, 404, "model: x"),
             (anthropic.PermissionDeniedError, 403, "nope")]
    for cls, status, msg in casos:
        l, c = _llm_con([_error(cls, status, msg), True])
        try:
            l.generar(system=None, mensajes=MSG, max_tokens=5)
            raise AssertionError("debio propagar")
        except cls:
            pass
        assert len(c.kwargs) == 1, f"{status} no se reintenta"
    # tampoco los errores de conexion (solo 429 y 5xx)
    l, c = _llm_con([anthropic.APIConnectionError(request=httpx.Request("POST", "https://x")), True])
    try:
        l.generar(system=None, mensajes=MSG, max_tokens=5)
        raise AssertionError("debio propagar")
    except anthropic.APIConnectionError:
        pass
    assert len(c.kwargs) == 1


def test_limite_de_uso_no_se_reintenta():
    llm._dormir = lambda s: None
    casos = [(anthropic.APIStatusError, 402, "billing problem"),
             (anthropic.BadRequestError, 400, "Your credit balance is too low to access the Anthropic API."),
             (anthropic.BadRequestError, 400, "You have reached your specified workspace API usage limits.")]
    for cls, status, msg in casos:
        l, c = _llm_con([_error(cls, status, msg), True])
        try:
            l.generar(system=None, mensajes=MSG, max_tokens=5)
            raise AssertionError("debio lanzar LimiteDeUsoError")
        except llm.LimiteDeUsoError as e:
            assert str(e) == "El servicio alcanzó su límite de uso"
        assert len(c.kwargs) == 1, f"{status} '{msg[:25]}' NO se reintenta"


def test_cache_en_disco():
    with tempfile.TemporaryDirectory() as t:
        base = llm.LLMFalso(modelo="m1")
        c = llm.LLMConCache(base, Path(t))
        a = c.generar(system="S", mensajes=MSG, max_tokens=10)
        b = c.generar(system="S", mensajes=MSG, max_tokens=10)
        assert (a.desde_cache, b.desde_cache) == (False, True) and a.texto == b.texto
        assert len(base.llamadas) == 1, "la 2a llamada sale del disco"
        c.generar(system="S", mensajes=MSG, max_tokens=11)           # otro max_tokens -> otra clave
        c.generar(system="OTRO", mensajes=MSG, max_tokens=10)        # otro system -> otra clave
        assert len(base.llamadas) == 3
        assert llm.clave_cache("m1", "S", MSG, 10) != llm.clave_cache("m2", "S", MSG, 10), "el modelo entra en la clave"
        # una respuesta truncada NO se guarda
        trunc = llm.LLMFalso()
        trunc.generar = lambda **kw: llm.RespuestaLLM("parcial", "m", detenido_por="max_tokens")
        ct = llm.LLMConCache(trunc, Path(t) / "t2")
        ct.generar(system=None, mensajes=MSG, max_tokens=1)
        assert not list((Path(t) / "t2").glob("*.json"))


def test_modo_sin_live_usa_cache_o_falso_y_no_escribe():
    with tempfile.TemporaryDirectory() as t:
        real = llm.LLMConCache(llm.LLMFalso(modelo="claude-haiku-5-5"), Path(t))
        real.generar(system="S", mensajes=MSG, max_tokens=7)             # siembra la cache "real"
        off = llm.LLMCacheOFalso("claude-haiku-5-5", Path(t))
        assert off.generar(system="S", mensajes=MSG, max_tokens=7).desde_cache is True
        r = off.generar(system="S", mensajes=MSG, max_tokens=8)           # no esta en cache -> falso
        assert r.modelo.startswith("falso:") and not r.desde_cache
        assert len(list(Path(t).glob("*.json"))) == 1, "no escribe en la cache"
        try:
            llm.LLMCacheOFalso("claude-haiku-5-5", Path(t), solo_cache=True).generar(system="S", mensajes=MSG, max_tokens=8)
            raise AssertionError("debio fallar")
        except llm.SinCacheError:
            pass


def test_responder_con_llm_falso():
    original = generate.recuperar
    generate.recuperar = lambda q, top_k=None: (consultas.append(q), _FRAGS)[1]
    consultas: list[str] = []
    try:
        def responder_falso(system, mensajes):
            seg = re.search(r"<pregunta_seguimiento>\s*(.*?)\s*</pregunta_seguimiento>", mensajes[-1]["content"], re.S)
            return seg.group(1) if seg else "Segun [1] y [2] y tambien [5]. Con $[0,1]$ no cita."
        falso = llm.LLMFalso(responder=responder_falso)
        r = generate.responder("¿Y si fuera 9x^2?", [generate.Turno("user", "q1"), generate.Turno("assistant", "a1")],
                               llm=falso)
        # 1) condensacion (sin system) + 2) generacion (con system, max_tokens explicito)
        assert len(falso.llamadas) == 2
        cond, gen = falso.llamadas
        assert cond["system"] is None and cond["max_tokens"] == config.MAX_TOKENS_CONDENSACION
        assert gen["system"] == prompts.SYSTEM_PROMPT and gen["max_tokens"] == config.MAX_TOKENS_RESPUESTA
        assert [m["role"] for m in gen["mensajes"]][-3:] == ["user", "assistant", "user"], "historial real + turno actual"
        assert consultas == ["¿Y si fuera 9x^2?"], "se recupera con la pregunta condensada (el falso la devuelve igual)"
        assert r.citas_invalidas == [5], f"cita inexistente detectada, no la del intervalo: {r.citas_invalidas}"
        # sin historial: NO se llama a la condensacion
        falso2 = llm.LLMFalso()
        generate.responder("pregunta sola", [], llm=falso2)
        assert len(falso2.llamadas) == 1
        # un limite de uso se propaga limpio
        class Agotado(llm.LLMFalso):
            def generar(self, **kw):
                raise llm.LimiteDeUsoError("x")
        try:
            generate.responder("p", [], llm=Agotado())
            raise AssertionError("debio propagar")
        except llm.LimiteDeUsoError:
            pass
    finally:
        generate.recuperar = original


def test_live_ping():
    """1 llamada minima real al modelo del tutor (cuesta una fraccion de centavo)."""
    l = llm.crear_llm("tutor", live=True, usar_cache=False)
    r = l.generar(system="Responde solo con la palabra: ok", mensajes=[{"role": "user", "content": "hola"}], max_tokens=300)
    assert r.texto, f"respuesta vacia (detenido_por={r.detenido_por})"
    print(f"      live: modelo={r.modelo} texto={r.texto[:30]!r} tokens={r.tokens_entrada}/{r.tokens_salida} stop={r.detenido_por}")


LOCALES = [v for k, v in sorted(globals().items()) if k.startswith("test_") and k != "test_live_ping"]

if __name__ == "__main__":
    pruebas = LOCALES + ([test_live_ping] if "--live" in sys.argv else [])
    fallos = 0
    for t in pruebas:
        try:
            t()
            print(f"  OK    {t.__name__}")
        except Exception as e:  # noqa: BLE001
            fallos += 1
            print(f"  FALLO {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(pruebas) - fallos}/{len(pruebas)} tests OK" + ("" if "--live" in sys.argv else "  (sin --live: ningun LLM real)"))
    sys.exit(1 if fallos else 0)
