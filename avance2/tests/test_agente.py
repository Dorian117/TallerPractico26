"""Tests del agente y sus tools SIN LLM real, SIN API y SIN red (cliente simulado). Gasto 0.

    python -m tests.test_agente

Comprueban: limites de buscar_en_base (K y longitud), formato de fuentes, limite de iteraciones, saludo sin
tools, parseo seguro del calculo (incluida una entrada maliciosa), restriccion de dominios y maximo de 3
resultados web, politica "primero la base", higiene de la consulta web y formato del log.

ESTO NO SUSTITUYE LA EJECUCION REAL: aqui el "modelo" es un guion; que Haiku decida bien cuando llamar cada
tool solo se comprueba con `python -m eval.evaluar_agente --casos --live`.
"""
from __future__ import annotations

import json
import sys
import tempfile
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx  # noqa: E402

from rag import agent, calculo, config, llm, tools, tools_web  # noqa: E402
from rag.llm import RespuestaLLM  # noqa: E402
from rag.prompts_agente import MENSAJE_MAX_ITER  # noqa: E402
from rag.retrieve import Fragmento  # noqa: E402


# --------------------------- utilidades de test -----------------------------
def _frags(n=10, largo=5000, pagina=0):
    return [Fragmento("A" * largo, "a.txt", f"Doc {i}", f"Sec {i} > Sub {i}", pagina, "teoria", f"id{i}", 0.1 + i / 100, i + 1)
            for i in range(n)]


class _Parche:
    """Sustituye atributos de un objeto durante un `with` y los restaura (sin pytest)."""

    def __init__(self, objeto, **cambios):
        self.objeto, self.cambios, self.viejos = objeto, cambios, {}

    def __enter__(self):
        for k, v in self.cambios.items():
            self.viejos[k] = getattr(self.objeto, k)
            setattr(self.objeto, k, v)

    def __exit__(self, *a):
        for k, v in self.viejos.items():
            setattr(self.objeto, k, v)


def _usar(nombre, id_="t1", **entrada):
    return {"type": "tool_use", "id": id_, "name": nombre, "input": entrada}


def _pide(*bloques):
    return RespuestaLLM("", "falso", 100, 10, "tool_use", bloques=list(bloques))


def _resp_web(items, status=200):
    return types.SimpleNamespace(status_code=status, json=lambda: {"results": items})


_URL = {"url": "https://tutorial.math.lamar.edu/Classes/CalcII/IntegrationByParts.aspx"}


# ------------------- 1. tool de busqueda: limites y formato ------------------
def test_buscar_en_base_respeta_K_MAX_y_MAX_CHARS():
    with _Parche(tools, recuperar=lambda q, top_k=None: _frags(10)):
        r = tools.buscar_en_base("cualquier cosa")
    assert len(r["fragmentos"]) == config.K_MAX, "el codigo recorta aunque el buscador devuelva mas"
    for f in r["fragmentos"]:
        assert len(f["texto"]) <= config.MAX_CHARS + len(" [...recortado]")
        assert f["texto"].endswith("[...recortado]")


def test_recorte_no_parte_una_formula():
    texto = "Intro " + "a" * 900 + "\n\n$$" + "x" * 400 + "$$\n\nfinal"
    corto = tools._recortar(texto, config.MAX_CHARS)
    assert corto.count("$$") % 2 == 0, "una formula de $$ nunca queda abierta"
    assert tools._recortar("breve", config.MAX_CHARS) == "breve"


def test_formato_de_fuentes_documento_y_seccion_o_pagina():
    with _Parche(tools, recuperar=lambda q, top_k=None: _frags(2, largo=50)):
        f = tools.buscar_en_base("x")["fragmentos"][0]
    assert f["origen"] == "base" and f["documento"] == "Doc 0" and f["seccion"]
    assert f["fuente"] == "Fuente: Doc 0, sec. «Sec 0 > Sub 0»"
    with _Parche(tools, recuperar=lambda q, top_k=None: _frags(1, largo=50, pagina=3)):
        assert tools.buscar_en_base("x")["fragmentos"][0]["fuente"] == "Fuente: Doc 0, pág. 3"


def test_buscar_en_base_valida_la_consulta():
    assert "error" in tools.buscar_en_base("   ") and "error" in tools.buscar_en_base(None)
    vistas = []
    with _Parche(tools, recuperar=lambda q, top_k=None: vistas.append(q) or _frags(1, 50)):
        tools.buscar_en_base("q" * 5000)
    assert len(vistas[0]) == config.CONSULTA_MAX


def test_buscar_en_base_real_con_el_indice():
    r = tools.buscar_en_base("regla ILATE integración por partes")
    assert 0 < len(r["fragmentos"]) <= config.K_MAX
    assert all(f["documento"] and f["seccion"] and len(f["texto"]) <= config.MAX_CHARS + 20 for f in r["fragmentos"])


def test_ejecutar_solo_acepta_argumentos_declarados():
    e = {"base_consultada": True}
    assert "no permitidos" in tools.ejecutar("buscar_en_web", {"consulta": "x", "include_domains": ["evil.com"]}, e)["error"]
    assert "desconocida" in tools.ejecutar("borrar_todo", {}, e)["error"]
    assert "JSON" in tools.ejecutar("buscar_en_base", "texto", e)["error"]
    assert "invalidos" in tools.ejecutar("calcular_integral", {}, e)["error"]      # falta el argumento obligatorio


def test_definiciones_traen_docstring_y_esquema():
    defs = {d["name"]: d for d in tools.definiciones()}
    assert set(defs) == {"buscar_en_base", "calcular_integral", "verificar_antiderivada", "buscar_en_web"}
    for d in defs.values():
        assert len(d["description"]) > 100 and d["input_schema"]["required"]
    assert "include_domains" not in json.dumps(defs["buscar_en_web"]), "el modelo no ve ni controla los dominios"


# ------------------- 2. calculo seguro ----------------------------------------
def test_calculo_resultados_correctos():
    assert calculo.integral("x*exp(x)")["resultado"] == "(x - 1)*exp(x) + C"
    assert calculo.integral("2x*cos(x**2)")["resultado"] == "sin(x**2) + C"        # multiplicacion implicita
    d = calculo.integral("x**2", "x", 0, 3)
    assert d["resultado_exacto"] == "9" and d["valor_numerico"].startswith("9")
    assert calculo.verificar("x*exp(x)", "x*exp(x) - exp(x) + C")["coincide"] is True
    assert calculo.verificar("x*exp(x)", "x*exp(x) + exp(x)")["coincide"] is False


def test_calculo_rechaza_entradas_maliciosas_o_mal_formadas():
    malas = ["__import__('os').system('echo hola')", "open('.env').read()", "x.__class__", "x**2; import os",
             "lambda: 1", "9**9**9**9", "x" * 300, "", "x**(", "\\frac{1}{x}", "eval('1')", "(1).__add__"]
    for m in malas:
        r = calculo.integral(m)
        assert "error" in r and "resultado" not in r, f"debia rechazar {m!r}: {r}"
    assert "error" in calculo.integral("x**2", variable="exec")
    assert "error" in calculo.integral("x", limite_inferior=0)                     # un solo limite
    assert "error" in calculo.integral("x", limite_inferior="x", limite_superior=1)  # limite con variable
    assert "error" in calculo.integral("1/x**2", limite_inferior=-1, limite_superior=1)   # diverge
    assert "error" in calculo.verificar("x", "os.system(1)")


def test_calculo_corre_en_proceso_con_tiempo_maximo():
    assert tools.calcular_integral("x*exp(x)")["resultado"] == "(x - 1)*exp(x) + C"
    with _Parche(config, CALCULO_TIMEOUT_S=0.2):       # menos de lo que tarda el hijo en arrancar
        assert "tiempo maximo" in tools.calcular_integral("x*exp(x)")["error"]


# ------------------- 3. busqueda web controlada --------------------------------
def _web(items, enviados=None, status=200, clave="tvly-clave-de-prueba"):
    def falso(payload, k):
        if enviados is not None:
            enviados.append((payload, k))
        return _resp_web(items, status)
    return _Parche(tools_web, _llamar_servicio=falso, _secreto=lambda n: clave, _corpus_normalizado=lambda: "texto del corpus " * 50)


def test_web_dominios_fijos_y_maximo_3_resultados():
    items = [{"title": f"R{i}", "url": f"https://{d}/p{i}", "content": "c" * 900}
             for i, d in enumerate(["evil.com", "ocw.mit.edu", "openstax.org", "tutorial.math.lamar.edu",
                                    "mathworld.wolfram.com", "ocw.mit.edu.evil.com", "openstax.org"])]
    enviados = []
    with _web(items, enviados):
        r = tools_web.buscar_en_web("regla de la cadena")
    assert len(r["resultados"]) == config.WEB_MAX_RESULTADOS == 3
    assert all(tools_web._permitido(x["url"]) and x["origen"] == "web" for x in r["resultados"])
    assert not any("evil.com" in x["url"] for x in r["resultados"]), "dominios ajenos filtrados aunque el servicio los devuelva"
    assert all(len(x["extracto"]) <= config.WEB_EXTRACTO_MAX for x in r["resultados"])
    assert set(r["resultados"][0]) == {"origen", "titulo", "url", "extracto"}
    payload, clave = enviados[0]
    assert payload["include_domains"] == list(tools_web.SITIOS_PERMITIDOS) and payload["max_results"] == 3
    assert clave == "tvly-clave-de-prueba" and "api_key" not in payload


def test_web_politica_primero_la_base():
    with _web([]):
        r = tools.ejecutar("buscar_en_web", {"consulta": "algo publico"}, {})
        assert "Primero consulta la base" in r["error"]
        assert "error" not in tools.ejecutar("buscar_en_web", {"consulta": "algo publico"}, {"base_consultada": True})


def test_web_higiene_de_la_consulta():
    enviados = []
    corpus = "la regla ilate establece un criterio de prioridad descendente para seleccionar la funcion u"
    with _Parche(tools_web, _llamar_servicio=lambda p, k: enviados.append(p) or _resp_web([]),
                 _secreto=lambda n: "tvly-x", _corpus_normalizado=lambda: corpus):
        r = tools_web.buscar_en_web("La regla ILATE establece un criterio de prioridad descendente para seleccionar la funcion u")
        assert "copiado de la base" in r["error"] and not enviados, "texto de la base: NO se envia nada al buscador"
        tools_web.buscar_en_web("q " * 400)
        assert len(enviados[0]["query"]) <= config.WEB_CONSULTA_MAX
        assert "\x00" not in enviados[0]["query"]


def test_web_fallos_controlados_sin_romper_al_agente():
    import httpx as h
    def roto(p, k):
        raise h.ConnectError("sin red")
    with _Parche(tools_web, _llamar_servicio=roto, _secreto=lambda n: "tvly-x", _corpus_normalizado=lambda: ""):
        assert "no respondio" in tools_web.buscar_en_web("algo")["error"]
    for status in (401, 429, 500):
        with _web([], status=status):
            assert "fallo" in tools_web.buscar_en_web("algo")["error"]
    with _web([], clave=None):
        assert "TAVILY_API_KEY" in tools_web.buscar_en_web("algo")["error"]
    with _web([]):
        r = tools_web.buscar_en_web("algo")
        assert r["resultados"] == [] and "Sin resultados" in r["nota"]


# ------------------- 4. el bucle del agente -----------------------------------
def test_saludo_se_responde_sin_tools():
    f = llm.LLMFalso()
    r = agent.responder_agente("Hola, ¿cómo estás?", llm=f, registrar=False)
    assert r.llamadas_tools == [] and r.n_llamadas_llm == 1 and r.estado == "ok"
    assert f.llamadas[0]["tools"] == [d["name"] for d in tools.definiciones()], "las tools se ofrecen; decide el modelo"


def test_limite_de_iteraciones_sin_bucle_infinito():
    f = llm.LLMFalso(lambda s, m: _pide(_usar("buscar_en_base", consulta="algo")))
    with _Parche(tools, recuperar=lambda q, top_k=None: _frags(1, 50)):
        r = agent.responder_agente("pregunta eterna", llm=f, registrar=False)
    assert r.n_llamadas_llm == config.MAX_ITER and r.estado == "max_iter" and r.texto == MENSAJE_MAX_ITER
    assert [x["sin_tools"] for x in f.llamadas] == [False] * (config.MAX_ITER - 1) + [True], "la ultima prohibe tools"
    assert len(r.llamadas_tools) == config.MAX_ITER - 1, "no se ejecuta la tool pedida en la ultima iteracion"


def test_al_agotar_iteraciones_responde_con_lo_que_tiene_si_el_modelo_obedece():
    def guion(system, mensajes):
        if len(mensajes) < 2 * (config.MAX_ITER - 1) + 1:
            return _pide(_usar("buscar_en_base", consulta="algo"))
        return "Con lo que encontré: ILATE ordena las funciones."
    with _Parche(tools, recuperar=lambda q, top_k=None: _frags(1, 50)):
        r = agent.responder_agente("pregunta larga", llm=llm.LLMFalso(guion), registrar=False)
    assert r.estado == "ok" and r.n_llamadas_llm == config.MAX_ITER and r.texto.startswith("Con lo que encontré")


def test_pregunta_compuesta_dos_busquedas_y_citas():
    def guion(system, mensajes):
        ultimo = mensajes[-1]["content"]
        if isinstance(ultimo, str):
            return _pide(_usar("buscar_en_base", "t1", consulta="ILATE"), _usar("buscar_en_base", "t2", consulta="sustitución trigonométrica"))
        assert [b["tool_use_id"] for b in ultimo] == ["t1", "t2"] and all("is_error" not in b for b in ultimo)
        buena = json.loads(ultimo[0]["content"])["fragmentos"][0]["fuente"]
        return f"Dos temas.\n\n{buena}\nFuente: Documento Inventado, sec. «Nada»"
    with _Parche(tools, recuperar=lambda q, top_k=None: _frags(2, 50)):
        r = agent.responder_agente("ILATE y sustitución trigonométrica", llm=llm.LLMFalso(guion), registrar=False)
    assert [c.nombre for c in r.llamadas_tools] == ["buscar_en_base"] * 2 and r.n_llamadas_llm == 2
    assert r.citas_invalidas == ["Fuente: Documento Inventado, sec. «Nada»"], "solo la fuente que ninguna tool devolvio"
    assert len(r.fragmentos_base) == 4, "2 busquedas x 2 fragmentos"


def test_error_de_tool_vuelve_al_modelo_como_is_error():
    def guion(system, mensajes):
        ultimo = mensajes[-1]["content"]
        if isinstance(ultimo, str):
            return _pide(_usar("tool_inventada", x=1))
        assert ultimo[0]["is_error"] is True and "desconocida" in ultimo[0]["content"]
        return "No pude usar esa herramienta."
    r = agent.responder_agente("hazlo", llm=llm.LLMFalso(guion), registrar=False)
    assert r.estado == "ok" and r.llamadas_tools[0].error


def test_historial_viaja_en_los_mensajes():
    from rag.generate import Turno
    f = llm.LLMFalso()
    agent.responder_agente("¿y si fuera 9x²?", [Turno("user", "q1"), Turno("assistant", "a1")], llm=f, registrar=False)
    assert [m["role"] for m in f.llamadas[0]["mensajes"]] == ["user", "assistant", "user"]


def test_limite_de_gasto_y_errores_se_propagan_y_quedan_en_el_log():
    class Agotado(llm.LLMFalso):
        def generar_con_tools(self, **kw):
            raise llm.LimiteDeUsoError("billing")
    with tempfile.TemporaryDirectory() as t, _Parche(config, LOGS_DIR=Path(t), LOG_USO_ACTIVO=True):
        try:
            agent.responder_agente("hola", llm=Agotado())
            raise AssertionError("debia propagar LimiteDeUsoError")
        except llm.LimiteDeUsoError:
            pass
        assert json.loads((Path(t) / "uso_tools.jsonl").read_text(encoding="utf-8"))["estado"] == "limite"


# ------------------- 5. registro de uso ---------------------------------------
def test_formato_del_log_de_uso():
    def guion(system, mensajes):
        if isinstance(mensajes[-1]["content"], str):
            return _pide(_usar("buscar_en_base", consulta="ILATE"))
        return "Respuesta."
    frs = [Fragmento("TEXTO-SECRETO-DEL-FRAGMENTO " * 20, "a.txt", "Doc A", "Sec A", 0, "t", "id", 0.1, 1)]
    with tempfile.TemporaryDirectory() as t, _Parche(config, LOGS_DIR=Path(t) / "logs", LOG_USO_ACTIVO=True), \
            _Parche(tools, recuperar=lambda q, top_k=None: frs):
        agent.responder_agente("¿Qué es ILATE? sk-ant-NO-DEBERIA-ESTAR", llm=llm.LLMFalso(guion))
        agent.responder_agente("Hola", llm=llm.LLMFalso())
        crudo = (Path(t) / "logs" / "uso_tools.jsonl").read_text(encoding="utf-8")
    lineas = [json.loads(l) for l in crudo.splitlines()]
    assert len(lineas) == 2, "una linea por pregunta"
    a = lineas[0]
    assert {"ts", "pregunta", "modelo", "tools", "n_llamadas_llm", "tokens_entrada", "tokens_salida", "segundos", "estado"} <= set(a)
    assert a["n_llamadas_llm"] == 2 and a["tokens_entrada"] > 100 and a["tokens_salida"] > 10 and a["estado"] == "ok"
    assert a["tools"][0]["nombre"] == "buscar_en_base" and a["tools"][0]["argumentos"] == {"consulta": "ILATE"}
    assert a["tools"][0]["fragmentos"] == [{"documento": "Doc A", "seccion": "Sec A", "pagina": 0}]
    assert "TEXTO-SECRETO-DEL-FRAGMENTO" not in crudo, "el log no guarda el texto de los fragmentos"
    assert "NO-DEBERIA-ESTAR" not in crudo and "[CLAVE-OCULTA]" in a["pregunta"], "una clave pegada en la pregunta no se guarda"
    assert lineas[1]["tools"] == [] and lineas[1]["n_llamadas_llm"] == 1


def test_el_log_se_puede_apagar():
    with tempfile.TemporaryDirectory() as t, _Parche(config, LOGS_DIR=Path(t) / "logs", LOG_USO_ACTIVO=False):
        agent.responder_agente("Hola", llm=llm.LLMFalso())
        assert not (Path(t) / "logs").exists()


# ------------------- 6. capa LLM con tools ------------------------------------
class _Cliente:
    def __init__(self, contenido, parada="end_turn"):
        self.kwargs, self.contenido, self.parada = [], contenido, parada
        self.messages = types.SimpleNamespace(create=self._crear)

    def _crear(self, **kw):
        self.kwargs.append(kw)
        return types.SimpleNamespace(content=self.contenido, stop_reason=self.parada, usage=types.SimpleNamespace(
            input_tokens=50, output_tokens=7, cache_creation_input_tokens=0, cache_read_input_tokens=0))


def _anthropic(contenido, parada="end_turn"):
    l = llm.LLMAnthropic("claude-haiku-5-5")
    l._cliente = _Cliente(contenido, parada)
    return l, l._cliente


def test_anthropic_con_tools_parametros_y_bloques():
    contenido = [types.SimpleNamespace(type="thinking", thinking="pienso", signature="firma"),
                 types.SimpleNamespace(type="text", text=""),
                 types.SimpleNamespace(type="tool_use", id="tu1", name="buscar_en_base", input={"consulta": "x"})]
    l, c = _anthropic(contenido, "tool_use")
    r = l.generar_con_tools(system="S", mensajes=[{"role": "user", "content": "q"}], tools=tools.definiciones(), max_tokens=123)
    k = c.kwargs[0]
    assert k["tools"] and k["system"] == "S" and k["max_tokens"] == 123 and "tool_choice" not in k
    assert not {"temperature", "top_p", "top_k"} & set(k), "sin muestreo"
    assert [b["type"] for b in r.bloques] == ["thinking", "tool_use"], "el texto vacio se descarta; el razonamiento se conserva"
    assert r.detenido_por == "tool_use" and r.tokens_entrada == 50 and r.tokens_salida == 7
    l.generar_con_tools(system="S", mensajes=[{"role": "user", "content": "q"}], tools=tools.definiciones(), max_tokens=1, sin_tools=True)
    assert c.kwargs[1]["tool_choice"] == {"type": "none"} and c.kwargs[1]["tools"], "ultima iteracion: sin tools pero con definicion"


def test_cache_con_tools_guarda_tool_use_y_no_trunca():
    llamadas = []

    class Base:
        modelo = "m"
        def generar_con_tools(self, **kw):
            llamadas.append(kw)
            return RespuestaLLM("", "m", 5, 5, "tool_use", bloques=[_usar("buscar_en_base", consulta="x")])
    with tempfile.TemporaryDirectory() as t:
        c = llm.LLMConCache(Base(), Path(t))
        args = dict(system="S", mensajes=[{"role": "user", "content": "q"}], tools=[{"name": "a"}], max_tokens=10)
        r1, r2 = c.generar_con_tools(**args), c.generar_con_tools(**args)
        assert len(llamadas) == 1 and r2.desde_cache and r2.bloques == r1.bloques, "una respuesta solo-tool_use tambien se cachea"
        c.generar_con_tools(**{**args, "sin_tools": True})
        assert len(llamadas) == 2, "sin_tools entra en la clave"
        escrita = list(Path(t).glob("*.json"))
        assert len(escrita) == 2


def test_ninguna_tool_ni_el_agente_usan_temperatura_ni_secretos_en_codigo():
    import re
    for nombre in ("agent.py", "tools.py", "tools_web.py", "calculo.py", "prompts_agente.py"):
        codigo = (Path(config.RAIZ) / "rag" / nombre).read_text(encoding="utf-8")
        assert not re.search(r"temperature|top_p", codigo), nombre
        assert not re.search(r"tvly-[A-Za-z0-9]{10,}|sk-ant-[A-Za-z0-9_-]{10,}", codigo), nombre
    assert "TAVILY_API_KEY=" in (Path(config.RAIZ) / ".env.example").read_text(encoding="utf-8")
    assert "tvly-" not in (Path(config.RAIZ) / ".env.example").read_text(encoding="utf-8")


LOCALES = [v for k, v in sorted(globals().items()) if k.startswith("test_")]

if __name__ == "__main__":
    fallos = 0
    for t in LOCALES:
        try:
            t()
            print(f"  OK    {t.__name__}")
        except Exception as e:  # noqa: BLE001
            fallos += 1
            print(f"  FALLO {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(LOCALES) - fallos}/{len(LOCALES)} tests OK  (ningun LLM real, ninguna API, ninguna red)")
    print("Esto NO sustituye la ejecucion real: `python -m eval.evaluar_agente --casos --live`.")
    sys.exit(1 if fallos else 0)
