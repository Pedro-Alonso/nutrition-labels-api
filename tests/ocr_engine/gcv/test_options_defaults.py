"""Property tests para defaults do bloco ``gcv`` de preset.

Cobre a tarefa **2.4** do plano de implementação: a Property 4 do design
documentando o contrato de ``GcvPresetOptions.from_dict`` com relação a
chaves ausentes do bloco ``gcv``.

Em palavras: para qualquer dict ``data`` passado a
``GcvPresetOptions.from_dict``, quando a chave ``"feature"`` está ausente
o resultado expõe ``feature == "DOCUMENT_TEXT_DETECTION"`` (Requirement
3.3); quando a chave ``"language_hints"`` está ausente o resultado expõe
``language_hints == ("pt",)`` (Requirement 3.6). Quando ``"model"`` está
ausente o resultado expõe ``model is None`` (parte do mesmo contrato de
defaults documentado no design).

A estratégia gera dicts arbitrários
(``dictionaries(text(), one_of(text(), none(), lists(text())))``) para
exercitar o domínio de entradas ruidosas que poderiam aparecer em JSONs
de preset reais. Quando os geradores produzem chaves coincidentes com
``"feature"`` ou ``"language_hints"``, usamos ``assume`` para focar a
propriedade no ramo de defaults — esse é o invariante que a Property 4
declara. Casos com ``feature`` ou ``language_hints`` declarados são
cobertos por testes determinísticos complementares.
"""

from __future__ import annotations

from pathlib import Path

from hypothesis import assume, given, settings
from hypothesis import strategies as st

from ocr.cloud_vision.app_config import GcvAppConfig
from ocr.cloud_vision.options import GcvPresetOptions


# Estratégia de valores conforme especificado na task: textos, ``None`` e
# listas de textos. Cobre o que aparece em JSONs reais (strings curtas,
# listas de hints) e também o cenário "campo declarado como ``None``"
# que é semanticamente diferente de "campo ausente" para o ``model``.
_VALUES = st.one_of(
    st.text(),
    st.none(),
    st.lists(st.text()),
)


# Estratégia principal: dicts arbitrários com chaves textuais e valores
# heterogêneos. Mantemos ``max_size`` desnecessário (default do Hypothesis
# já produz dicts pequenos) — a propriedade não depende do tamanho do
# dict, só da ausência/presença das chaves específicas.
_ARBITRARY_DICTS = st.dictionaries(st.text(), _VALUES)


# -----------------------------------------------------------------------
# Property 4 — Validates: Requirements 3.3, 3.6
# -----------------------------------------------------------------------


@given(data=_ARBITRARY_DICTS)
@settings(max_examples=100, deadline=None)
def test_feature_defaults_to_document_text_detection_when_absent(
    data: dict,
) -> None:
    """**Property 4 (parte feature)**: ``feature`` ausente ⇒ default canônico.

    **Validates: Requirements 3.3**

    Para qualquer dict ``data`` que não declare a chave ``"feature"``,
    ``GcvPresetOptions.from_dict(data).feature`` é igual a
    ``"DOCUMENT_TEXT_DETECTION"`` e ``invalid_feature`` permanece
    ``False`` (ausência não é inválida — apenas dispara o default).
    """

    assume("feature" not in data)

    options = GcvPresetOptions.from_dict(data)

    assert options.feature == "DOCUMENT_TEXT_DETECTION"
    assert options.invalid_feature is False
    assert options.raw_feature is None


@given(data=_ARBITRARY_DICTS)
@settings(max_examples=100, deadline=None)
def test_language_hints_defaults_to_pt_when_absent(data: dict) -> None:
    """**Property 4 (parte language_hints)**: ``language_hints`` ausente ⇒ ``("pt",)``.

    **Validates: Requirements 3.6**

    Para qualquer dict ``data`` que não declare ``"language_hints"``, o
    resultado expõe a tupla canônica ``("pt",)``. A imutabilidade do
    valor (tupla, não lista) é parte do contrato do cache (Requirement
    7.3) — verificamos o tipo explicitamente.
    """

    assume("language_hints" not in data)

    options = GcvPresetOptions.from_dict(data)

    assert options.language_hints == ("pt",)
    assert isinstance(options.language_hints, tuple)


@given(data=_ARBITRARY_DICTS)
@settings(max_examples=100, deadline=None)
def test_model_defaults_to_none_when_absent(data: dict) -> None:
    """**Property 4 (parte model)**: ``model`` ausente ⇒ ``None``.

    Complementa a Property 4: o mesmo dict que aciona os defaults de
    ``feature`` e ``language_hints`` também precisa expor ``model is
    None`` quando a chave está ausente, conforme documentado no bloco
    ``gcv`` do design.
    """

    assume("model" not in data)

    options = GcvPresetOptions.from_dict(data)

    assert options.model is None


# -----------------------------------------------------------------------
# Casos determinísticos auxiliares
# -----------------------------------------------------------------------


def test_from_dict_none_yields_full_defaults() -> None:
    """``from_dict(None)`` aplica todos os defaults canônicos.

    Caso degenerado equivalente a "bloco ``gcv`` ausente do preset".
    Garante que a propriedade vale também para ``data is None`` —
    cenário que o gerador de dicts não cobre por construção.
    """

    options = GcvPresetOptions.from_dict(None)

    assert options.feature == "DOCUMENT_TEXT_DETECTION"
    assert options.language_hints == ("pt",)
    assert options.model is None
    assert options.invalid_feature is False
    assert options.raw_feature is None


def test_from_dict_empty_dict_yields_full_defaults() -> None:
    """``from_dict({})`` aplica todos os defaults canônicos.

    Caso explícito: bloco ``"gcv": {}`` declarado no JSON com objeto
    vazio. O resultado tem que ser indistinguível de ``from_dict(None)``.
    """

    options = GcvPresetOptions.from_dict({})

    assert options.feature == "DOCUMENT_TEXT_DETECTION"
    assert options.language_hints == ("pt",)
    assert options.model is None
    assert options.invalid_feature is False
    assert options.raw_feature is None


# =======================================================================
# Property 11: Coerção de configuração inválida com warning
# =======================================================================
#
# Cobre a tarefa **2.5** do plano de implementação. A Property 11 do
# design declara o contrato de ``GcvAppConfig.from_dict`` para o campo
# ``max_requests_per_minute``: qualquer valor que não seja um inteiro
# positivo (``null``, ``0``, inteiro negativo, string, float, ``bool``)
# deve ser coercido para ``None`` E gerar uma string descritiva em
# ``config_warnings`` citando o valor original e a coerção aplicada.
#
# A estratégia abaixo segue exatamente a especificada em ``tasks.md``:
# ``one_of(integers(max_value=0), text(), floats(), booleans(), none())``.
# Cada branch dessa união cobre uma classe de valor inválido distinta:
#
# - ``integers(max_value=0)`` exercita ``0`` e negativos (ramo
#   "não-positivo" do parser).
# - ``text()`` exercita strings arbitrárias, incluindo strings vazias e
#   strings que parecem números mas não são (ramo "string").
# - ``floats()`` exercita floats incluindo ``nan``, ``inf``, ``-inf`` e
#   floats que coincidiriam com inteiros positivos (ex.: ``5.0``) —
#   mesmo assim devem ser rejeitados porque o contrato é de **inteiro
#   estrito** (Requirement 8.2).
# - ``booleans()`` exercita ``True``/``False`` que, sendo subclasse de
#   ``int`` em Python, enganariam ``isinstance(raw, int)`` se o parser
#   não tratasse ``bool`` antes do branch numérico.
# - ``none()`` exercita o caso "valor declarado como ``null``", que é
#   semanticamente diferente de "chave ausente do bloco ``gcv``": o
#   primeiro é uma escolha consciente do operador e merece registro
#   explícito em ``config_warnings``; o segundo cai no default
#   silencioso da Requirement 4.2.
# -----------------------------------------------------------------------


# Path estável para passar como ``project_root`` em todos os exemplos do
# Hypothesis. A função ``from_dict`` só usa esse argumento para resolver
# paths relativos de ``credentials_path`` e ``cache_dir`` — nenhum dos
# dois é exercitado por este teste, então o caminho não precisa existir
# em disco. Mantemos o valor fora da função para evitar que o Hypothesis
# o veja como dependência de fixture function-scoped.
_FAKE_PROJECT_ROOT: Path = Path(__file__).resolve().parent / "_pbt_fake_root"


# Estratégia da Property 11 — espelha literalmente ``tasks.md §2.5``.
# O ``allow_nan=True`` e ``allow_infinity=True`` são deliberadamente
# mantidos (defaults de ``floats()``) para garantir cobertura de
# corner-cases numéricos que poderiam quebrar o ``f"{raw!r}"`` da
# mensagem de warning.
_INVALID_RPM_VALUES = st.one_of(
    st.integers(max_value=0),
    st.text(),
    st.floats(),
    st.booleans(),
    st.none(),
)


@given(invalid_rpm=_INVALID_RPM_VALUES)
@settings(max_examples=200, deadline=None)
def test_invalid_max_requests_per_minute_is_coerced_with_warning(
    invalid_rpm: object,
) -> None:
    """**Property 11**: valor inválido de ``max_requests_per_minute`` ⇒ ``None`` + warning.

    **Validates: Requirements 4.2, 8.5**

    Para qualquer dict ``data`` que declare ``max_requests_per_minute``
    com um valor fora do domínio "inteiro positivo" (``null``, ``0``,
    inteiro negativo, string, float ou ``bool``),
    ``GcvAppConfig.from_dict(data, project_root)`` produz:

    1. ``max_requests_per_minute is None`` — o rate limiter é
       silenciosamente desabilitado, equivalendo a "campo ausente"
       em termos operacionais (Requirement 8.1).
    2. ``len(config_warnings) >= 1`` — pelo menos um warning textual é
       registrado, garantindo que o ``NutritionReader`` propague a
       coerção em ``metadata.gcv_config_warnings`` da primeira
       tentativa GCV (Requirement 8.5).
    3. Cada string em ``config_warnings`` é não-vazia, menciona
       ``max_requests_per_minute`` e contém ``repr(invalid_rpm)``
       (a "string descritiva citando o valor original" exigida pelo
       design) — assim o operador consegue identificar exatamente o
       valor que ``app.json`` tinha quando a coerção ocorreu.

    O dict de entrada tem apenas a chave testada para isolar o
    invariante: outros campos do bloco ``gcv`` aplicariam seus próprios
    defaults silenciosos (Requirement 4.2) e poluiriam a verificação de
    ``len(config_warnings) >= 1``.
    """

    data = {"max_requests_per_minute": invalid_rpm}

    config = GcvAppConfig.from_dict(data, project_root=_FAKE_PROJECT_ROOT)

    # (1) Coerção para ``None``: o rate limiter fica desabilitado para
    # qualquer valor fora do domínio aceito.
    assert config.max_requests_per_minute is None

    # (2) Pelo menos um warning textual foi registrado. A coerção do
    # rate limiter é a única que o contrato exige sinalizar
    # explicitamente em ``config_warnings`` (Requirement 8.5), portanto
    # ``len >= 1`` é suficiente — não fixamos ``== 1`` porque, em
    # cenários futuros com outras coerções no mesmo dict, a invariante
    # continua válida.
    assert len(config.config_warnings) >= 1

    # (3) A string descritiva existe e referencia o campo + o valor
    # original. ``repr`` é robusto para todos os tipos sorteados pela
    # estratégia (incluindo ``nan``/``inf``/``True``/``False``/``''``).
    expected_value_marker = repr(invalid_rpm)
    matching_warnings = [
        w
        for w in config.config_warnings
        if "max_requests_per_minute" in w and expected_value_marker in w
    ]
    assert matching_warnings, (
        "Esperava ao menos um warning citando 'max_requests_per_minute' "
        f"e {expected_value_marker!r}, mas obteve: {config.config_warnings!r}"
    )
    for warning in matching_warnings:
        assert warning.strip(), (
            "Warning não pode ser uma string vazia ou só whitespace; "
            f"obteve: {warning!r}"
        )


# -----------------------------------------------------------------------
# Casos determinísticos auxiliares
# -----------------------------------------------------------------------
#
# Os exemplos abaixo fixam casos canônicos da Property 11 que a
# estratégia ``_INVALID_RPM_VALUES`` cobre estatisticamente, mas que
# vale a pena travar como regressão explícita: zero, negativo, string
# vazia, ``True``/``False``, ``None`` declarado e ``float`` positivo.
# A intenção é que, se algum branch do parser quebrar, o relatório de
# falha mostre o caso específico em vez de depender do shrinking do
# Hypothesis.
# -----------------------------------------------------------------------


def test_zero_max_requests_per_minute_is_coerced_with_warning() -> None:
    """``max_requests_per_minute=0`` (não-positivo) ⇒ coerção + warning."""

    config = GcvAppConfig.from_dict(
        {"max_requests_per_minute": 0},
        project_root=_FAKE_PROJECT_ROOT,
    )

    assert config.max_requests_per_minute is None
    assert any(
        "max_requests_per_minute" in w and "0" in w
        for w in config.config_warnings
    )


def test_negative_max_requests_per_minute_is_coerced_with_warning() -> None:
    """``max_requests_per_minute=-5`` (negativo) ⇒ coerção + warning."""

    config = GcvAppConfig.from_dict(
        {"max_requests_per_minute": -5},
        project_root=_FAKE_PROJECT_ROOT,
    )

    assert config.max_requests_per_minute is None
    assert any(
        "max_requests_per_minute" in w and "-5" in w
        for w in config.config_warnings
    )


def test_boolean_max_requests_per_minute_is_coerced_with_warning() -> None:
    """``max_requests_per_minute=True`` (bool) ⇒ coerção + warning.

    ``bool`` é subclasse de ``int`` em Python e ``True == 1`` enganaria
    a validação ``isinstance(..., int)`` se o parser não tratasse o
    branch booleano antes do numérico. Este caso trava esse contrato.
    """

    config = GcvAppConfig.from_dict(
        {"max_requests_per_minute": True},
        project_root=_FAKE_PROJECT_ROOT,
    )

    assert config.max_requests_per_minute is None
    assert any(
        "max_requests_per_minute" in w and "True" in w
        for w in config.config_warnings
    )


def test_string_max_requests_per_minute_is_coerced_with_warning() -> None:
    """``max_requests_per_minute="60"`` (string) ⇒ coerção + warning.

    Mesmo que a string seja parseável como inteiro, o contrato é de
    inteiro estrito — strings sempre disparam coerção.
    """

    config = GcvAppConfig.from_dict(
        {"max_requests_per_minute": "60"},
        project_root=_FAKE_PROJECT_ROOT,
    )

    assert config.max_requests_per_minute is None
    assert any(
        "max_requests_per_minute" in w and "'60'" in w
        for w in config.config_warnings
    )


def test_float_max_requests_per_minute_is_coerced_with_warning() -> None:
    """``max_requests_per_minute=5.0`` (float positivo) ⇒ coerção + warning.

    Floats são rejeitados mesmo quando positivos: o contrato é de
    inteiro estrito (Requirement 8.2).
    """

    config = GcvAppConfig.from_dict(
        {"max_requests_per_minute": 5.0},
        project_root=_FAKE_PROJECT_ROOT,
    )

    assert config.max_requests_per_minute is None
    assert any(
        "max_requests_per_minute" in w and "5.0" in w
        for w in config.config_warnings
    )


def test_explicit_none_max_requests_per_minute_emits_warning() -> None:
    """``max_requests_per_minute=None`` declarado ⇒ coerção + warning.

    Caso especial: ``None`` declarado explicitamente (``"max_requests_per_minute":
    null`` em ``app.json``) é diferente de "chave ausente". A Property
    11 exige warning para o primeiro caso, registrando a escolha
    consciente do operador em ``metadata.gcv_config_warnings``. A
    chave ausente cai no default silencioso da Requirement 4.2 e é
    coberta pelo teste ``test_absent_max_requests_per_minute_is_silent_default``.
    """

    config = GcvAppConfig.from_dict(
        {"max_requests_per_minute": None},
        project_root=_FAKE_PROJECT_ROOT,
    )

    assert config.max_requests_per_minute is None
    assert any(
        "max_requests_per_minute" in w and "None" in w
        for w in config.config_warnings
    )


def test_absent_max_requests_per_minute_is_silent_default() -> None:
    """Chave ``max_requests_per_minute`` ausente ⇒ ``None`` SEM warning.

    Contrapondo a Property 11: a Requirement 4.2 documenta que campos
    AUSENTES caem em defaults silenciosos. Apenas a coerção de valor
    DECLARADO (incluindo ``null`` explícito) gera warning. Este teste
    delimita a fronteira entre os dois comportamentos para evitar
    regressões futuras que confundissem "ausente" com "null declarado".
    """

    config = GcvAppConfig.from_dict({}, project_root=_FAKE_PROJECT_ROOT)

    assert config.max_requests_per_minute is None
    assert config.config_warnings == ()


def test_positive_int_max_requests_per_minute_is_kept_without_warning() -> None:
    """Inteiro positivo ⇒ valor preservado, ``config_warnings`` vazio.

    Caso "feliz" da Property 11: garante que o branch de aceitação não
    foi acidentalmente removido junto com a lógica de coerção. ``60`` é
    o valor canônico exemplificado em ``docs/CONFIG.md``.
    """

    config = GcvAppConfig.from_dict(
        {"max_requests_per_minute": 60},
        project_root=_FAKE_PROJECT_ROOT,
    )

    assert config.max_requests_per_minute == 60
    assert config.config_warnings == ()
