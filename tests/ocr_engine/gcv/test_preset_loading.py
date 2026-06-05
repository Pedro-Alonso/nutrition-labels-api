"""Testes property-based para o carregamento de GCV presets.

Cobre **Property 3** do design (``.kiro/specs/gcv-ocr-preset/design.md``):
*Categoria de GCV preset deriva do diretório*. A propriedade afirma que,
para qualquer GCV preset (``kind == "cloud_vision"``) carregado a partir
de ``config/presets/<dir>/<file>.json`` com
``dir ∈ {"table", "text", "ingredients"}``, a categoria efetiva exposta
pelo ``PresetRepository`` (``for_category(dir)`` o devolve;
``Preset.category`` retorna ``dir``) é sempre ``dir`` —
independentemente de qualquer campo declarado dentro do JSON.

Requirements validados:

- **2.2** GCV preset em ``config/presets/table/`` é classificado como
  ``table``.
- **2.3** GCV preset em ``config/presets/text/`` é classificado como
  ``text``.
- **2.4** GCV preset em ``config/presets/ingredients/`` é classificado
  como ``ingredients``.
- **2.6** O diretório de origem é a autoridade da categoria — o
  ``PresetRepository`` ignora qualquer ``kind`` ou metadado interno
  que tente reclassificar o preset.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from nutrition.presets import PresetRepository
from tests.ocr_engine.gcv.strategies import bcp47_hints


# Categorias canônicas reconhecidas pelo ``PresetRepository._load`` —
# qualquer outro nome de diretório é silenciosamente ignorado pelo
# carregador, então restringimos o gerador a este conjunto.
_CATEGORY_DIRS: tuple[str, ...] = ("table", "text", "ingredients")

# Características aceitas para ``gcv.feature`` (mantidas em sintonia
# com ``ocr.cloud_vision.types.ALLOWED_FEATURES``). A propriedade não
# depende do valor específico, mas alimentamos os dois para garantir
# que ambos atravessam o carregador sem distinção.
_GCV_FEATURES: tuple[str, ...] = ("TEXT_DETECTION", "DOCUMENT_TEXT_DETECTION")

# Alfabeto enxuto para nomes de preset — letras minúsculas, dígitos e
# underscore, garantindo que o nome também serve como nome de arquivo
# em qualquer plataforma sem sanitização extra.
_NAME_ALPHABET = st.characters(
    whitelist_categories=(),
    whitelist_characters="abcdefghijklmnopqrstuvwxyz0123456789_",
)


@st.composite
def _gcv_preset_payload(draw: st.DrawFn, *, target_dir: str) -> dict[str, Any]:
    """Constrói um payload de GCV preset com campos potencialmente polluintes.

    O payload sempre declara ``kind == "cloud_vision"`` e os campos
    obrigatórios (``name``, ``steps``, ``ocr``). Para falsificar a
    propriedade caso o ``PresetRepository`` use o JSON como autoridade
    da categoria, embutimos pollutantes plausíveis:

    - um campo ``category`` apontando para um diretório arbitrário
      (incluindo nomes diferentes do diretório real);
    - um ``description`` contendo o nome de outra categoria;
    - um ``name`` que pode incluir o nome de outra categoria como
      sufixo.

    Esses campos NÃO são lidos pelo carregador atual; o gerador os
    inclui justamente para que qualquer regressão futura que comece a
    consultá-los seja detectada pela propriedade.
    """

    suffix_pool = sorted(set(_CATEGORY_DIRS) | {""})
    name_suffix = draw(st.sampled_from(suffix_pool))
    raw_name = draw(
        st.text(alphabet=_NAME_ALPHABET, min_size=1, max_size=12)
    )
    name = f"gcv_{raw_name}" + (f"_{name_suffix}" if name_suffix else "")

    feature = draw(st.sampled_from(_GCV_FEATURES))
    hints = draw(bcp47_hints())
    polluting_category = draw(st.sampled_from(suffix_pool))
    polluting_description_dir = draw(st.sampled_from(suffix_pool))
    priority = draw(st.integers(min_value=1, max_value=200))

    return {
        "name": name,
        "description": (
            f"Preset GCV (origem real: {target_dir}; "
            f"pollutante: {polluting_description_dir or 'nenhum'})"
        ),
        "kind": "cloud_vision",
        "priority": priority,
        "steps": [],
        "ocr": {"lang": "por", "psm": 6, "oem": 3},
        "gcv": {
            "feature": feature,
            "language_hints": list(hints) if hints else ["pt"],
            "model": None,
        },
        # Pollutante: o ``PresetRepository`` deve ignorar este campo —
        # categoria é decidida pelo diretório (Requirement 2.6).
        "category": polluting_category,
    }


def _materialize_preset_tree(
    root: Path, target_dir: str, payload: dict[str, Any]
) -> Path:
    """Cria a árvore ``presets/<dir>/`` e grava o JSON do preset.

    Cria também os diretórios irmãos (``table``, ``text``,
    ``ingredients``) vazios para reproduzir o layout canônico do
    projeto e exercitar o caminho em que ``for_category`` é chamado
    para categorias não-alvo.
    """

    for category in _CATEGORY_DIRS:
        (root / category).mkdir(parents=True, exist_ok=True)

    preset_file = root / target_dir / f"{payload['name']}.json"
    preset_file.write_text(
        json.dumps(payload, ensure_ascii=False), encoding="utf-8"
    )
    return preset_file


@given(
    target_dir=st.sampled_from(_CATEGORY_DIRS),
    data=st.data(),
)
@settings(
    max_examples=100,
    deadline=None,
    # ``tmp_path`` é function-scoped: o Hypothesis emite o health check
    # ``function_scoped_fixture`` por padrão. Cada exemplo é isolado
    # em um subdiretório único derivado de ``uuid4`` (ver corpo do
    # teste), portanto é seguro suprimir o aviso.
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
def test_gcv_preset_category_derives_from_directory(
    tmp_path: Path, target_dir: str, data: st.DataObject
) -> None:
    """**Validates: Requirements 2.2, 2.3, 2.4, 2.6**

    Property 3: para qualquer GCV preset gravado em
    ``presets/<target_dir>/<file>.json``, o ``PresetRepository``
    expõe-o sob a categoria ``target_dir``. ``Preset.category`` retorna
    ``target_dir`` independentemente de qualquer pollutante embutido no
    JSON, e o preset NÃO aparece nas demais categorias.
    """

    payload = data.draw(_gcv_preset_payload(target_dir=target_dir))

    # Isolamento por exemplo: ``tmp_path`` é compartilhado entre as
    # iterações do Hypothesis, então cada execução grava em uma
    # subárvore única.
    iteration_root = tmp_path / uuid.uuid4().hex
    presets_root = iteration_root / "presets"
    preset_file = _materialize_preset_tree(presets_root, target_dir, payload)

    repo = PresetRepository(presets_root)

    # 1. O preset aparece exatamente uma vez na categoria do diretório.
    presets_in_target = repo.for_category(target_dir)
    matches = [p for p in presets_in_target if p.source_path == preset_file]
    assert len(matches) == 1, (
        f"Preset gravado em {target_dir}/ deveria aparecer 1 vez em "
        f"for_category({target_dir!r}); presets encontrados: "
        f"{[p.name for p in presets_in_target]}"
    )
    preset = matches[0]

    # 2. ``Preset.category`` reflete o diretório, não o JSON.
    assert preset.kind == "cloud_vision"
    assert preset.category == target_dir, (
        f"Preset.category={preset.category!r} difere do diretório "
        f"{target_dir!r}; pollutante 'category' do JSON era "
        f"{payload.get('category')!r}"
    )

    # 3. O preset não vaza para as outras categorias.
    for other in _CATEGORY_DIRS:
        if other == target_dir:
            continue
        leaked = [p for p in repo.for_category(other) if p.source_path == preset_file]
        assert leaked == [], (
            f"Preset gravado em {target_dir}/ vazou para a categoria "
            f"{other!r}: {[p.name for p in leaked]}"
        )


# ---------------------------------------------------------------------------
# Property 13: PresetRepository rejeita kinds inválidos
# ---------------------------------------------------------------------------

import pytest

from ocr.cloud_vision.types import ALLOWED_KINDS
from tests.ocr_engine.gcv.strategies import kind_strings_invalid


@given(invalid_kind=kind_strings_invalid())
@settings(
    max_examples=100,
    deadline=None,
    # ``tmp_path`` é function-scoped; cada exemplo grava em um sub-path
    # único para evitar colisão entre iterações do Hypothesis.
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
def test_preset_repository_rejects_invalid_kinds(
    tmp_path: Path, invalid_kind: str
) -> None:
    """**Validates: Requirements 1.2**

    Property 13: para qualquer string ``s`` que NÃO pertença ao
    conjunto ``ALLOWED_KINDS`` (``{"linear_table", "linear_text",
    "linear_ingredient", "cell_based", "cloud_vision"}``),
    ``PresetRepository._parse`` levanta ``ValueError`` quando consome
    um JSON cujo campo ``kind`` é ``s``.

    Cobre casing alterado (``"Cloud_Vision"``), strings vazias,
    strings com whitespace nas bordas, valores notórios (``"None"``,
    ``"null"``) e strings imprimíveis arbitrárias filtradas para
    excluir os kinds válidos. O payload mantém todos os outros campos
    obrigatórios (``name``, ``steps``, ``ocr``) bem-formados, garantindo
    que o ``ValueError`` venha exclusivamente da validação de ``kind``,
    não de ``missing keys``.
    """

    # Sanidade: o gerador nunca pode produzir um kind aceito —
    # se isso acontecer, o teste é nulo.
    assert invalid_kind not in ALLOWED_KINDS, (
        f"Gerador kind_strings_invalid produziu valor aceito: {invalid_kind!r}"
    )

    iteration_root = tmp_path / uuid.uuid4().hex
    iteration_root.mkdir(parents=True, exist_ok=True)
    preset_file = iteration_root / "preset.json"

    payload: dict[str, Any] = {
        "name": "preset_invalido",
        "description": "Preset com kind inválido para Property 13",
        "kind": invalid_kind,
        "priority": 10,
        "steps": [],
        "ocr": {"lang": "por", "psm": 6, "oem": 3},
    }
    preset_file.write_text(
        json.dumps(payload, ensure_ascii=False), encoding="utf-8"
    )

    with pytest.raises(ValueError):
        PresetRepository._parse(preset_file)
