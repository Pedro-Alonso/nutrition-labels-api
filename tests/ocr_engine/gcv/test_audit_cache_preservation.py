"""Property test P17: ``AuditRecorder.clean_previous`` preserva ``cache_dir``.

Validates: Requirements 7.6

A propriedade afirma que, *para qualquer* conjunto de arquivos ``F`` em
``extractions/.gcv_cache/`` (ou em qualquer ``cache_dir`` configurado
desde que esteja **fora** de ``extractions/<input>/``), após instanciar
``AuditRecorder(project_root, input_path, clean_previous=True)`` o
conjunto de arquivos em ``cache_dir`` permanece idêntico a ``F``
(mesmos paths, mesmos bytes).

Como o design coloca o ``cache_dir`` em ``extractions/.gcv_cache/`` —
irmão de ``extractions/<input_slug>/`` e não filho — o ciclo de limpeza
do ``AuditRecorder`` (que itera apenas ``extractions/<input_slug>/`` e
``images/pipeline/<input_slug>/``) é cego ao cache por construção. Esta
suíte falsifica a invariante exercendo várias topologias possíveis do
``cache_dir`` via ``cache_states()`` em ``tests/gcv/strategies.py``,
incluindo entradas válidas, ``response`` corrompido, ``meta`` corrompido
e cache vazio.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from hypothesis import HealthCheck, given, settings

from audit.recorder import AuditRecorder
from tests.ocr_engine.gcv.strategies import cache_states


def _materialize_cache(
    cache_dir: Path, entries: dict[str, dict[str, Any]]
) -> dict[Path, bytes]:
    """Grava em disco os pares ``<sha>.json`` + ``<sha>.meta.json``.

    A representação em ``entries`` (saída de ``cache_states()``) descreve
    se o ``response`` ou o ``meta`` devem ser corrompidos: usamos isso
    para garantir que P17 vale mesmo na presença de entradas inválidas
    (P17 é estritamente sobre preservação física dos arquivos, não sobre
    a semântica do cache — esta última é objeto de P18/P19).

    Devolve um snapshot ``{path: bytes}`` para comparação byte-a-byte
    após a execução do ``AuditRecorder``.
    """

    cache_dir.mkdir(parents=True, exist_ok=True)
    snapshot: dict[Path, bytes] = {}

    for sha, entry in entries.items():
        json_path = cache_dir / f"{sha}.json"
        meta_path = cache_dir / f"{sha}.meta.json"

        if entry["response_corrupt"]:
            # Bytes não-JSON propositais: P17 deve preservar este arquivo
            # tal e qual, mesmo sendo "lixo" do ponto de vista do parser.
            response_bytes = b"<<corrupted-non-json-payload>>"
        else:
            response_bytes = json.dumps(
                entry["response_payload"], ensure_ascii=False
            ).encode("utf-8")

        if entry["meta_corrupt"]:
            meta_bytes = b"{not valid json"
        else:
            meta_bytes = json.dumps(
                {
                    "feature": entry["feature"],
                    "language_hints": list(entry["language_hints"]),
                    "image_size_bytes": entry["image_size_bytes"],
                },
                ensure_ascii=False,
            ).encode("utf-8")

        json_path.write_bytes(response_bytes)
        meta_path.write_bytes(meta_bytes)

        snapshot[json_path] = response_bytes
        snapshot[meta_path] = meta_bytes

    return snapshot


def _snapshot_cache(cache_dir: Path) -> dict[Path, bytes]:
    """Lê o estado físico atual do ``cache_dir`` em um dict path→bytes."""

    if not cache_dir.exists():
        return {}
    return {entry: entry.read_bytes() for entry in cache_dir.iterdir() if entry.is_file()}


@given(cache_states())
@settings(
    max_examples=50,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
def test_clean_previous_preserva_cache_dir(
    tmp_path_factory: pytest.TempPathFactory,
    entries: dict[str, dict[str, Any]],
) -> None:
    """**Property 17**: ``AuditRecorder.clean_previous`` preserva ``cache_dir``.

    Validates: Requirements 7.6
    """

    # ``tmp_path_factory`` é session-scoped; ``mktemp`` cria um diretório
    # único a cada exemplo do Hypothesis para isolar iterações sucessivas.
    project_root = tmp_path_factory.mktemp("p17_audit_cache_preservation")
    input_slug = "label_subject"

    # Estrutura mínima esperada pelo ``AuditRecorder``.
    (project_root / "extractions").mkdir(parents=True, exist_ok=True)
    (project_root / "images" / "pipeline").mkdir(parents=True, exist_ok=True)

    # Seed do ``cache_dir`` (irmão de ``extractions/<input_slug>/``).
    cache_dir = project_root / "extractions" / ".gcv_cache"
    snapshot_before = _materialize_cache(cache_dir, entries)

    # Seed de ``extractions/<input_slug>/`` e ``images/pipeline/<input_slug>/``
    # com conteúdo descartável: se a limpeza realmente atua, esse conteúdo
    # some — o que serve de sanity check para que o teste não passe
    # trivialmente em uma situação onde nada foi limpo.
    extraction_dir = project_root / "extractions" / input_slug
    pipeline_dir = project_root / "images" / "pipeline" / input_slug
    extraction_dir.mkdir(parents=True, exist_ok=True)
    pipeline_dir.mkdir(parents=True, exist_ok=True)
    (extraction_dir / "old_artifact.txt").write_bytes(b"stale-extraction")
    (extraction_dir / "nested").mkdir()
    (extraction_dir / "nested" / "deeper.json").write_bytes(b'{"old": true}')
    (pipeline_dir / "old_stage.png").write_bytes(b"\x89PNG\r\n\x1a\nfake-old")

    # Caminho do "subject" que dá origem ao slug — não precisa existir
    # como arquivo real porque o ``AuditRecorder`` apenas usa ``stem``.
    input_path = project_root / "subjects" / f"{input_slug}.png"

    # Instancia em modo limpeza explícita.
    AuditRecorder(
        project_root=project_root,
        input_path=input_path,
        clean_previous=True,
    )

    # Sanity check: a limpeza atuou no escopo esperado (descarta a
    # interpretação trivial em que ``_clean`` seria um no-op).
    assert not (extraction_dir / "old_artifact.txt").exists()
    assert not (extraction_dir / "nested").exists()
    assert not (pipeline_dir / "old_stage.png").exists()

    # Invariante de P17: o ``cache_dir`` permanece byte-idêntico ao snapshot.
    snapshot_after = _snapshot_cache(cache_dir)
    assert snapshot_after.keys() == snapshot_before.keys(), (
        "conjunto de arquivos do cache_dir mudou: "
        f"faltando={set(snapshot_before) - set(snapshot_after)} "
        f"extras={set(snapshot_after) - set(snapshot_before)}"
    )
    for path, expected_bytes in snapshot_before.items():
        assert snapshot_after[path] == expected_bytes, (
            f"conteúdo de {path.name} foi alterado por clean_previous"
        )

    # O próprio diretório também precisa sobreviver, mesmo quando
    # ``entries`` está vazio (cache pré-existente porém sem entradas).
    assert cache_dir.is_dir(), "cache_dir foi removido por clean_previous"
