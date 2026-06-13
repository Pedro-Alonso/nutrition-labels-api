"""Property test P17: ``AuditRecorder`` (Null) não toca o sistema de arquivos.

Validates: Requirements 7.6

No backend REST, ``AuditRecorder`` é um no-op (``NullAuditRecorder`` — ver
``audit/recorder.py``): o construtor apenas guarda ``input_slug`` e
``manifest`` em memória, sem criar, limpar ou inspecionar nenhum diretório,
independentemente de ``clean_previous``.

A propriedade afirma que, *para qualquer* conjunto de arquivos ``F`` em
``extractions/.gcv_cache/`` e em ``extractions/<input_slug>/`` /
``images/pipeline/<input_slug>/``, após instanciar
``AuditRecorder(project_root, input_path, clean_previous=True)`` o conjunto
de arquivos permanece idêntico a ``F`` (mesmos paths, mesmos bytes) — o
construtor é puramente em memória.

Esta suíte falsifica a invariante exercendo várias topologias possíveis do
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
    """**Property 17**: ``AuditRecorder`` (Null) não apaga nada do disco.

    Validates: Requirements 7.6
    """

    # ``tmp_path_factory`` é session-scoped; ``mktemp`` cria um diretório
    # único a cada exemplo do Hypothesis para isolar iterações sucessivas.
    project_root = tmp_path_factory.mktemp("p17_audit_cache_preservation")
    input_slug = "label_subject"

    # Estrutura mínima — não é exigida pelo Null recorder, mas reflete o
    # layout real de um projeto para tornar o teste representativo.
    (project_root / "extractions").mkdir(parents=True, exist_ok=True)
    (project_root / "images" / "pipeline").mkdir(parents=True, exist_ok=True)

    # Seed do ``cache_dir`` (irmão de ``extractions/<input_slug>/``).
    cache_dir = project_root / "extractions" / ".gcv_cache"
    cache_snapshot_before = _materialize_cache(cache_dir, entries)

    # Seed de ``extractions/<input_slug>/`` e ``images/pipeline/<input_slug>/``
    # com conteúdo pré-existente — o Null recorder não deve tocar nenhum
    # desses arquivos, com ``clean_previous=True`` ou não.
    extraction_dir = project_root / "extractions" / input_slug
    pipeline_dir = project_root / "images" / "pipeline" / input_slug
    extraction_dir.mkdir(parents=True, exist_ok=True)
    pipeline_dir.mkdir(parents=True, exist_ok=True)
    (extraction_dir / "old_artifact.txt").write_bytes(b"stale-extraction")
    (extraction_dir / "nested").mkdir()
    (extraction_dir / "nested" / "deeper.json").write_bytes(b'{"old": true}')
    (pipeline_dir / "old_stage.png").write_bytes(b"\x89PNG\r\n\x1a\nfake-old")

    extraction_snapshot_before = _snapshot_cache(extraction_dir)
    extraction_nested_before = (extraction_dir / "nested" / "deeper.json").read_bytes()
    pipeline_snapshot_before = _snapshot_cache(pipeline_dir)

    # Caminho do "subject" que dá origem ao slug — não precisa existir
    # como arquivo real porque o ``AuditRecorder`` apenas usa ``stem``.
    input_path = project_root / "subjects" / f"{input_slug}.png"

    # Instancia com ``clean_previous=True``: o construtor do Null recorder
    # aceita o parâmetro sem erro, mas não realiza nenhuma limpeza.
    recorder = AuditRecorder(
        project_root=project_root,
        input_path=input_path,
        clean_previous=True,
    )
    assert recorder.input_slug == input_slug

    # Nada em ``extractions/<input_slug>/`` ou ``images/pipeline/<input_slug>/``
    # foi removido ou alterado.
    assert (extraction_dir / "old_artifact.txt").exists()
    assert (extraction_dir / "nested" / "deeper.json").exists()
    assert (pipeline_dir / "old_stage.png").exists()
    assert _snapshot_cache(extraction_dir) == extraction_snapshot_before
    assert (extraction_dir / "nested" / "deeper.json").read_bytes() == extraction_nested_before
    assert _snapshot_cache(pipeline_dir) == pipeline_snapshot_before

    # Invariante de P17: o ``cache_dir`` permanece byte-idêntico ao snapshot.
    cache_snapshot_after = _snapshot_cache(cache_dir)
    assert cache_snapshot_after.keys() == cache_snapshot_before.keys(), (
        "conjunto de arquivos do cache_dir mudou: "
        f"faltando={set(cache_snapshot_before) - set(cache_snapshot_after)} "
        f"extras={set(cache_snapshot_after) - set(cache_snapshot_before)}"
    )
    for path, expected_bytes in cache_snapshot_before.items():
        assert cache_snapshot_after[path] == expected_bytes, (
            f"conteúdo de {path.name} foi alterado pelo construtor do AuditRecorder"
        )

    # O próprio diretório também precisa sobreviver, mesmo quando
    # ``entries`` está vazio (cache pré-existente porém sem entradas).
    assert cache_dir.is_dir(), "cache_dir foi removido pelo AuditRecorder"
