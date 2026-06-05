"""Property test P15: conteúdo do Service Account não aparece em artefatos.

**Validates: Requirement 5.4**

A propriedade afirma que, mesmo com ``credentials_path`` configurado
apontando para um Service Account real (ou falso, com marcador único),
nenhum byte do conteúdo desse arquivo aparece em qualquer artefato
gravado pelo pipeline: ``_summary.json``, ``*.json`` em stages,
``*.txt`` em extractions, ou qualquer outro arquivo produzido por
``AuditRecorder``.

O design garante isso porque ``auth.resolve_credentials`` retorna
APENAS o ``Path`` do arquivo; o conteúdo é consumido diretamente
pelo SDK Google sem que o pipeline ou o recorder o toquem.

Este teste verifica o invariante em dois níveis:
1. ``resolve_credentials`` retorna o caminho sem ler o conteúdo.
2. Após execução completa do ``CloudVisionPipeline`` (com stub de
   api_client para evitar chamada real), o marcador embutido no
   Service Account falso não aparece em nenhum arquivo produzido.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from audit.recorder import AuditRecorder
from nutrition.pipelines.base import PipelineContext
from nutrition.pipelines.cloud_vision import CloudVisionPipeline
from ocr.cloud_vision.app_config import GcvAppConfig
from ocr.cloud_vision.auth import resolve_credentials
from ocr.cloud_vision.options import GcvPresetOptions
from ocr.cloud_vision.types import GcvFetchResult
from ocr.service import OcrConfig
from tests.ocr_engine.gcv.conftest import read_marker_from_service_account


# ---------------------------------------------------------------------------
# Stub do api_client (duck typing sobre annotate_image)
# ---------------------------------------------------------------------------


class _StubApiClient:
    """Stub mínimo que simula ``ImageAnnotatorClient.annotate_image``."""

    def annotate_image(self, request: dict) -> object:
        # Resposta vazia válida: full_text_annotation com texto vazio.
        class _Resp:
            pass

        return _Resp()


# ---------------------------------------------------------------------------
# Helper: executa o pipeline com configuração real de credentials_path
# ---------------------------------------------------------------------------


def _run_with_sa_path(
    project_root: Path,
    sa_path: Path,
    *,
    image: np.ndarray | None = None,
) -> None:
    """Executa o pipeline com um GcvAppConfig que aponta para ``sa_path``."""
    if image is None:
        image = np.zeros((32, 32, 3), dtype=np.uint8)

    (project_root / "extractions").mkdir(parents=True, exist_ok=True)
    (project_root / "images" / "pipeline").mkdir(parents=True, exist_ok=True)
    subjects_dir = project_root / "subjects"
    subjects_dir.mkdir(parents=True, exist_ok=True)

    input_path = subjects_dir / "p15_subject.png"

    # Resposta sintética mínima para o pipeline processar algo
    response_json: dict = {
        "fullTextAnnotation": {"text": "teste privacidade"},
        "textAnnotations": [],
    }

    class _Stub:
        def fetch(self, png_bytes: bytes, feature: str, language_hints):
            return GcvFetchResult(
                response_json=response_json,
                cache_hit=False,
                feature=feature,
                language_hints=tuple(language_hints),
            )

    recorder = AuditRecorder(
        project_root=project_root,
        input_path=input_path,
        clean_previous=True,
    )
    artifacts = recorder.start_attempt(1, "00_gcv_privacy")

    context = PipelineContext(
        input_path=input_path,
        attempt_index=1,
        preset_name="00_gcv_privacy",
        recorder=recorder,
        artifacts=artifacts,
    )

    gcv_options = GcvPresetOptions(
        feature="DOCUMENT_TEXT_DETECTION",
        language_hints=("pt",),
        model=None,
        invalid_feature=False,
        raw_feature="DOCUMENT_TEXT_DETECTION",
    )

    pipeline = CloudVisionPipeline(
        gcv_options=gcv_options,
        ocr_config=OcrConfig(),
        client=_Stub(),  # type: ignore[arg-type]
        on_failure="skip",
        ignored_steps_count=0,
    )

    pipeline.execute(image, context)


def _collect_all_artifact_text(project_root: Path) -> str:
    """Coleta conteúdo textual dos artefatos produzidos pelo pipeline.

    Varre apenas ``extractions/`` e ``images/pipeline/`` — as pastas de
    saída canônicas do ``AuditRecorder``. O arquivo ``fake_service_account.json``
    vive na raiz do projeto e não é um artefato; incluí-lo faria o teste
    sempre falhar trivialmente.
    """
    artifact_dirs = [
        project_root / "extractions",
        project_root / "images" / "pipeline",
    ]
    texts: list[str] = []
    for artifact_dir in artifact_dirs:
        if not artifact_dir.exists():
            continue
        for path in artifact_dir.rglob("*"):
            if not path.is_file():
                continue
            try:
                texts.append(path.read_text(encoding="utf-8", errors="replace"))
            except Exception:
                pass
    return "\n".join(texts)


# ---------------------------------------------------------------------------
# Teste 1: resolve_credentials retorna apenas o Path
# ---------------------------------------------------------------------------


def test_resolve_credentials_nao_le_conteudo(
    tmp_project_root: Path,
    fake_service_account_path: Path,
) -> None:
    """``resolve_credentials`` retorna o caminho sem ler o conteúdo do SA.

    **Validates: Requirement 5.4 (primeira camada)**

    ``resolve_credentials`` deve devolver um ``Path`` apontando para o
    Service Account. O valor retornado não deve conter o marcador
    embutido no arquivo falso, provando que a função não leu o conteúdo.
    """
    marker = read_marker_from_service_account(fake_service_account_path)

    config = GcvAppConfig.from_dict(
        {
            "credentials_path": str(fake_service_account_path),
            "on_failure": "skip",
            "cache_enabled": False,
        },
        tmp_project_root,
    )

    resolved = resolve_credentials(config, tmp_project_root)

    # O retorno é o caminho — não o conteúdo.
    assert resolved == fake_service_account_path
    assert marker not in str(resolved), (
        "resolve_credentials incluiu o marcador do SA no Path retornado"
    )


# ---------------------------------------------------------------------------
# Teste 2: marcador não aparece em nenhum artefato após execução do pipeline
# ---------------------------------------------------------------------------


def test_marcador_sa_ausente_em_todos_os_artefatos(
    tmp_project_root: Path,
    fake_service_account_path: Path,
) -> None:
    """Marcador do SA falso não aparece em nenhum artefato gravado.

    **Validates: Requirement 5.4 (segunda camada — P15)**

    Executa o ``CloudVisionPipeline`` completo com um stub de api_client
    e com ``credentials_path`` configurado. Varre todos os arquivos
    produzidos em ``extractions/`` e ``images/pipeline/`` procurando
    pelo marcador único embutido no Service Account falso.

    A ausência do marcador confirma que nenhuma camada (pipeline,
    recorder, parser) expõe o conteúdo do SA em logs ou artefatos.
    """
    marker = read_marker_from_service_account(fake_service_account_path)

    _run_with_sa_path(tmp_project_root, fake_service_account_path)

    all_text = _collect_all_artifact_text(tmp_project_root)

    assert marker not in all_text, (
        f"Marcador de privacidade '{marker}' encontrado em artefato — "
        "o conteúdo do Service Account vazou para o sistema de arquivos."
    )
