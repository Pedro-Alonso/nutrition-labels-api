from __future__ import annotations

import sys
from pathlib import Path

_ENGINE_ROOT = Path(__file__).parent

# Adiciona ocr_engine/ ao sys.path para que os módulos copiados do monolito
# (nutrition/, ocr/, imaging/, ingredients/, audit/) sejam importáveis via
# seus nomes originais sem necessidade de alterar nenhum import nos arquivos copiados.
if str(_ENGINE_ROOT) not in sys.path:
    sys.path.insert(0, str(_ENGINE_ROOT))

from ocr_engine.nutrition.reader import NutritionReader, build_default_reader  # noqa: E402


def build_reader() -> NutritionReader:
    """Constrói e retorna o NutritionReader configurado para o backend.

    Deve ser chamado uma única vez no startup da aplicação (via lifespan do FastAPI).
    O resultado é reutilizado como singleton em todas as requisições.
    """
    return build_default_reader(_ENGINE_ROOT)
