"""Camada de operações de imagem reutilizáveis.

Centraliza todo Processamento Digital de Imagens (PDI) do projeto em funções puras,
para que qualquer módulo (leitura de tabela, leitura de texto corrido, futura leitura
de ingredientes) consuma as mesmas primitivas sem duplicação.
"""

from .io import read_image, write_image
from . import operations, morphology
from .roi import RoiDetectionConfig, RoiDetector

__all__ = [
    "read_image",
    "write_image",
    "operations",
    "morphology",
    "RoiDetectionConfig",
    "RoiDetector",
]
