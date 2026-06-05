"""Gerador determinístico do fixture ``tiny_label.png``.

Este script é executado uma única vez para materializar o arquivo PNG
sintético ``tiny_label.png`` (16x16, BGR uint8) usado pelos testes da
feature GCV. A imagem é construída como um pequeno padrão com gradiente
horizontal nos canais BGR, garantindo que o PNG resultante:

- tenha dimensões exatas 16x16 pixels;
- use 3 canais (BGR) coerentes com o formato lido por
  ``imaging.io.read_image``;
- decodifique de volta para um ``np.ndarray`` de shape ``(16, 16, 3)``
  via ``cv2.imdecode``.

Não há dependência de runtime nos testes — a saída é commitada no
repositório. Re-executar este script apenas regrava bytes idênticos
porque a função geradora é determinística.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def build_tiny_label() -> np.ndarray:
    """Constrói um array BGR uint8 16x16 com um gradiente sintético."""

    array = np.zeros((16, 16, 3), dtype=np.uint8)
    # Gradiente horizontal nos canais BGR para gerar conteúdo previsível.
    for col in range(16):
        array[:, col, 0] = (col * 16) % 256          # canal B
        array[:, col, 1] = (col * 8) % 256            # canal G
        array[:, col, 2] = (255 - col * 16) % 256     # canal R
    return array


def main() -> None:
    output_path = Path(__file__).resolve().parent / "tiny_label.png"
    image = build_tiny_label()
    success, buffer = cv2.imencode(".png", image)
    if not success:  # pragma: no cover - guarda defensiva
        raise RuntimeError("cv2.imencode falhou ao codificar tiny_label.png")
    output_path.write_bytes(buffer.tobytes())
    print(f"Escrito {output_path} ({output_path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
