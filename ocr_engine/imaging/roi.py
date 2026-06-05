"""Detecção de Região de Interesse (ROI).

Usa OpenCV DNN com MobileNet SSD (TensorFlow ou Caffe) e cai para uma heurística
baseada em contornos quando o modelo não está disponível. A intenção é isolar a
área da tabela nutricional ou do bloco de texto antes das etapas de PDI.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re

import cv2
import numpy as np


COCO_FALLBACK_CLASSES: tuple[str, ...] = (
    "background",
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
)


@dataclass(slots=True)
class RoiDetectionConfig:
    prototxt_path: Path | None = None
    weights_path: Path | None = None
    pb_path: Path | None = None
    pbtxt_path: Path | None = None
    confidence_threshold: float = 0.2
    target_class_names: tuple[str, ...] = field(
        default_factory=lambda: ("diningtable", "tvmonitor", "bottle", "tabela_nutricional")
    )
    use_contour_fallback: bool = True


class RoiDetector:
    """Detecta e recorta a ROI mais provável. Retorna a imagem original em último caso."""

    def __init__(self, config: RoiDetectionConfig) -> None:
        self.config = config
        self._backend: str = "none"
        self._tf_label_map: dict[int, str] = {}
        self._net = self._load_network()

    # ------------------------------ API pública ------------------------------

    def detect(
        self,
        image_bgr: np.ndarray,
        category_hint: str | None = None,
    ) -> np.ndarray:
        """Detecta e recorta a ROI.

        Args:
            image_bgr: Imagem de entrada (BGR).
            category_hint: ``"table"`` usa detecção por grade; ``"text"`` ou
                ``"ingredient"`` usa text-blobbing morfológico (a lista de
                ingredientes não tem linhas de grade para o algoritmo de grid
                achar). ``None`` tenta grade e cai para contornos genéricos.
        """
        is_text = category_hint in ("text", "ingredient")

        if self._net is not None:
            box = self._best_detection_box(image_bgr)
            if box is not None:
                if is_text:
                    return self._crop_with_padding(image_bgr, box, 0.03)
                x1, y1, x2, y2 = box
                crop = image_bgr[y1:y2, x1:x2]
                return self._refine_to_table_region(crop)

        if is_text:
            rect = self._text_blob_roi(image_bgr)
            if rect is not None:
                return self._crop_with_padding(image_bgr, rect, 0.03)
            return image_bgr

        return self._contour_fallback(image_bgr)

    # ------------------------------ carregamento ------------------------------

    def _load_network(self) -> cv2.dnn.Net | None:
        if self.config.pb_path and self.config.pb_path.exists():
            try:
                self._backend = "tensorflow"
                if self.config.pbtxt_path and self.config.pbtxt_path.exists():
                    self._tf_label_map = self._parse_label_map(self.config.pbtxt_path)
                    return cv2.dnn.readNetFromTensorflow(
                        str(self.config.pb_path),
                        str(self.config.pbtxt_path),
                    )
                return cv2.dnn.readNetFromTensorflow(str(self.config.pb_path))
            except cv2.error:
                self._backend = "none"

        if self.config.prototxt_path and self.config.weights_path:
            if self.config.prototxt_path.exists() and self.config.weights_path.exists():
                self._backend = "caffe"
                return cv2.dnn.readNetFromCaffe(
                    str(self.config.prototxt_path),
                    str(self.config.weights_path),
                )

        return None

    # ------------------------------ detecção SSD ------------------------------

    def _best_detection_box(self, image_bgr: np.ndarray) -> tuple[int, int, int, int] | None:
        h, w = image_bgr.shape[:2]
        blob = self._make_blob(image_bgr)
        self._net.setInput(blob)
        detections = self._net.forward()

        target_labels = {self._normalize_label(n) for n in self.config.target_class_names}
        best_box: tuple[int, int, int, int] | None = None
        best_confidence = -1.0

        for i in range(detections.shape[2]):
            confidence = float(detections[0, 0, i, 2])
            if confidence < self.config.confidence_threshold:
                continue
            class_id = int(detections[0, 0, i, 1])
            if class_id < 0:
                continue
            class_name = self._class_name(class_id)
            if target_labels and class_name and self._normalize_label(class_name) not in target_labels:
                continue

            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = box.astype("int")
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 <= x1 or y2 <= y1:
                continue
            if float((x2 - x1) * (y2 - y1)) < 0.04 * float(w * h):
                continue
            if confidence > best_confidence:
                best_confidence = confidence
                best_box = (int(x1), int(y1), int(x2), int(y2))

        return best_box

    def _make_blob(self, image_bgr: np.ndarray) -> np.ndarray:
        if self._backend == "tensorflow":
            return cv2.dnn.blobFromImage(image_bgr, size=(300, 300), swapRB=True, crop=False)
        return cv2.dnn.blobFromImage(
            cv2.resize(image_bgr, (300, 300)),
            scalefactor=0.007843,
            size=(300, 300),
            mean=127.5,
        )

    def _class_name(self, class_id: int) -> str | None:
        if self._backend == "tensorflow":
            return self._tf_label_map.get(class_id)
        if 0 <= class_id < len(COCO_FALLBACK_CLASSES):
            return COCO_FALLBACK_CLASSES[class_id]
        return None

    @staticmethod
    def _normalize_label(label: str) -> str:
        return re.sub(r"[^a-z0-9]", "", label.lower())

    @staticmethod
    def _parse_label_map(pbtxt_path: Path) -> dict[int, str]:
        text = pbtxt_path.read_text(encoding="utf-8", errors="ignore")
        ids = re.findall(r"id\s*:\s*(\d+)", text)
        names = re.findall(r"(?:display_name|name)\s*:\s*['\"]([^'\"]+)['\"]", text)
        return {int(ids[i]): names[i] for i in range(min(len(ids), len(names)))}

    # ------------------------------ fallback por contornos ------------------------------

    def _contour_fallback(self, image_bgr: np.ndarray) -> np.ndarray:
        if not self.config.use_contour_fallback:
            return image_bgr

        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        grid_rect = self._detect_grid_region(gray)
        if grid_rect is not None:
            return self._crop_with_padding(image_bgr, grid_rect, pad_ratio=0.04)

        # Heurística secundária: quadrilátero grande e retangular.
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
        edges = cv2.Canny(clahe, 50, 150)
        edges = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1)

        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        h, w = image_bgr.shape[:2]
        image_area = float(h * w)
        best_rect: tuple[int, int, int, int] | None = None
        best_score = -1.0

        for contour in contours:
            perimeter = cv2.arcLength(contour, True)
            if perimeter <= 0:
                continue
            approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
            if len(approx) != 4 or not cv2.isContourConvex(approx):
                continue
            x, y, cw, ch = cv2.boundingRect(approx)
            if cw <= 0 or ch <= 0:
                continue
            area = float(cw * ch)
            ratio = area / image_area
            if ratio < 0.08 or ratio > 0.85:
                continue
            score = ratio
            if score > best_score:
                best_score = score
                best_rect = (x, y, x + cw, y + ch)

        if best_rect is None:
            return image_bgr
        return self._refine_to_table_region(self._crop_with_padding(image_bgr, best_rect, pad_ratio=0.02))

    def _text_blob_roi(
        self, image_bgr: np.ndarray
    ) -> tuple[int, int, int, int] | None:
        """Localiza o parágrafo principal de texto (ingredientes / texto corrido).

        Retorna (x1, y1, x2, y2) ou None. O padding é aplicado pelo chamador.
        """
        if not self.config.use_contour_fallback:
            return None

        h, w = image_bgr.shape[:2]
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY) if image_bgr.ndim == 3 else image_bgr

        blur = cv2.bilateralFilter(gray, 11, 75, 75)
        _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

        k_w = max(3, int(w * 0.01))
        k_h = max(10, int(h * 0.04))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_w, k_h))
        dilated = cv2.dilate(binary, kernel, iterations=2)

        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        best_rect: tuple[int, int, int, int] | None = None
        max_score = -1.0

        for cnt in contours:
            bx, by, bw, bh = cv2.boundingRect(cnt)
            if bw > w * 0.65:
                continue
            if bw < 50 or bh < 50:
                continue
            score = float(bw * bh) * bh
            if score > max_score:
                max_score = score
                best_rect = (bx, by, bx + bw, by + bh)

        return best_rect

    def _refine_to_table_region(self, roi_bgr: np.ndarray) -> np.ndarray:
        if roi_bgr.size == 0:
            return roi_bgr
        h, w = roi_bgr.shape[:2]
        if float(h * w) < 10_000:
            return roi_bgr

        gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
        grid_rect = self._detect_grid_region(gray)
        if grid_rect is None:
            return roi_bgr
        return self._crop_with_padding(roi_bgr, grid_rect, pad_ratio=0.04)

    @staticmethod
    def _crop_with_padding(
        image_bgr: np.ndarray,
        rect: tuple[int, int, int, int],
        pad_ratio: float,
    ) -> np.ndarray:
        h, w = image_bgr.shape[:2]
        x1, y1, x2, y2 = rect
        pad_x = int(pad_ratio * w)
        pad_y = int(pad_ratio * h)
        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(w, x2 + pad_x)
        y2 = min(h, y2 + pad_y)
        if x2 <= x1 or y2 <= y1:
            return image_bgr
        return image_bgr[y1:y2, x1:x2]

    @staticmethod
    def _detect_grid_region(gray: np.ndarray) -> tuple[int, int, int, int] | None:
        h, w = gray.shape[:2]
        if float(h * w) < 10_000:
            return None

        binary = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            31,
            12,
        )
        horizontal = cv2.morphologyEx(
            binary,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_RECT, (max(20, w // 6), 1)),
        )
        vertical = cv2.morphologyEx(
            binary,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(20, h // 8))),
        )
        grid = cv2.dilate(
            cv2.bitwise_or(horizontal, vertical),
            cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)),
            iterations=1,
        )
        contours, _ = cv2.findContours(grid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        image_area = float(h * w)
        best_rect: tuple[int, int, int, int] | None = None
        best_score = -1.0
        for contour in contours:
            x, y, cw, ch = cv2.boundingRect(contour)
            if cw <= 0 or ch <= 0:
                continue
            ratio = (cw * ch) / image_area
            if ratio < 0.06 or ratio > 0.75:
                continue
            density = cv2.countNonZero(grid[y : y + ch, x : x + cw]) / float(cw * ch)
            if density < 0.05:
                continue
            score = (1.2 * min(density / 0.2, 1.0)) + ratio
            if score > best_score:
                best_score = score
                best_rect = (x, y, x + cw, y + ch)
        return best_rect
