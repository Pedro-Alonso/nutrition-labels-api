"""Pós-processamento de texto OCR produzido pelo CloudVisionPipeline.

Diferente do NutritionTextPostProcessor (projetado para corrigir erros típicos
do Tesseract), este módulo assume que o texto GCV já é de alta qualidade e
realiza apenas:

- Normalização do nome do nutriente via Levenshtein (inclusive quando o GCV
  fragmentou o nome em duas colunas adjacentes).
- Preservação de todas as colunas de dados intermediárias sem alteração.
- Normalização da última coluna como %VD (inteiro 1–3 dígitos ou "**");
  sequências de 4+ dígitos são descartadas como %VD garbled.
- Linhas sem tabs: retornadas inalteradas.

O que NÃO é feito:
- Substituições _OCR_DIGIT_TABLE (O→0, S→5 etc.) — texto GCV não precisa.
- Regexes de correção de unidade específicas para Tesseract.
- Alteração de valores numéricos nas colunas do meio.
"""

from __future__ import annotations

import re

from ocr.levenshtein import levenshtein_distance
from ocr.postprocessing import NutritionTextPostProcessor, _normalize


_QUOTES_RE = re.compile(r"[''\"\"»«]")
_VD_RE = re.compile(r"^\d{1,3}$")
_VD_GARBLED_RE = re.compile(r"^\d{4,}$")


class GcvTablePostProcessor:
    _NUTRIENT_SPECS = NutritionTextPostProcessor._NUTRIENT_SPECS

    def postprocess(self, text: str) -> str:
        lines = text.splitlines()
        return "\n".join(
            self._process_tabular(line) if "\t" in line else line
            for line in lines
        )

    def _process_tabular(self, line: str) -> str:
        cols = [c.strip() for c in line.split("\t")]
        if not any(cols):
            return line

        # Identificar nome do nutriente — pode estar fragmentado em 2 colunas.
        # Usa distância para preferir a opção mais confiante: se a fusão de
        # duas colunas dá distância menor, prefere o merge.
        nutrient_key, dist1 = self._closest_nutrient_with_dist(cols[0])
        name_cols = 1

        if len(cols) > 1:
            combined = cols[0] + " " + cols[1]
            nutrient_key2, dist2 = self._closest_nutrient_with_dist(combined)
            if nutrient_key2 is not None and dist2 < dist1:
                nutrient_key = nutrient_key2
                name_cols = 2

        display_name = (
            self._NUTRIENT_SPECS[nutrient_key]["display"]
            if nutrient_key
            else cols[0]
        )

        data = cols[name_cols:]
        if not data:
            return display_name

        # Última coluna: normalizar como %VD.
        last = data[-1]
        vd: str | None = None
        if "*" in last or _QUOTES_RE.search(last):
            vd = "**"
            data = data[:-1]
        elif _VD_RE.match(last):
            vd = last
            data = data[:-1]
        elif _VD_GARBLED_RE.match(last):
            # 4+ dígitos consecutivos = %VD garbled (ex.: "45272") → descartar.
            data = data[:-1]
        # else: coluna com letras/vírgula = dado legítimo, mantém em data.

        parts = [display_name] + data
        if vd is not None:
            parts.append(vd)
        return "\t".join(parts)

    def _closest_nutrient_with_dist(self, label: str) -> tuple[str | None, int]:
        norm = _normalize(label)
        if not norm:
            return None, 999
        best_name, best_dist = None, 999
        for nutrient in self._NUTRIENT_SPECS:
            dist = levenshtein_distance(norm, nutrient)
            if dist < best_dist:
                best_dist = dist
                best_name = nutrient
        if best_name is None or best_dist > 6:
            return None, 999
        return best_name, best_dist

    def _closest_nutrient(self, label: str) -> str | None:
        key, _ = self._closest_nutrient_with_dist(label)
        return key
