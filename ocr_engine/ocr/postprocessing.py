"""Pós-processamento léxico de OCR de rótulos nutricionais.

Combina substituições diretas de símbolos com correção por Levenshtein e regras
de domínio por coluna.

Saída do CellBasedPipeline:  linhas com \t entre colunas (nome, quantidade, %VD…)
Saída do LinearPipeline:     linhas de texto livre

O método `postprocess()` detecta o formato e despacha para o branch correto.

Branch tabular — regras por coluna
───────────────────────────────────
• Coluna de nome: Levenshtein para o nutriente mais próximo → nome canônico.
• Colunas restantes: papel detectado por conteúdo, não por índice fixo.
  - Puro alfabético correspondente à unidade esperada → fragmento de unidade.
  - Número decimal (vírgula) ou número + unidade → quantidade.
  - Inteiro 0-999 depois da quantidade já estabelecida → %VD.
  - Asterisco(s) ou lixo → %VD inválido ("**").
• Quantidade: regex (\d+(?:[.,]\d+)?)\s*([A-Za-z]+)? sobre célula pré-traduzida.
  Se número ≥ 100 sem unidade e termina em "9"/"q" → strip último dígito + unidade
  (corrige confusão visual OCR "g" → "9", gerando "159" no lugar de "15 g").
• %VD: aceita inteiro puro; rejeita decimais (vírgula) e strings sem dígitos → "**".
"""

from __future__ import annotations

import re
import unicodedata

from .levenshtein import levenshtein_distance


# Substituições de caracteres OCR — aplicadas à célula ANTES do parse numérico.
# Inclui l→1 (corrigido para kcal via Levenshtein posterior na unidade).
_OCR_DIGIT_TABLE = str.maketrans(
    "ÚúUOoQlIiSZsz",
    "0000001115522"
    # Ú/ú/U → 0   O/o/Q → 0   l/I/i → 1   S/s → 5   Z/z → 2
)


class NutritionTextPostProcessor:
    _NUTRIENT_SPECS: dict[str, dict] = {
        "valor energetico": {"unit": "kcal", "display": "Valor Energético"},
        "carboidratos":     {"unit": "g",    "display": "Carboidratos"},
        "acucares totais":  {"unit": "g",    "display": "Açúcares Totais"},
        "proteinas":        {"unit": "g",    "display": "Proteínas"},
        "gorduras totais":  {"unit": "g",    "display": "Gorduras Totais"},
        "gorduras saturadas": {"unit": "g",  "display": "Gorduras Saturadas"},
        "gorduras trans":   {"unit": "g",    "display": "Gorduras Trans"},
        "fibra alimentar":  {"unit": "g",    "display": "Fibra Alimentar"},
        "sodio":            {"unit": "mg",   "display": "Sódio"},
    }

    _NUTRIENT_LINE_PATTERN = re.compile(
        r"^\s*([A-Za-zÀ-ÿ\s]+?)\s+([\d.,]+)\s*([A-Za-z%*()]+)?\s*([\d.,*]+)?\s*$",
    )

    # -----------------------------------------------------------------------
    # Ponto de entrada
    # -----------------------------------------------------------------------

    def postprocess(self, text: str) -> str:
        lines = text.splitlines()
        corrected: list[str] = []
        for line in lines:
            if "\t" in line:
                corrected.append(self._postprocess_tabular_line(line))
            else:
                cleaned = self._normalize_symbols(line)
                cleaned = self._fix_nutrient_line_units(cleaned)
                corrected.append(cleaned)
        return "\n".join(corrected)

    # -----------------------------------------------------------------------
    # Branch tabular — saída do CellBasedPipeline (colunas separadas por \t)
    # -----------------------------------------------------------------------

    def _postprocess_tabular_line(self, line: str) -> str:
        cols = [c.strip() for c in line.split("\t")]
        if not cols or not any(cols):
            return line

        # Coluna 0: nome do nutriente
        name_raw = self._normalize_symbols(cols[0])
        label = self._extract_label(name_raw)
        closest = self._closest_nutrient(label) if label else None
        if closest:
            display_name = self._NUTRIENT_SPECS[closest]["display"]
            expected_unit = self._NUTRIENT_SPECS[closest]["unit"]
        else:
            display_name = name_raw
            expected_unit = None

        if len(cols) == 1:
            return display_name

        # Colunas restantes: detecta quantidade e %VD por conteúdo, não por índice.
        qty, vd = self._parse_remaining_cols(cols[1:], expected_unit)

        parts = [display_name]
        if qty:
            parts.append(qty)
        if vd is not None:
            parts.append(vd)
        return "\t".join(parts)

    @classmethod
    def _parse_remaining_cols(
        cls,
        cols: list[str],
        expected_unit: str | None,
    ) -> tuple[str, str | None]:
        """Interpreta colunas restantes e devolve (quantidade, %VD).

        Não assume índice fixo: detecta o papel de cada coluna pelo conteúdo.
        """
        qty_number = ""   # parte numérica da quantidade
        qty_unit   = ""   # unidade da quantidade
        vd: str | None = None

        for col in cols:
            if not col:
                continue
            stripped = col.strip()

            # --- %VD com asterisco ou aspas curvas (antes de qualquer tradução) ---
            if "*" in stripped or re.search(r"[''""»«]", stripped):
                vd = "**"
                continue

            # Fix 1: decimal com trailing 9 (confusão OCR "g"→"9").
            # Ex: "1,49" → "1,4g" antes do translate, para que o regex numérico
            # reconheça número + unidade corretamente.
            if expected_unit == "g" and re.fullmatch(r"\d+[.,]\d*9", stripped):
                stripped = stripped[:-1] + "g"

            # Tradução OCR aplicada ANTES de qualquer outra verificação.
            # Isso garante que "S"→"5" não passe pelo teste de unidade pura,
            # e que "Ug"/"lg" não sejam confundidos com fragmentos de unidade.
            translated = stripped.translate(_OCR_DIGIT_TABLE)

            # Fix 2: Valor Energético — extrai parte kcal, ignora kJ e resto.
            # "75 kcal = 315kJ" → qty_number="75", qty_unit="kcal".
            if expected_unit == "kcal":
                m_kcal = re.search(r"(\d+(?:[.,]\d+)?)\s*kca[l1]", translated, re.IGNORECASE)
                if m_kcal:
                    qty_number = m_kcal.group(1)
                    qty_unit = "kcal"
                    continue

            # --- fragmento de unidade pura ---
            # Verifica o valor JÁ TRADUZIDO: "g" permanece "g", "S" vira "5" (não é unidade).
            # O grupo de unidade aceita [A-Za-z0-9] pois translate pode gerar "kca1" de "kcal".
            if re.fullmatch(r"[A-Za-z]+", translated) and expected_unit:
                dist = levenshtein_distance(translated.lower(), expected_unit.lower())
                if dist <= 2:
                    # É a unidade da quantidade anterior — anexa e segue
                    if qty_number and not qty_unit:
                        qty_unit = expected_unit
                    continue

            # --- tenta extrair número (decimal br) + unidade ---
            # [A-Za-z0-9]+ na unidade aceita "kca1" (artefato de l→1 no translate).
            m = re.match(
                r"^(\d+(?:[.,]\d+)?)\s*([A-Za-z0-9]+)?\s*$",
                translated,
            )
            if m:
                num_str  = m.group(1)
                unit_str = (m.group(2) or "").strip()

                if unit_str:
                    # Número + unidade: definitivamente quantidade
                    if expected_unit and levenshtein_distance(unit_str.lower(), expected_unit.lower()) <= 2:
                        unit_str = expected_unit
                    elif cls._is_trailing_unit_confusion(num_str, expected_unit):
                        num_str  = num_str[:-1]
                        unit_str = expected_unit or unit_str
                    qty_number = num_str
                    qty_unit   = unit_str

                elif qty_number:
                    # Já temos quantidade → inteiro isolado é %VD
                    vd = translated if re.fullmatch(r"\d{1,3}", translated) else cls._fix_vd_cell(col)

                else:
                    # Sem quantidade ainda → é a parte numérica da quantidade
                    qty_number = num_str

            else:
                # Não parseável como número — se já temos quantidade, trata como %VD lixo
                if qty_number and vd is None:
                    vd = cls._fix_vd_cell(col)

        # --- monta string de quantidade ---
        if qty_number:
            if not qty_unit and expected_unit:
                if cls._is_trailing_unit_confusion(qty_number, expected_unit):
                    qty_number = qty_number[:-1]
                qty_unit = expected_unit
            qty = f"{qty_number} {qty_unit}".strip() if qty_unit else qty_number
        else:
            qty = ""

        return qty, vd

    @staticmethod
    def _is_trailing_unit_confusion(num_str: str, expected_unit: str | None) -> bool:
        """Verdadeiro se o último caractere é provavelmente um 'g' lido como '9'.

        Ex: "159" onde o Tesseract leu "g" como "9" → strip → "15" + "g".
        Só aplicado quando a unidade esperada é "g" e o número ≥ 100 para evitar
        falsos positivos em "9 g" (legítimo) ou "19 g".
        """
        if expected_unit != "g":
            return False
        if not num_str or "," in num_str:
            return False
        if num_str[-1] not in "9q":
            return False
        try:
            return int(num_str) >= 100
        except ValueError:
            return False

    @staticmethod
    def _fix_vd_cell(cell: str | None) -> str | None:
        """%VD deve ser inteiro 0-999 ou '**'.

        Rejeita decimais (vírgula) para não confundir com quantidades.
        Qualquer lixo restante (incluindo '.»') é mapeado para '**'.
        Retorna None para células vazias (linha sem %VD).
        """
        if cell is None:
            return None
        stripped = cell.strip()
        if not stripped:
            return None

        # Asterisco(s) → %VD não estabelecido
        if "*" in stripped or re.search(r"[''""»«]", stripped):
            return "**"

        # Traduz ruído de dígito e verifica se é inteiro limpo
        candidate = stripped.translate(_OCR_DIGIT_TABLE)

        # Rejeita decimais (contém vírgula) — provavelmente quantidade no lugar errado
        if "," in candidate:
            return None

        if re.fullmatch(r"\d{1,3}", candidate):
            return candidate

        # Extrai dígitos embutidos (ex: "2%", "4 ")
        digits = re.search(r"\d{1,3}", candidate)
        if digits:
            return digits.group(0)

        # Conteúdo irreconhecível
        return "**"

    # -----------------------------------------------------------------------
    # Branch linear — saída do LinearPipeline (texto livre, sem tabs)
    # -----------------------------------------------------------------------

    def _fix_nutrient_line_units(self, line: str) -> str:
        line = self._strip_line_noise(line)
        if not line.strip():
            return line

        nutrient_name = self._extract_label(line)
        if not nutrient_name:
            return line
        closest = self._closest_nutrient(nutrient_name)
        if closest is None:
            return line
        if closest == "valor energetico":
            return self._fix_energetic_line(line)

        parsed = self._NUTRIENT_LINE_PATTERN.match(line)
        if not parsed:
            return line

        _, value, unit_token, vd_token = parsed.groups()
        expected_unit = self._NUTRIENT_SPECS[closest]["unit"]
        display_name  = self._NUTRIENT_SPECS[closest]["display"]
        vd = self._normalize_vd_token(vd_token)

        if unit_token and self._is_probable_unit(unit_token, expected_unit):
            fixed_value = self._fix_value_unit_collision(value, expected_unit, vd is not None)
            if fixed_value is not None:
                value = fixed_value
            return self._rebuild_line(display_name, value, expected_unit, vd)

        fixed_value = self._fix_value_unit_collision(value, expected_unit, vd is not None)
        if fixed_value is not None:
            value = fixed_value
        return self._rebuild_line(display_name, value, expected_unit, vd)

    def _fix_energetic_line(self, line: str) -> str:
        line = re.sub(r"\bk\s*[\)\]\|]?\b", "kJ", line, flags=re.IGNORECASE)
        line = re.sub(r"k\s?j", "kJ", line, flags=re.IGNORECASE)
        match = re.search(
            r"(\d+[\.,]?\d*)\s*kcal\s*=?\s*(\d+[\.,]?\d*)\s*kJ\s*(\d{1,2})?",
            line,
            flags=re.IGNORECASE,
        )
        if match:
            kcal, kj, vd = match.groups()
            tail = f" {vd}" if vd else ""
            return f"Valor Energético {kcal}kcal={kj}kJ{tail}".strip()
        return line

    # -----------------------------------------------------------------------
    # Helpers comuns
    # -----------------------------------------------------------------------

    @staticmethod
    def _normalize_symbols(line: str) -> str:
        replacements = {
            "k]": "kJ", "k|": "kJ", "k)": "kJ",
            "%ND()": "%VD(*)", "AVD()": "%VD(*)", "WVD()": "%VD(*)",
            "soDa": "sopa",
        }
        out = line
        for src, dst in replacements.items():
            out = out.replace(src, dst)
        out = re.sub(
            r"Quantidade\s+por\s+porc[aã]o\s+(WVD|%VD)\(\)",
            "Quantidade por porção %VD(*)",
            out,
            flags=re.IGNORECASE,
        )
        return re.sub(r"\s+", " ", out).strip()

    def _closest_nutrient(self, label: str) -> str | None:
        norm = _normalize(label)
        if not norm:
            return None
        best_name, best_dist = None, 999
        for nutrient in self._NUTRIENT_SPECS:
            dist = levenshtein_distance(norm, nutrient)
            if dist < best_dist:
                best_dist = dist
                best_name = nutrient
        if best_name is None or best_dist > 6:
            return None
        return best_name

    @staticmethod
    def _extract_label(line: str) -> str:
        match = re.match(r"\s*([A-Za-zÀ-ÿ\s]+)", line)
        if not match:
            return ""
        return match.group(1).strip()

    @staticmethod
    def _strip_line_noise(line: str) -> str:
        cleaned = line.replace("“", " ").replace("”", " ").replace("|", " ")
        return re.sub(r"\s+", " ", cleaned).strip()

    @staticmethod
    def _normalize_vd_token(vd_token: str | None) -> str | None:
        if not vd_token:
            return None
        if re.search(r"\*{1,2}", vd_token):
            return "**"
        match = re.search(r"\d{1,2}", vd_token)
        return match.group(0) if match else None

    @staticmethod
    def _is_probable_unit(token: str, expected: str) -> bool:
        normalized = token.lower().replace("|", "l").replace("]", "j")
        if expected == "g":
            return normalized in {"g", "9", "q", "og", "0g", "cg"} or levenshtein_distance(normalized, "g") <= 1
        if expected == "mg":
            return normalized in {"mg", "m9", "mq"} or levenshtein_distance(normalized, "mg") <= 1
        if expected == "kcal":
            return levenshtein_distance(normalized, "kcal") <= 2
        return levenshtein_distance(normalized, expected) <= 1

    @staticmethod
    def _fix_value_unit_collision(value: str, expected_unit: str, has_vd: bool) -> str | None:
        if expected_unit != "g":
            return None
        decimal_match = re.fullmatch(r"(\d+)([\.,])(\d{2})", value)
        if decimal_match:
            integer_part, sep, decimals = decimal_match.groups()
            if decimals.endswith("9"):
                return f"{integer_part}{sep}{decimals[0]}"
        if not has_vd:
            if any(sep in value for sep in ",."):
                return None
            if len(value) < 3 or not value.isdigit():
                return None
            if value[-1].lower() not in {"9", "q"}:
                return None
            if int(value) < 100:
                return None
            return value[:-1]
        if any(sep in value for sep in ",."):
            return None
        if len(value) < 2:
            return None
        if value[-1].lower() not in {"9", "q"}:
            return None
        return value[:-1]

    @staticmethod
    def _rebuild_line(name: str, value: str, unit: str, vd_token: str | None) -> str:
        parts = [name, value, unit]
        if vd_token:
            parts.append(vd_token)
        return re.sub(r"\s+", " ", " ".join(parts).strip())


def _normalize(text: str) -> str:
    no_accents = "".join(
        ch for ch in unicodedata.normalize("NFD", text) if unicodedata.category(ch) != "Mn"
    )
    lowered = no_accents.lower()
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s]", "", lowered)).strip()
