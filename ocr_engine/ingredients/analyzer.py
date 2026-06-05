"""Analisador clínico de listas de ingredientes para diabetes mellitus.

Orquestra tokenizador → matcher → relatório estruturado.

Saída (`IngredientReport`):
  - tokens_found: todos os tokens extraídos pelo OCR
  - matches: pares (token, OntologyEntry) identificados
  - unmatched: tokens sem correspondência na ontologia
  - risk_summary: contagem por nível de risco DM
  - risco_global: nível de risco mais elevado encontrado
  - clinical_alerts: alertas ordenados por prioridade clínica
  - high_risk_ingredients: ingredientes de ALTO risco encontrados
  - safe_sweeteners: edulcorantes seguros identificados (relevante para DM)

`to_dict()` produz o contrato `analysis.json` consumido pelo app mobile.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .tokenizer import tokenize
from .matcher import OntologyMatcher, MatchResult


_RISK_ORDER: dict[str, int] = {
    "ALTO": 0,
    "MODERADO-ALTO": 1,
    "MODERADO": 2,
    "BAIXO": 3,
    "SEGURO": 4,
    "BENEFICO": 5,
    "INFORMATIVO": 6,
}


def _global_risk(risk_summary: dict[str, int]) -> str:
    if not risk_summary:
        return "NENHUM"
    return min(risk_summary, key=lambda r: _RISK_ORDER.get(r, 99))


@dataclass
class IngredientReport:
    image_name: str
    tokens_found: list[str]
    matches: list[MatchResult]
    unmatched: list[str]
    risk_summary: dict[str, int]
    risco_global: str
    clinical_alerts: list[dict]
    high_risk_ingredients: list[str]
    safe_sweeteners: list[str]

    def to_dict(self) -> dict:
        """Contrato analysis.json para o app mobile."""
        return {
            "imagem": self.image_name,
            "risco_global": self.risco_global,
            "ingredientes_identificados": [
                {
                    "nome_lido": a["ingrediente"],
                    "classe": a["classe"],
                    "risco": a["risco"],
                    "alerta": a["alerta"],
                    "indice_glicemico": a["indice_glicemico"],
                    "nota_clinica": a["nota_clinica"],
                }
                for a in self.clinical_alerts
            ],
            "nao_identificados": self.unmatched,
        }

    def to_text_report(self) -> str:
        """Relatório humano para feedback_clinico.txt."""
        lines: list[str] = [
            "=== ANALISE DE INGREDIENTES — DIABETES MELLITUS ===",
            f"Imagem: {self.image_name}",
            f"Risco global: {self.risco_global}",
            f"Tokens extraidos: {len(self.tokens_found)} | "
            f"Identificados: {len(self.matches)} | "
            f"Nao classificados: {len(self.unmatched)}",
        ]

        if self.high_risk_ingredients:
            lines.append("\n[ALTO RISCO]")
            for ing in self.high_risk_ingredients:
                lines.append(f"  - {ing}")

        high_mod = [a for a in self.clinical_alerts if a["risco"] in ("ALTO", "MODERADO-ALTO")]
        if high_mod:
            lines.append("\n--- Alertas principais ---")
            for alert in high_mod:
                ig = alert["indice_glicemico"]
                ig_str = f" | IG: {ig}" if ig is not None else ""
                lines.append(f"\n[{alert['risco']}] {alert['ingrediente']}{ig_str}")
                lines.append(f"  Classe : {alert['classe']}")
                lines.append(f"  Alerta : {alert['alerta']}")
                if alert.get("fisiopatologia"):
                    lines.append(f"  Mecanismo : {alert['fisiopatologia']}")
                if alert.get("nota_clinica"):
                    lines.append(f"  Nota clínica : {alert['nota_clinica']}")

        moderate = [a for a in self.clinical_alerts if a["risco"] == "MODERADO"]
        if moderate:
            lines.append("\n--- Moderado ---")
            for a in moderate:
                ig = a["indice_glicemico"]
                ig_str = f" | IG: {ig}" if ig is not None else ""
                lines.append(f"  - {a['ingrediente']} ({a['classe']}){ig_str}")
                if a.get("nota_clinica"):
                    lines.append(f"    Nota clínica: {a['nota_clinica']}")

        if self.safe_sweeteners:
            lines.append("\n[EDULCORANTES SEGUROS]")
            for sw in self.safe_sweeteners:
                lines.append(f"  - {sw}")

        beneficial = [a for a in self.clinical_alerts if a["risco"] in ("BENEFICO", "BAIXO")
                      and a["classe"] not in ("edulcorante_artificial", "edulcorante_natural", "poliol")]
        if beneficial:
            lines.append("\n[COMPONENTES DE BAIXO RISCO]")
            for b in beneficial:
                lines.append(f"  - {b['ingrediente']} ({b['classe']})")

        if self.unmatched:
            lines.append(f"\nNao classificados ({len(self.unmatched)}): "
                         + ", ".join(self.unmatched[:10])
                         + (" ..." if len(self.unmatched) > 10 else ""))

        return "\n".join(lines)

    def __str__(self) -> str:
        return self.to_text_report()


class IngredientAnalyzer:
    """Analisa listas de ingredientes OCR e produz relatório clínico para DM."""

    def __init__(self, ontology_path: Path) -> None:
        self._matcher = OntologyMatcher(ontology_path)

    def analyze(self, ocr_text: str, image_name: str = "") -> IngredientReport:
        tokens = tokenize(ocr_text)
        results = self._matcher.match_all(tokens)

        matched: list[MatchResult] = []
        unmatched: list[str] = []
        for r in results:
            if r.matched_entry:
                matched.append(r)
            else:
                unmatched.append(r.token_original)

        # Deduplica por chave canônica
        seen: set[str] = set()
        deduped: list[MatchResult] = []
        for r in matched:
            k = r.matched_entry.key  # type: ignore[union-attr]
            if k not in seen:
                seen.add(k)
                deduped.append(r)

        risk_summary: dict[str, int] = {}
        for r in deduped:
            risco = r.matched_entry.risco  # type: ignore[union-attr]
            risk_summary[risco] = risk_summary.get(risco, 0) + 1

        clinical_alerts: list[dict] = []
        for r in deduped:
            entry = r.matched_entry  # type: ignore[union-attr]
            clinical_alerts.append({
                "ingrediente": r.token_original,
                "entrada_ontologia": entry.display,
                "classe": entry.classe,
                "risco": entry.risco,
                "alerta": entry.alerta,
                "indice_glicemico": entry.indice_glicemico,
                "fisiopatologia": entry.fisiopatologia,
                "nota_clinica": entry.nota_clinica,
                "match_type": r.match_type,
                "edit_distance": r.edit_distance,
            })
        clinical_alerts.sort(key=lambda a: _RISK_ORDER.get(a["risco"], 99))

        high_risk = [
            a["ingrediente"] for a in clinical_alerts
            if a["risco"] in ("ALTO", "MODERADO-ALTO")
        ]
        safe_sweeteners = [
            a["ingrediente"] for a in clinical_alerts
            if a["classe"] in ("edulcorante_artificial", "edulcorante_natural")
            and a["risco"] in ("BAIXO", "SEGURO")
        ]

        return IngredientReport(
            image_name=image_name,
            tokens_found=[orig for orig, _ in tokens],
            matches=deduped,
            unmatched=unmatched,
            risk_summary=risk_summary,
            risco_global=_global_risk(risk_summary),
            clinical_alerts=clinical_alerts,
            high_risk_ingredients=high_risk,
            safe_sweeteners=safe_sweeteners,
        )
