from __future__ import annotations

import hashlib
import re
import uuid
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.analysis.models import Scan
from app.analysis.schemas import IngredientAnalysisSchema, IngredientItemSchema
from app.products.models import IngredientList, NutritionalTable, Product
from app.products.schemas import (
    IngredientsData,
    NutritionalRowData,
    NutritionalTableData,
    ProductCreateRequest,
    ProductResponse,
    ProductUpdateRequest,
)


def split_ingredient_text(raw: str) -> list[str]:
    """Quebra texto cru de OCR em itens de ingredientes.

    Usa vírgula/ponto-e-vírgula como separador; na ausência deles, cai para
    quebras de linha (listas em coluna estreita não usam vírgula). Descarta
    fragmentos curtos ou puramente numéricos (ruído de OCR).
    """
    raw = (raw or "").strip()
    if not raw:
        return []

    delimiter = r"[,;]" if re.search(r"[,;]", raw) else r"[\r\n]+"
    parts = [t.strip(" \t-•·.") for t in re.split(delimiter, raw)]
    return [p for p in parts if len(p) >= 2 and not p.isdigit()]


def parse_postprocessed_to_nutritional_table(text: str) -> NutritionalTableData | None:
    """Converte o texto pós-processado do OCR em NutritionalTableData estruturada.

    Suporta saída tab-separada (CellBasedPipeline) e saída linear (LinearPipeline).
    """
    if not text.strip():
        return None

    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    if not lines:
        return None

    has_tabs = any("\t" in ln for ln in lines)
    rows: list[NutritionalRowData] = []

    for line in lines:
        if has_tabs:
            parts = [p.strip() for p in line.split("\t")]
        else:
            # Tenta split por 2+ espaços primeiro; senão, separa no primeiro número
            parts = re.split(r"\s{2,}", line)
            if len(parts) == 1:
                m = re.match(r"^([A-Za-zÀ-ÿ\s\-]+?)\s+([\d].*)", line)
                parts = [m.group(1), m.group(2)] if m else parts

        parts = [p.strip() for p in parts if p.strip()]
        if not parts:
            continue

        nutrient = parts[0]
        values = parts[1:]
        # Descarta linhas-cabeçalho/ruído (ex.: "INFORMAÇÃO NUTRICIONAL",
        # "Porções por embalagem"): uma linha de nutriente precisa de ao menos
        # um valor numérico.
        if not any(any(ch.isdigit() for ch in v) for v in values):
            continue
        if nutrient:
            rows.append(NutritionalRowData(nutrient=nutrient, values=values))

    if not rows:
        return None

    max_vals = max((len(r.values) for r in rows), default=0)
    columns: list[str] = ["Quantidade por porção"]
    if max_vals >= 2:
        columns.append("% VD")

    return NutritionalTableData(portion_description=None, columns=columns, rows=rows)


def _compute_analysis(
    analyzer, items: list[str], barcode: str
) -> IngredientAnalysisSchema | None:
    """Computa análise DM on-the-fly a partir da lista de ingredientes."""
    if analyzer is None or not items:
        return None
    ocr_text = ", ".join(items)
    report = analyzer.analyze(ocr_text, image_name=barcode)
    ing_dict = report.to_dict()
    return IngredientAnalysisSchema(
        risco_global=ing_dict["risco_global"],
        ingredientes_identificados=[
            IngredientItemSchema(**item)
            for item in ing_dict.get("ingredientes_identificados", [])
        ],
        nao_identificados=ing_dict.get("nao_identificados", []),
        high_risk_ingredients=list(report.high_risk_ingredients),
        safe_sweeteners=list(report.safe_sweeteners),
    )


def _build_nutritional_table_data(nt: NutritionalTable) -> NutritionalTableData:
    rows = [
        NutritionalRowData(nutrient=r["nutrient"], values=list(r.get("values", [])))
        for r in (nt.rows or [])
    ]
    return NutritionalTableData(
        portion_description=nt.portion_description,
        columns=list(nt.columns or []),
        rows=rows,
    )


def build_product_response(product: Product, analyzer=None) -> ProductResponse:
    """Constrói ProductResponse incluindo análise DM on-the-fly se houver ingredientes."""
    nt_data = (
        _build_nutritional_table_data(product.nutritional_table)
        if product.nutritional_table is not None
        else None
    )
    ing_data = (
        IngredientsData(items=list(product.ingredient_list.items or []))
        if product.ingredient_list is not None
        else None
    )
    analysis = (
        _compute_analysis(analyzer, ing_data.items, product.barcode)
        if ing_data is not None
        else None
    )
    return ProductResponse(
        barcode=product.barcode,
        name=product.name,
        brand=product.brand,
        nutritional_table=nt_data,
        ingredients=ing_data,
        analysis=analysis,
        created_at=product.created_at,
        updated_at=product.updated_at,
    )


async def record_product_scan(
    db: AsyncSession,
    user_id: str,
    barcode: str,
    analysis: IngredientAnalysisSchema | None,
    name: str | None = None,
    brand: str | None = None,
) -> None:
    """Persiste um Scan no histórico do usuário ao criar um produto.

    O fluxo de produtos não passa por `POST /analyze`, então sem isto o
    histórico (`GET /users/me/scans`) ficaria vazio. `result_json` segue o
    shape de `AnalyzeResponse` para a tela de detalhe do app consumir sem ajuste.
    Não há imagem neste fluxo (corpo JSON): `image_hash` usa o SHA-256 do barcode.
    """
    result_json = {
        "barcode": barcode,
        "name": name,
        "brand": brand,
        "ingredient_analysis": analysis.model_dump() if analysis is not None else None,
        "llm_summary": analysis.natural_language_summary if analysis is not None else None,
        "final_postprocessed_text": "",
    }
    scan = Scan(
        id=str(uuid.uuid4()),
        user_id=user_id,
        image_hash=hashlib.sha256(barcode.encode("utf-8")).hexdigest(),
        detected_format="ingredient",
        winning_preset=None,
        passed=True,
        risco_global=analysis.risco_global if analysis is not None else None,
        result_json=result_json,
        created_at=datetime.now(timezone.utc),
    )
    db.add(scan)
    await db.commit()


async def get_by_barcode(db: AsyncSession, barcode: str) -> Product | None:
    result = await db.execute(select(Product).where(Product.barcode == barcode))
    return result.scalar_one_or_none()


async def create_product(
    db: AsyncSession,
    barcode: str,
    user_id: str,
    body: ProductCreateRequest,
) -> Product:
    now = datetime.now(timezone.utc)
    product = Product(
        barcode=barcode,
        name=body.name,
        brand=body.brand,
        created_at=now,
        updated_at=now,
        created_by_user_id=user_id,
    )
    db.add(product)

    if body.nutritional_table is not None:
        nt = NutritionalTable(
            id=str(uuid.uuid4()),
            product_barcode=barcode,
            portion_description=body.nutritional_table.portion_description,
            columns=[c for c in body.nutritional_table.columns],
            rows=[r.model_dump() for r in body.nutritional_table.rows],
            updated_at=now,
            updated_by_user_id=user_id,
        )
        db.add(nt)

    if body.ingredients is not None:
        il = IngredientList(
            id=str(uuid.uuid4()),
            product_barcode=barcode,
            items=list(body.ingredients.items),
            updated_at=now,
            updated_by_user_id=user_id,
        )
        db.add(il)

    await db.commit()
    await db.refresh(product)
    return product


async def update_product(
    db: AsyncSession,
    product: Product,
    user_id: str,
    body: ProductUpdateRequest,
) -> Product:
    now = datetime.now(timezone.utc)
    changed = False
    fields_set = body.model_fields_set

    if "name" in fields_set:
        product.name = body.name
        changed = True
    if "brand" in fields_set:
        product.brand = body.brand
        changed = True

    if "nutritional_table" in fields_set:
        changed = True
        if body.nutritional_table is None:
            # Apaga a tabela nutricional existente
            if product.nutritional_table is not None:
                await db.delete(product.nutritional_table)
        else:
            nt_data = body.nutritional_table
            if product.nutritional_table is not None:
                nt = product.nutritional_table
                nt.portion_description = nt_data.portion_description
                nt.columns = [c for c in nt_data.columns]
                nt.rows = [r.model_dump() for r in nt_data.rows]
                nt.updated_at = now
                nt.updated_by_user_id = user_id
            else:
                nt = NutritionalTable(
                    id=str(uuid.uuid4()),
                    product_barcode=product.barcode,
                    portion_description=nt_data.portion_description,
                    columns=[c for c in nt_data.columns],
                    rows=[r.model_dump() for r in nt_data.rows],
                    updated_at=now,
                    updated_by_user_id=user_id,
                )
                db.add(nt)

    if "ingredients" in fields_set:
        changed = True
        if body.ingredients is None:
            if product.ingredient_list is not None:
                await db.delete(product.ingredient_list)
        else:
            if product.ingredient_list is not None:
                product.ingredient_list.items = list(body.ingredients.items)
                product.ingredient_list.updated_at = now
                product.ingredient_list.updated_by_user_id = user_id
            else:
                il = IngredientList(
                    id=str(uuid.uuid4()),
                    product_barcode=product.barcode,
                    items=list(body.ingredients.items),
                    updated_at=now,
                    updated_by_user_id=user_id,
                )
                db.add(il)

    if changed:
        product.updated_at = now

    await db.commit()
    await db.refresh(product)
    return product
