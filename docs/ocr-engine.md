# Motor OCR

## Visão Geral

O diretório `ocr_engine/` é uma cópia fiel do monolito `teste-pytesseract`,
integrada ao backend sem modificar nenhum arquivo interno. O único ponto de
integração é `ocr_engine/__init__.py`.

```python
# ocr_engine/__init__.py
_ENGINE_ROOT = Path(__file__).parent

if str(_ENGINE_ROOT) not in sys.path:
    sys.path.insert(0, str(_ENGINE_ROOT))

from ocr_engine.nutrition.reader import NutritionReader, build_default_reader

def build_reader() -> NutritionReader:
    return build_default_reader(_ENGINE_ROOT)
```

`sys.path` é manipulado para que imports absolutos do motor (`from nutrition.reader import ...`)
resolvam dentro de `ocr_engine/` sem alterar os arquivos copiados.

---

## Singleton `NutritionReader`

Carregado uma vez no `lifespan` do FastAPI:

```python
app.state.reader = build_reader()
```

**Custo de inicialização:** carrega todos os presets JSON, `FormatDetector`,
`RoiDetector` (rede SSD opcional), `IngredientAnalyzer` (ontologia + wordlist) e
`NutritionTextPostProcessor`. **Nunca reinstanciar por requisição.**

---

## Fluxo de `NutritionReader.read()`

```
image_path (pathlib.Path)
    │
    ├─ 1. Leitura da imagem (imaging.io.read_image — suporta Unicode paths)
    │
    ├─ 2. Detecção de formato
    │       category_override do endpoint → bypassa FormatDetector
    │       force_category em routing.json → bypassa heurística
    │       heurística: estimate_grid_density → "table" ou "text"
    │       category "ingredient" só por override
    │
    ├─ 3. ROI (opcional, roi_enabled)
    │       RoiDetector: SSD MobileNet v2 → fallback por contornos
    │       hint: "table" → busca grade; "text"/"ingredient" → text-blobbing
    │
    ├─ 4. Cascata de presets (para a categoria detectada)
    │       Para cada preset por ordem de priority:
    │           pipeline.execute(image) → PipelineResult
    │           QualityEvaluator → QualityScore (passed, score contínuo)
    │           NutritionTextPostProcessor → texto normalizado
    │           Se passed e stop_on_first_pass → encerra
    │       Vencedor = maior score contínuo se nenhum passou
    │
    ├─ 5. Pós-processamento final do texto vencedor
    │
    ├─ 6. Persistência em disco (AuditRecorder)
    │       extractions/<input>/{final.txt, _summary.json, ...}
    │       images/pipeline/<input>/<preset>/*.png (auditoria visual)
    │
    └─ 7. Análise de ingredientes (se category == "ingredient")
            IngredientAnalyzer.analyze(final_ocr_text)
            tokenizer → segmenter Viterbi → OntologyMatcher
            → IngredientReport
```

---

## Configuração (`ocr_engine/config/`)

| Arquivo | Conteúdo |
|---|---|
| `app.json` | Paths, ROI, defaults de opções, bloco `gcv` |
| `routing.json` | `grid_density_threshold`, `force_category` |
| `ontology_diabetes.json` | Base de conhecimento clínico DM (36+ entradas) |
| `wordlist_pt_food.txt` | Vocabulário PT-BR para Viterbi (`palavra<TAB>freq`) |
| `presets/table/*.json` | Estratégias para tabelas nutricionais |
| `presets/text/*.json` | Estratégias para texto corrido |
| `presets/ingredients/*.json` | Estratégias para listas de ingredientes |

---

## Presets OCR

Cada preset é um arquivo JSON declarativo que define:

```json
{
  "name": "gcv_doc_text",
  "description": "Google Cloud Vision DOCUMENT_TEXT_DETECTION.",
  "kind": "cloud_vision",
  "priority": 5,
  "steps": [],
  "ocr": { "lang": "por", "psm": 6, "oem": 3 },
  "gcv": {
    "feature": "DOCUMENT_TEXT_DETECTION",
    "language_hints": ["pt"],
    "model": null
  },
  "quality_thresholds": {
    "min_mean_confidence": 75,
    "min_text_length": 40,
    "min_keyword_hits": 3,
    "expected_keywords": ["valor energetico", "carboidratos", "proteinas", ...]
  }
}
```

### Tipos de pipeline (`kind`)

| `kind` | Executor | Descrição |
|---|---|---|
| `linear_table` | `LinearPipeline` | Sequência de steps PDI + OCR único |
| `linear_text` | `LinearPipeline` | Idem |
| `linear_ingredient` | `LinearPipeline` | Idem; OCR vai para `IngredientAnalyzer` |
| `cell_based` | `CellBasedPipeline` | Detecção de células + OCR por célula |
| `cloud_vision` | `CloudVisionPipeline` | Google Cloud Vision API |

### Presets disponíveis

**Tabela (`presets/table/`)**

| Arquivo | `kind` | Prioridade | Destaque |
|---|---|---|---|
| `00_gcv_doc_text.json` | `cloud_vision` | 5 | GCV — alta precisão |
| `01_otsu_basic.json` | `linear_table` | 10 | Otsu simples |
| `02_adaptive_threshold.json` | `linear_table` | 20 | Adaptativo |
| `03_adaptive_lineremoval.json` | `linear_table` | 30 | Remove linhas de grade |
| `04_clahe_otsu_lineremoval.json` | `linear_table` | 40 | CLAHE + Otsu + remoção |
| `05_aggressive_adaptive.json` | `linear_table` | 50 | Adaptativo agressivo |
| `06_deskew_adaptive.json` | `linear_table` | 60 | Correção de inclinação |
| `07_cell_based_otsu.json` | `cell_based` | 70 | OCR por célula, Otsu |
| `08_cell_based_adaptive.json` | `cell_based` | 80 | OCR por célula, adaptativo |
| `09_highres_clahe_psm4.json` | `linear_table` | 90 | Alta resolução, PSM 4 |
| `10_bilateral_sharpen_otsu.json` | `linear_table` | 100 | Bilateral + realce |

**Texto (`presets/text/`)**

| Arquivo | `kind` | Prioridade |
|---|---|---|
| `00_gcv_doc_text.json` | `cloud_vision` | 5 |
| `01_otsu_psm6.json` | `linear_text` | 10 |
| `02_adaptive_psm6.json` | `linear_text` | 20 |
| `03_adaptive_psm4.json` | `linear_text` | 30 |
| `04_deskew_psm3.json` | `linear_text` | 40 |
| `05_aggressive_enhance.json` | `linear_text` | 50 |

**Ingredientes (`presets/ingredients/`)**

| Arquivo | `kind` | Prioridade |
|---|---|---|
| `00_gcv_doc_text.json` | `cloud_vision` | 5 |
| `01_psm4_otsu.json` | `linear_ingredient` | 10 |
| `02_psm6_adaptive.json` | `linear_ingredient` | 20 |
| `03_otsu_erosion.json` | `linear_ingredient` | 30 |
| `04_bilateral_adaptive.json` | `linear_ingredient` | 40 |
| `05_sharpened_otsu.json` | `linear_ingredient` | 50 |

---

## Qualidade e Score

`QualityEvaluator` avalia cada resultado com três sinais:

| Sinal | Descrição |
|---|---|
| `mean_confidence` | Confiança média do Tesseract (0–100) |
| `text_length` | Comprimento do texto extraído |
| `keyword_hits` | Keywords esperadas encontradas (Levenshtein) |

**`passed`** (bool) = AND dos três limiares de `quality_thresholds`.

**`score`** (0–1) = combinação contínua para ranquear quando nenhum passa:

```
score = 0.50 × (mean_confidence/100)
      + 0.40 × min(keyword_hits/len(expected_keywords), 1.0)
      + 0.10 × min(text_length/500, 1.0)
```

---

## Google Cloud Vision (GCV)

Integração opcional via service account. Ativada automaticamente quando presets
`cloud_vision` estão presentes e `credentials_path` ou
`GOOGLE_APPLICATION_CREDENTIALS` está configurado.

### Configuração em `ocr_engine/config/app.json`

```json
"gcv": {
  "credentials_path": null,
  "on_failure": "skip",
  "cache_enabled": true,
  "cache_dir": "extractions/.gcv_cache",
  "max_requests_per_minute": null,
  "request_timeout_seconds": 30
}
```

| Campo | Descrição |
|---|---|
| `credentials_path` | Path do service account (relativo ou absoluto) |
| `on_failure` | `"skip"` (retorna vazio) ou `"raise"` (propaga exceção) |
| `cache_enabled` | Habilita cache em disco por SHA-256 da imagem |
| `cache_dir` | Diretório de cache (fora do caminho de limpeza do AuditRecorder) |
| `max_requests_per_minute` | Rate limit (null = sem limite) |
| `request_timeout_seconds` | Timeout por chamada à API |

### Resolução de credenciais

Ordem de prioridade:
1. `credentials_path` em `app.json`
2. Variável de ambiente `GOOGLE_APPLICATION_CREDENTIALS`
3. Erro `GcvError(error="auth_error")`

O conteúdo do service account nunca aparece em logs ou artefatos.

### Cache em disco

Layout: `extractions/.gcv_cache/<sha256>.json` + `<sha256>.meta.json`

O cache é indexado por hash SHA-256 (PNG) + `feature` + `language_hints`.
Sem TTL — entradas não expiram automaticamente. O diretório de cache fica fora
do caminho de limpeza do `AuditRecorder` por construção.

### Classificação de erros

| Exceção | `error` |
|---|---|
| `PermissionDenied`, `Unauthenticated` | `auth_error` |
| `ResourceExhausted`, HTTP 429 | `quota_exceeded` |
| `DeadlineExceeded`, timeout | `timeout` |
| `ImportError` da biblioteca GCV | `import_error` |
| Qualquer outra | `generic_error` |

Com `on_failure="skip"`, erros produzem `PipelineResult` vazio que percorre a
cascata normalmente (`passed=False`). Com `"raise"`, a exceção é propagada.

---

## Análise Clínica de Ingredientes (DM)

Ativada quando `detected_format.category == "ingredient"` e o arquivo
`config/ontology_diabetes.json` existe.

### Pipeline em três camadas

```
ocr_text
    │
    ├─ 1. tokenizer.tokenize()
    │       limpeza de ruído (E-numbers, parênteses, caracteres especiais)
    │       split por vírgula/ponto-e-vírgula/ponto
    │       segmentação Viterbi para tokens > 10 chars
    │
    ├─ 2. OntologyMatcher.match_all()
    │       exato → containment (word-boundary) → Levenshtein → word_decomposition
    │
    └─ 3. IngredientAnalyzer.analyze()
            dedup por chave canônica
            cálculo de risco_global (pior risco)
            ordenação clínica
            segregação de edulcorantes
```

### Hierarquia de risco

```
ALTO > MODERADO-ALTO > MODERADO > BAIXO > SEGURO > BENEFICO > INFORMATIVO
```

`risco_global` = pior risco entre todos os ingredientes identificados.

### Ontologia DM (`config/ontology_diabetes.json`)

Base de conhecimento com 36+ entradas. Cada entrada:

```json
"sacarose": {
  "classe": "acucar_simples",
  "risco": "ALTO",
  "alerta": "Eleva glicemia rapidamente. IG ≈ 65.",
  "sinonimos": ["açúcar", "acucar", "sugar", "sucrose"],
  "indice_glicemico": 65,
  "fisiopatologia": "...",
  "nota_clinica": null,
  "referencias": ["DOI: ..."]
}
```

Entradas especiais: `maltodextrina` (ALTO) vs `maltodextrina_resistente` (BENEFICO)
são tratadas separadamente para evitar confusão clínica.

---

## Operações PDI disponíveis

Operações registradas em `OPERATION_REGISTRY` (`imaging/operations.py`):

`grayscale`, `resize_max_height`, `clahe`, `blur`, `median_blur`,
`gaussian_blur`, `bilateral_filter`, `otsu_threshold`, `adaptive_threshold`,
`ensure_black_text_on_white`, `morph_open`, `morph_close`, `morph_dilate`,
`morph_erode`, `thin_text_if_thick`, `deskew`, `unsharp_mask`, `sharpen`

Operações morfológicas complexas despachadas manualmente em `apply_operation`:
`remove_grid_lines` (via `imaging.morphology`).

---

## Auditoria

O `AuditRecorder` persiste cada etapa de cada preset como PNG:

```
images/pipeline/<input>/<NN>_<preset>/
    <input>__<preset>__01_input.<ext>
    <input>__<preset>__02_<stage>.<ext>
    ...
    <input>__<preset>__NN_output.<ext>
```

E textos/scores em:

```
extractions/<input>/
    final.txt
    final_postprocessed.txt
    _summary.json
    <NN>_<preset>/ocr.txt
    <NN>_<preset>/score.json
    [analysis.json]          ← apenas para ingredientes
    [feedback_clinico.txt]   ← apenas para ingredientes
    [ocr_tokens.txt]         ← apenas para ingredientes
    [metrics_wer_cer.json]   ← apenas se ground-truth existir
```

No contexto HTTP, os artefatos em disco são gerados dentro do container Docker
(não acessíveis diretamente pelo cliente). O resultado estruturado é retornado
via JSON da API.
