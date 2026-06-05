"""Executor que delega o OCR à Google Cloud Vision API.

``CloudVisionPipeline`` é o ``kind`` de pipeline ``cloud_vision``: recebe a
imagem já preparada pelo ``NutritionReader`` (ROI aplicado quando habilitado)
e devolve um ``PipelineResult`` no mesmo contrato dos demais executores
(``LinearPipeline``, ``CellBasedPipeline``), permitindo que GCV concorra com
presets Tesseract dentro da mesma cascata controlada pelo
``QualityEvaluator``.

Características relevantes para o orquestrador:

- O pipeline NÃO aplica nenhuma operação do ``OPERATION_REGISTRY`` à imagem
  antes do envio (Requirements 1.6 e 1.7). Os ``steps`` declarados no JSON
  são intencionalmente ignorados — o ``ignored_steps_count`` é registrado
  em ``metadata`` para auditoria.
- O fluxo segue o flowchart do design: persiste ``input`` (stage 01),
  curto-circuita imediatamente quando ``gcv.feature`` é inválido
  (Requirement 3.5), codifica PNG, calcula SHA-256, delega a chamada ao
  ``GcvClient.fetch`` (que trata cache, rate-limit e classificação de
  erros internamente), parseia a resposta, gera overlay de bounding boxes
  e grava resposta crua local em conformidade com Requirement 10.4
  (auditoria simétrica entre cache hit e miss).
- Em falha (``GcvError``) o tratamento é controlado pela política
  ``on_failure`` injetada pelo reader: ``skip`` produz ``PipelineResult``
  vazio mas válido (Requirement 6.2) preservando ``stage final = "output"``;
  ``raise`` propaga a exceção para abortar APENAS a leitura da imagem
  corrente (Requirement 6.4).
- ``gcv_config_warnings`` é injetado pelo reader e retransmitido em
  ``metadata`` SEM modificação. O reader é responsável por zerar a tupla
  após a primeira tentativa GCV de cada execução (Requirement 8.5).
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

import cv2
import numpy as np

from ocr.cloud_vision.options import GcvPresetOptions
from ocr.cloud_vision.parser import WordBox, WordToken, encode_png, parse_response
from ocr.cloud_vision.types import GcvError
from ocr.service import OcrConfig

from .base import Pipeline, PipelineContext, PipelineResult, StageRecord

if TYPE_CHECKING:  # pragma: no cover - import só para anotação de tipo.
    # ``GcvClient`` é injetado pelo ``NutritionReader``; o pipeline depende
    # apenas do método ``fetch`` (duck typing). O import sob ``TYPE_CHECKING``
    # evita acoplar este módulo ao SDK ``google-cloud-vision`` em tempo de
    # carga (Requirement 14.3 — feature opcional).
    from ocr.cloud_vision.client import GcvClient


# ---------------------------------------------------------------------------
# Constantes de naming canônico das stages
# ---------------------------------------------------------------------------

# Rótulos das stages persistidas via ``AuditRecorder``. ``input`` e ``output``
# preservam a convenção comum a todos os pipelines (Requirement 1.5).
# ``gcv_boxes_overlay`` é exigido literalmente pelo Requirement 10.2.
# ``gcv_response`` é o nome do artefato JSON gravado por
# ``AuditRecorder.save_stage_json`` (Requirement 10.1) e segue o naming
# canônico ``<input>__<preset>__NN_gcv_response.json``.
_STAGE_INPUT = "input"
_STAGE_OVERLAY = "gcv_boxes_overlay"
_STAGE_RESPONSE = "gcv_response"
_STAGE_OUTPUT = "output"

# Op canônico exposto em ``StageRecord.op``. Espelha o estilo de
# ``cell_based.py`` (op descreve a operação que produziu o artefato, não o
# rótulo da stage).
_OP_INPUT = "input"
_OP_OUTPUT = "output"
_OP_OVERLAY = "gcv_word_boxes"
_OP_RESPONSE = "gcv_response_dump"

# Tamanho máximo da mensagem de erro em ``metadata.error_message``
# (Requirement 6.2). Truncamento defensivo evita inflar ``_summary.json``
# com tracebacks longos do SDK.
_ERROR_MESSAGE_MAX_LEN = 500


class CloudVisionPipeline(Pipeline):
    """Adapter entre ``Pipeline`` e ``GcvClient``.

    Não faz I/O de rede diretamente — delega ao ``client.fetch`` e consome
    apenas a interface canônica ``GcvFetchResult`` / ``GcvError``. Toda
    persistência de artefatos (PNG e JSON) passa exclusivamente pelo
    ``AuditRecorder`` injetado via ``PipelineContext``.

    Attributes:
        gcv_options: Opções derivadas do bloco ``gcv`` do preset (feature,
            language_hints, model, sinalização ``invalid_feature``).
        ocr_config: ``OcrConfig`` recebido para uniformidade com os demais
            pipelines (Tesseract). A GCV não consome ``OcrConfig``; o campo
            é mantido apenas para que o reader use a mesma assinatura de
            construção em todos os ``kind``.
        client: Instância de ``GcvClient`` construída no boot do reader.
            O pipeline não faz preflight nem inicialização lazy — qualquer
            falha de import/auth é traduzida em ``GcvError`` por
            ``client.fetch`` na primeira chamada.
        on_failure: Política de tratamento de falha (``"skip"`` | ``"raise"``)
            replicada de ``GcvAppConfig.on_failure``.
        ignored_steps_count: Quantidade de ``steps`` declarados no JSON do
            preset, exposta em ``metadata.ignored_steps_count`` para
            auditoria (Requirement 1.7).
        gcv_config_warnings: Tupla de warnings de coerção produzidos pelo
            ``GcvAppConfig.from_dict``. Repassada como lista em
            ``metadata.gcv_config_warnings`` (Requirement 8.5). O reader
            é responsável por entregar tupla vazia em tentativas
            posteriores à primeira de cada execução.
    """

    def __init__(
        self,
        gcv_options: GcvPresetOptions,
        ocr_config: OcrConfig,
        client: "GcvClient",
        on_failure: str,
        ignored_steps_count: int,
        gcv_config_warnings: tuple[str, ...] = (),
    ) -> None:
        self.gcv_options = gcv_options
        # ``ocr_config`` é guardado por uniformidade com outros pipelines,
        # mas a GCV não consome o objeto — o reader passa a mesma instância
        # para todos os ``kind`` na construção.
        self.ocr_config = ocr_config
        self.client = client
        self.on_failure = on_failure
        self.ignored_steps_count = ignored_steps_count
        # Coerção defensiva para tupla imutável: o reader pode passar lista
        # ou tupla; preservamos imutabilidade dentro do pipeline.
        self.gcv_config_warnings = tuple(gcv_config_warnings)

    # ------------------------------------------------------------------
    # API pública (contrato ``Pipeline``)
    # ------------------------------------------------------------------

    def execute(
        self,
        image: np.ndarray,
        context: PipelineContext,
    ) -> PipelineResult:
        """Executa o pipeline GCV seguindo o flowchart do design.

        A imagem recebida é a entregue pelo reader (já com ROI aplicado se
        ``options.roi_enabled``). Ela é persistida como ``stage 01 = "input"``
        e usada como entrada para o cálculo do PNG/SHA-256 e para o overlay
        final — garantindo que ``cache_hit`` seja reproduzível: bytes
        idênticos ⇒ hash idêntico ⇒ resposta servida do cache.
        """

        stages: list[StageRecord] = []
        recorder = context.recorder
        artifacts = context.artifacts

        # ---------------------------------------------------------------
        # 1. Stage 01 = "input" — referência visual obrigatória do contrato
        # de pipelines (Requirement 1.5; ``AGENTS.md`` reforça que a etapa
        # ``input`` nunca pode ser removida).
        # ---------------------------------------------------------------
        stage_index = 1
        path_input = recorder.save_stage(artifacts, stage_index, _STAGE_INPUT, image)
        stages.append(
            StageRecord(stage_index, _STAGE_INPUT, _OP_INPUT, str(path_input), {})
        )

        # ---------------------------------------------------------------
        # 2. invalid_feature: curto-circuito antes de qualquer I/O. O
        # ``CloudVisionPipeline`` NÃO consulta cache nem chama a API
        # quando o preset declarou ``gcv.feature`` fora de
        # ``ALLOWED_FEATURES`` (Requirement 3.5). O caminho ``raise``
        # também NÃO se aplica aqui: ``invalid_feature`` é erro de
        # configuração, não de chamada à API; o pipeline produz
        # ``PipelineResult`` vazio para que a cascata Tesseract tome
        # frente.
        # ---------------------------------------------------------------
        if self.gcv_options.invalid_feature:
            err = GcvError(
                error="invalid_feature",
                message=(
                    f"feature inválida no preset: {self.gcv_options.raw_feature!r}"
                ),
            )
            return self._build_failure_result(image, stages, stage_index, context, err)

        # ---------------------------------------------------------------
        # 3. Redimensionamento defensivo + codificação PNG + SHA-256.
        #
        # Imagens acima de ``gcv_options.max_image_dimension`` pixels no
        # lado maior são redimensionadas proporcionalmente antes da
        # codificação. Isso evita:
        #   a) Timeouts de rede ao enviar imagens de vários MB inline.
        #   b) O limite de 20 MB da GCV API para conteúdo inline.
        # ``max_image_dimension == 0`` desabilita o redimensionamento.
        #
        # ``image_for_gcv`` é o array que vai para a API; ``image`` (original)
        # é preservado para a stage ``output`` e para o contrato visual da
        # auditoria.
        # ---------------------------------------------------------------
        max_dim = self.gcv_options.max_image_dimension
        if max_dim > 0:
            h, w = image.shape[:2]
            if max(h, w) > max_dim:
                scale = max_dim / max(h, w)
                image_for_gcv = cv2.resize(
                    image,
                    (int(w * scale), int(h * scale)),
                    interpolation=cv2.INTER_AREA,
                )
            else:
                image_for_gcv = image
        else:
            image_for_gcv = image

        png_bytes = encode_png(image_for_gcv)
        # ``hashlib.sha256`` é determinística e independente de plataforma;
        # o ``hexdigest`` é a chave canônica do ``GcvCache`` e é também
        # passado adiante caso o cliente queira loggar.
        _ = hashlib.sha256(png_bytes).hexdigest()

        # ---------------------------------------------------------------
        # 4. Delegação ao ``GcvClient.fetch``. Toda a complexidade de
        # cache/rate-limit/classify vive lá; o pipeline só vê dois
        # desfechos canônicos: ``GcvFetchResult`` em sucesso (com
        # ``cache_hit`` indicando se a resposta veio do disco) ou
        # ``GcvError`` em falha (Requirements 6.2 e 6.5–6.8).
        # ---------------------------------------------------------------
        try:
            fetch_result = self.client.fetch(
                png_bytes,
                self.gcv_options.feature,
                # ``list`` para alinhar com a API do cliente, que aceita
                # iteráveis comuns; preservamos a ordem do preset (a GCV
                # interpreta hints como lista de prioridade).
                list(self.gcv_options.language_hints),
            )
        except GcvError as err:
            return self._handle_failure(image, stages, stage_index, context, err)

        # ---------------------------------------------------------------
        # 5. Sucesso (chamada real OU cache hit). Auditoria simétrica
        # (Requirement 10.4): geramos o overlay e gravamos a cópia local
        # da resposta JSON SEMPRE, independentemente de ``cache_hit``.
        # ---------------------------------------------------------------
        parsed = parse_response(fetch_result.response_json, self.gcv_options.feature)

        # Stage 02 — overlay de bounding boxes em verde sobre a imagem
        # ENVIADA à API (Requirement 10.2). Usa ``image_for_gcv`` (não
        # ``image``) para que as coordenadas dos boxes (que vêm da imagem
        # redimensionada) estejam alinhadas com o canvas do overlay.
        stage_index += 1
        overlay = self._draw_word_boxes(image_for_gcv, parsed.word_boxes)
        overlay_path = recorder.save_stage(
            artifacts, stage_index, _STAGE_OVERLAY, overlay
        )
        stages.append(
            StageRecord(
                stage_index,
                _STAGE_OVERLAY,
                _OP_OVERLAY,
                str(overlay_path),
                # ``box_count`` é informativo para inspeção do
                # ``_summary.json``; não afeta nenhum consumidor.
                {"box_count": len(parsed.word_boxes)},
            )
        )

        # Stage 03 — resposta crua local (Requirements 10.1 e 10.4). O
        # naming canônico é tratado por ``AuditRecorder.save_stage_json``;
        # o ``stage_index`` zero-padded em duas casas garante NN consistente
        # com o restante da auditoria.
        stage_index += 1
        response_path = recorder.save_stage_json(
            artifacts,
            stage_index,
            _STAGE_RESPONSE,
            fetch_result.response_json,
        )
        stages.append(
            StageRecord(
                stage_index,
                _STAGE_RESPONSE,
                _OP_RESPONSE,
                str(response_path),
                # Registrar ``cache_hit`` aqui facilita o debug visual no
                # ``_summary.json`` mesmo antes de inspecionar
                # ``metadata.cache_hit``.
                {"cache_hit": fetch_result.cache_hit},
            )
        )

        # Stage final — ``output`` (Requirement 1.5). Para o GCV, a
        # imagem ``output`` é a MESMA do ``input`` porque o pipeline não
        # transforma a entrada — preserva o contrato visual e o caller
        # consegue cruzar input/output esperando equivalência.
        stage_index += 1
        output_path = recorder.save_stage(
            artifacts, stage_index, _STAGE_OUTPUT, image
        )
        stages.append(
            StageRecord(stage_index, _STAGE_OUTPUT, _OP_OUTPUT, str(output_path), {})
        )

        # ---------------------------------------------------------------
        # 6. Construção do ``PipelineResult`` final. A estrutura de
        # ``metadata`` segue o schema definido no design (seção "Campos
        # novos em ``PipelineResult.metadata`` quando o vencedor é GCV").
        # As listas (``language_hints``, ``error_secondary``,
        # ``gcv_config_warnings``) são gravadas como ``list`` para que o
        # ``_summary.json`` seja JSON-serializável sem custom encoder.
        # ---------------------------------------------------------------
        metadata: dict = {
            "feature": self.gcv_options.feature,
            "language_hints": list(self.gcv_options.language_hints),
            "block_count": parsed.block_count,
            "paragraph_count": parsed.paragraph_count,
            "word_count": parsed.word_count,
            "cache_hit": fetch_result.cache_hit,
            "gcv_response_path": str(response_path),
            "ignored_steps_count": self.ignored_steps_count,
            "confidence_warning": parsed.confidence_warning,
            "error": None,
            "error_message": None,
            "error_secondary": [],
            "gcv_config_warnings": list(self.gcv_config_warnings),
        }

        # Reconstrução espacial de tabela (opt-in por preset).
        # Quando ``table_reconstruction=True``, usa as posições dos
        # word tokens para agrupar palavras em linhas e detectar colunas
        # por gap horizontal, produzindo texto com ``\t`` que o
        # ``NutritionTextPostProcessor`` interpreta como tabular.
        # Fallback automático para ``parsed.text`` quando não há tokens.
        if self.gcv_options.table_reconstruction and parsed.word_tokens:
            ocr_text = self._reconstruct_table_text(
                parsed.word_tokens,
                image_for_gcv.shape[1],
            )
        else:
            ocr_text = parsed.text

        return PipelineResult(
            ocr_text=ocr_text,
            mean_confidence=parsed.mean_confidence,
            stages=stages,
            final_image=image,
            metadata=metadata,
        )

    # ------------------------------------------------------------------
    # Helpers privados
    # ------------------------------------------------------------------

    def _handle_failure(
        self,
        image: np.ndarray,
        stages: list[StageRecord],
        stage_index: int,
        context: PipelineContext,
        err: GcvError,
    ) -> PipelineResult:
        """Aplica a política ``on_failure`` para um ``GcvError``.

        - ``"raise"`` propaga a exceção tal como veio do ``client.fetch``;
          o reader/main aborta APENAS a leitura da imagem corrente
          (Requirements 6.4 e 6.10).
        - ``"skip"`` (default) produz um ``PipelineResult`` vazio mas
          válido, registrando ``stage final = "output"`` para preservar o
          contrato visual da auditoria (Requirement 6.2). O
          ``QualityEvaluator`` gera ``passed=False`` automaticamente, e a
          cascata avança para o próximo preset (Requirement 6.3).

        Para qualquer valor de ``on_failure`` diferente de ``"raise"``,
        adotamos o caminho ``skip`` por defesa — ``GcvAppConfig.from_dict``
        já coage valores inválidos para ``"skip"`` antes de chegar aqui,
        portanto este ramo é apenas uma rede de segurança defensiva.
        """

        if self.on_failure == "raise":
            raise err
        return self._build_failure_result(image, stages, stage_index, context, err)

    def _build_failure_result(
        self,
        image: np.ndarray,
        stages: list[StageRecord],
        stage_index: int,
        context: PipelineContext,
        err: GcvError,
    ) -> PipelineResult:
        """Constrói ``PipelineResult`` vazio do caminho ``skip``/``invalid_feature``.

        Mantém o contrato canônico:

        1. Persiste ``stage final = "output"`` com a imagem do input para
           que ``stages[-1].name == "output"`` (Property 1) e a inspeção
           visual continue funcional mesmo sem texto produzido.
        2. ``ocr_text=""`` e ``mean_confidence=0.0`` — o
           ``QualityEvaluator`` enxerga texto vazio + confiança zero e
           gera ``passed=False`` com ``score`` próximo de zero, deixando
           a cascata progredir naturalmente para o próximo preset
           (Requirement 6.3).
        3. ``metadata`` segue a mesma forma do caminho de sucesso, com os
           campos zero-erros substituídos pela classificação canônica via
           ``_populate_error_metadata`` (centraliza truncamento +
           classificação para evitar duplicação entre os dois callers
           desta helper).
        """

        recorder = context.recorder
        artifacts = context.artifacts

        # Stage final — ``output`` com a imagem do input (Requirement 1.5
        # + Property 1). Sem isso, ``stages[-1].name == "output"`` falha e
        # consumidores que iteram pela última etapa para rastrear a saída
        # final (``_summary.json``) ficam desalinhados.
        stage_index += 1
        output_path = recorder.save_stage(
            artifacts, stage_index, _STAGE_OUTPUT, image
        )
        stages.append(
            StageRecord(stage_index, _STAGE_OUTPUT, _OP_OUTPUT, str(output_path), {})
        )

        # ``metadata`` no caminho de falha replica a forma do sucesso para
        # que consumidores externos (``_summary.json``, UI) não precisem
        # ramificar por presença/ausência de chave. Os campos derivados da
        # parse (block/paragraph/word counts, gcv_response_path,
        # confidence_warning) são neutralizados.
        metadata: dict = {
            "feature": self.gcv_options.feature,
            "language_hints": list(self.gcv_options.language_hints),
            "block_count": 0,
            "paragraph_count": 0,
            "word_count": 0,
            "cache_hit": False,
            "gcv_response_path": None,
            "ignored_steps_count": self.ignored_steps_count,
            "confidence_warning": None,
            "gcv_config_warnings": list(self.gcv_config_warnings),
        }
        self._populate_error_metadata(err, metadata)

        return PipelineResult(
            ocr_text="",
            mean_confidence=0.0,
            stages=stages,
            final_image=image,
            metadata=metadata,
        )

    @staticmethod
    def _populate_error_metadata(err: GcvError, metadata: dict) -> None:
        """Centraliza truncamento + classificação de ``GcvError`` em ``metadata``.

        - ``metadata["error"]`` recebe o código canônico (Requirements 6.5–6.8
          + ``invalid_feature`` + ``import_error``).
        - ``metadata["error_message"]`` é truncada a 500 caracteres
          (Requirement 6.2). Mensagens ``None`` são mapeadas para string
          vazia para que consumidores não precisem checar tipo.
        - ``metadata["error_secondary"]`` recebe ``list(err.secondary)``
          preservando a precedência aplicada em ``GcvClient._classify``
          (Requirement 6.8).
        """

        metadata["error"] = err.error
        # ``err.message`` é tipado como ``str``, mas defendemos contra
        # ``None`` defensivamente — evita ``TypeError`` em fatiamento se
        # algum caller construir ``GcvError`` com mensagem nula.
        message = err.message if err.message is not None else ""
        if len(message) > _ERROR_MESSAGE_MAX_LEN:
            message = message[:_ERROR_MESSAGE_MAX_LEN]
        metadata["error_message"] = message
        metadata["error_secondary"] = list(err.secondary)

    @staticmethod
    def _reconstruct_table_text(
        word_tokens: tuple[WordToken, ...],
        image_w: int,
    ) -> str:
        """Reconstrói texto de tabela usando posições pixel dos word tokens.

        Algoritmo:

        1. **Agrupamento em linhas**: ordena tokens por Y-centro e agrupa os
           que têm Y-centros próximos (tolerância = 0,55 × altura mediana).

        2. **Threshold de gap por quebra natural (Jenks simplificado)**:
           coleta TODOS os gaps inter-token de TODAS as linhas, ordena e
           localiza o maior salto. Esse salto separa "espaço dentro de frase"
           (0–20 px) de "separação entre colunas" (40–200 px). O threshold
           é o ponto médio do maior salto, garantindo robustez mesmo para
           imagens com fontes e layouts variados.

        3. **Montagem**: para cada linha, insere ``\\t`` onde o gap supera o
           threshold; caso contrário insere espaço. Produz texto com tabs que
           o ``NutritionTextPostProcessor`` processa no branch tabular.

        Fallback seguro: se não houver salto significativo nos gaps (imagem sem
        estrutura tabular clara), devolve ``parsed.text`` original via
        ausência de tabs — a cascata avança para presets Tesseract.
        """

        if not word_tokens:
            return ""

        # ------------------------------------------------------------------ #
        # 1. Agrupamento em linhas por Y-centro                               #
        # ------------------------------------------------------------------ #

        heights = [t.box.y2 - t.box.y1 for t in word_tokens if t.box.y2 > t.box.y1]
        if not heights:
            return " ".join(t.text for t in word_tokens)
        heights_sorted = sorted(heights)
        median_h = heights_sorted[len(heights_sorted) // 2]
        y_tol = max(6, int(median_h * 0.55))

        def _y_center(t: WordToken) -> int:
            return (t.box.y1 + t.box.y2) // 2

        sorted_tokens = sorted(word_tokens, key=lambda t: (_y_center(t), t.box.x1))

        rows: list[list[WordToken]] = []
        row_y_sums: list[float] = []
        row_counts: list[int] = []

        for token in sorted_tokens:
            tc_y = _y_center(token)
            placed = False
            for i, (ys, cnt) in enumerate(zip(row_y_sums, row_counts)):
                if abs(tc_y - ys / cnt) <= y_tol:
                    rows[i].append(token)
                    row_y_sums[i] += tc_y
                    row_counts[i] += 1
                    placed = True
                    break
            if not placed:
                rows.append([token])
                row_y_sums.append(float(tc_y))
                row_counts.append(1)

        rows = [sorted(row, key=lambda t: t.box.x1) for row in rows]

        # ------------------------------------------------------------------ #
        # 2. Detecção de separadores de coluna por consistência entre linhas  #
        # ------------------------------------------------------------------ #

        # Para cada linha, registra quais intervalos X estão "vazios"
        # (gap entre dois tokens consecutivos). Um intervalo X vazio que
        # aparece em ≥ _MIN_ROWS_WITH_GAP linhas distintas é classificado
        # como separador de coluna real.
        #
        # Esse critério de consistência filtra:
        #   - Espaços entre palavras dentro de uma mesma frase (aparecem
        #     em posições X variadas por linha → baixa contagem).
        #   - Separações de coluna de tabela (mesma faixa X em todas as
        #     linhas da tabela → alta contagem).
        _MIN_ROWS_WITH_GAP = 2
        _MIN_GAP_PX = 10  # gap abaixo disso é descartado (sobreposição/kerning)

        max_x = max(t.box.x2 for t in word_tokens)
        # coverage_count[x] = número de linhas que têm um gap passando por x
        coverage_count: list[int] = [0] * (max_x + 2)

        for row in rows:
            if len(row) < 2:
                continue
            row_gap_mask = bytearray(max_x + 2)
            for i in range(1, len(row)):
                g_start = row[i - 1].box.x2
                g_end = row[i].box.x1
                if g_end - g_start >= _MIN_GAP_PX:
                    for x in range(g_start, min(g_end, max_x + 2)):
                        row_gap_mask[x] = 1
            for x in range(max_x + 2):
                if row_gap_mask[x]:
                    coverage_count[x] += 1

        col_sep_pixels: set[int] = {
            x for x in range(max_x + 2)
            if coverage_count[x] >= _MIN_ROWS_WITH_GAP
        }

        def _is_col_sep(x_start: int, x_end: int) -> bool:
            """True se o intervalo [x_start, x_end) cruza algum separador consistente."""
            if x_end <= x_start + _MIN_GAP_PX:
                return False
            return any(x in col_sep_pixels for x in range(x_start, x_end))

        # ------------------------------------------------------------------ #
        # 3. Montagem das linhas com ``\t`` nos separadores de coluna         #
        # ------------------------------------------------------------------ #

        result_lines: list[str] = []
        for row in rows:
            if not row:
                continue
            parts: list[str] = [row[0].text]
            for i in range(1, len(row)):
                sep = "\t" if _is_col_sep(row[i - 1].box.x2, row[i].box.x1) else " "
                parts.append(sep)
                parts.append(row[i].text)
            result_lines.append("".join(parts))

        return "\n".join(result_lines)

    @staticmethod
    def _draw_word_boxes(
        image: np.ndarray,
        boxes: tuple[WordBox, ...],
    ) -> np.ndarray:
        """Desenha retângulos verdes axis-aligned sobre uma cópia da imagem.

        Espelha o estilo de ``_draw_cells_overlay`` em ``cell_based.py``:
        canvas BGR (converte grayscale via ``cv2.cvtColor`` quando
        necessário), retângulos ``(0, 255, 0)`` com espessura 2. As
        coordenadas vêm de ``GcvParsedResponse.word_boxes`` já
        normalizadas para axis-aligned (``min``/``max`` dos vértices em
        ``parser.py``); ``cv2.rectangle`` clipa silenciosamente quando
        algum vértice extrapola o canvas.
        """

        if image.ndim == 2:
            # Imagem grayscale: convertemos para BGR para que o desenho
            # em verde seja visível. ``cvtColor`` retorna nova matriz —
            # não há risco de mutação da imagem de input.
            canvas = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            # 3 canais (BGR) — ``copy()`` garante que o overlay não
            # mute a imagem original que ainda será usada na stage
            # ``output``.
            canvas = image.copy()
        for box in boxes:
            cv2.rectangle(
                canvas,
                (box.x1, box.y1),
                (box.x2, box.y2),
                (0, 255, 0),
                2,
            )
        return canvas
