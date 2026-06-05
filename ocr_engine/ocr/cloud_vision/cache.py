"""Cache em disco para respostas da Google Cloud Vision API.

Cada resposta é persistida como dois arquivos no diretório ``cache_dir``:

* ``<sha256>.json``      — resposta crua serializada (saída de
  ``MessageToDict(AnnotateImageResponse)``).
* ``<sha256>.meta.json`` — metadados de filtragem
  (``created_at``, ``feature``, ``language_hints``, ``image_size_bytes``).

A chave do cache é o SHA-256 dos bytes PNG efetivamente enviados à API
(Requirement 7.1). O filtro de compatibilidade em ``get`` exige igualdade
exata da ``feature`` e da lista de ``language_hints`` (ordem-sensível,
Requirement 7.3) — assim a mesma imagem pode coexistir no cache com
respostas distintas para ``TEXT_DETECTION`` e ``DOCUMENT_TEXT_DETECTION``
sem colisão semântica.

Características explícitas:

* Não há TTL nem invalidação por versão (Requirement 7.8). Entradas
  válidas vivem indefinidamente; o operador limpa manualmente quando
  necessário.
* Corrupção de uma única entrada (JSON inválido em qualquer dos dois
  arquivos) é absorvida silenciosamente: ``get`` retorna ``None`` e nunca
  remove arquivos vizinhos (Requirement 7.7). A entrada será sobrescrita
  pelo próximo ``put`` da mesma chave.
* Toda I/O usa ``encoding="utf-8"`` explícito e ``pathlib.Path``, em
  conformidade com ``AGENTS.md``.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable
import json


@dataclass(slots=True)
class GcvCache:
    """Cache em disco indexado por SHA-256 da imagem enviada.

    Attributes:
        cache_dir: Diretório raiz onde os pares ``<sha>.json`` /
            ``<sha>.meta.json`` são persistidos. É criado sob demanda em
            ``put`` se ainda não existir.
    """

    cache_dir: Path

    # ------------------------------------------------------------------
    # API pública
    # ------------------------------------------------------------------

    def get(
        self,
        sha256: str,
        feature: str,
        language_hints: Iterable[str],
    ) -> dict | None:
        """Retorna a resposta cacheada se existir e for compatível.

        Lê ``<sha256>.meta.json`` primeiro: a entrada é considerada
        compatível se ``meta["feature"]`` for idêntico à ``feature``
        solicitada e ``meta["language_hints"]`` (convertido para lista)
        bater elemento-a-elemento com ``language_hints`` (convertido para
        lista). A comparação preserva ordem para refletir que a GCV trata
        ``language_hints`` como lista de prioridade.

        Args:
            sha256: Hash hexadecimal dos bytes PNG enviados à API.
            feature: Modalidade da request (``TEXT_DETECTION`` ou
                ``DOCUMENT_TEXT_DETECTION``).
            language_hints: Hints BCP-47 enviados na request. Aceita
                ``list``, ``tuple`` ou qualquer iterável; a comparação é
                feita após conversão para ``list``.

        Returns:
            ``dict`` com a resposta crua quando há entrada compatível.
            ``None`` quando: a entrada não existe, ``feature`` ou
            ``language_hints`` divergem, ou um dos arquivos está
            corrompido (JSON inválido). Em todos os casos, nenhum arquivo
            é removido — o cache vizinho permanece íntegro
            (Requirement 7.7).
        """

        meta_path = self._meta_path(sha256)
        json_path = self._json_path(sha256)

        # Ausência de qualquer um dos dois arquivos é uma "não-entrada"
        # — não tratamos como erro. ``Path.exists`` é seguro para paths
        # com acentos no Windows porque ``pathlib`` delega ao stat nativo.
        if not meta_path.is_file() or not json_path.is_file():
            return None

        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            # ``ValueError`` cobre ``json.JSONDecodeError``; ``OSError``
            # cobre falhas de leitura raras (ex.: arquivo truncado).
            # Em ambos os casos descartamos APENAS esta entrada e
            # devolvemos ``None`` — o operador pode regravar via novo
            # ``put`` ou limpar manualmente.
            return None

        # Filtro de compatibilidade exato (Requirement 7.3). Convertemos
        # ``language_hints`` em ``list`` em ambos os lados para que
        # entradas geradas por ``put`` (que armazena lista) batam mesmo
        # quando o caller passa ``tuple``.
        cached_hints = meta.get("language_hints")
        if not isinstance(cached_hints, list):
            return None
        if meta.get("feature") != feature:
            return None
        if cached_hints != list(language_hints):
            return None

        try:
            return json.loads(json_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            # Resposta corrompida — mesmo critério da meta.
            return None

    def put(
        self,
        sha256: str,
        feature: str,
        language_hints: Iterable[str],
        response_json: dict,
        image_size_bytes: int,
    ) -> Path:
        """Persiste a resposta crua e os metadados associados.

        Cria ``cache_dir`` se ele ainda não existir (``parents=True``,
        ``exist_ok=True``). Grava primeiro o ``.json`` e depois o
        ``.meta.json``; ambas as escritas usam ``ensure_ascii=False`` e
        ``indent=2`` para facilitar inspeção manual.

        Args:
            sha256: Hash hexadecimal dos bytes PNG enviados à API. Será
                usado como nome base dos dois arquivos.
            feature: Modalidade efetivamente enviada na request.
            language_hints: Hints BCP-47 enviados na request. Convertidos
                para ``list`` antes de serializar para JSON, mesmo quando
                a entrada é ``tuple`` — JSON não tem tipo tupla.
            response_json: Resposta da API serializável como JSON.
            image_size_bytes: Tamanho dos bytes PNG enviados, registrado
                no ``.meta.json`` para auditoria/diagnóstico.

        Returns:
            ``Path`` absoluto do arquivo ``<sha256>.json`` recém-gravado.
        """

        # ``mkdir`` com ``parents=True`` garante que diretórios
        # intermediários sejam criados (ex.: ``extractions/`` ainda não
        # existe quando o projeto é clonado limpo).
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        json_path = self._json_path(sha256)
        meta_path = self._meta_path(sha256)

        # Resposta crua: gravamos primeiro o conteúdo "pesado" porque
        # ``get`` lê a meta antes do JSON; se algum erro de escrita
        # ocorrer aqui, a meta nunca aparecerá e a entrada inteira fica
        # ausente no próximo ``get``, que devolve ``None`` corretamente.
        json_path.write_text(
            json.dumps(response_json, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        # Timestamp UTC ISO-8601 (com offset ``+00:00``) facilita parsing
        # determinístico em ferramentas externas. ``language_hints`` é
        # explicitamente convertido para ``list`` para que a serialização
        # JSON seja simétrica entre ``put`` e ``get``.
        meta = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "feature": feature,
            "language_hints": list(language_hints),
            "image_size_bytes": int(image_size_bytes),
        }
        meta_path.write_text(
            json.dumps(meta, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        return json_path

    # ------------------------------------------------------------------
    # Helpers privados
    # ------------------------------------------------------------------

    def _json_path(self, sha256: str) -> Path:
        """Caminho absoluto do arquivo de resposta para a chave dada."""

        return self.cache_dir / f"{sha256}.json"

    def _meta_path(self, sha256: str) -> Path:
        """Caminho absoluto do arquivo de metadados para a chave dada."""

        return self.cache_dir / f"{sha256}.meta.json"
