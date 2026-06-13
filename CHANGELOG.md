# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

### Changed

### Fixed

### Removed

## [1.1.0] - 2026-06-13

### Added

- Extração estruturada da tabela nutricional via LLM (`clean_nutritional_table`)
  no preview de OCR, com o parser por regex como fallback quando a LLM falha
  ou não retorna linhas.
- Cache de resumo em linguagem natural por produto (`ProductSummary`), gerado
  e reaproveitado por combinação de tipo de diabetes e nível de linguagem do
  usuário.
- Registro de leitura no histórico de scans ao consultar um produto já
  cadastrado (scan-on-read), com deduplicação por código de barras.
- Nome e marca do produto incluídos no prompt de geração do resumo e
  persistidos no histórico de leituras.
- ROI desabilitado por padrão no motor de OCR (`default_roi_enabled: false`),
  favorecendo o recorte manual feito no app.

### Changed

- Scans de produtos revisados/persistidos pelo usuário agora são marcados
  como `passed=true` no histórico, refletindo a qualidade após edição manual.

### Fixed

- Limpeza de ingredientes via LLM agora detecta recusas/explicações ("não há
  ingredientes...", "texto fornecido", frases longas sem vírgula) e descarta
  o resultado em vez de salvá-lo como ingrediente.
- Preview de OCR descarta um único item de ingrediente longo e sem separadores
  (provável frase/recusa), evitando que ele seja exibido como ingrediente real.

### Removed

## [1.0.0] - 2026-06-13

### Added

- API REST em FastAPI com autenticação JWT (registro, login, refresh e logout
  com revogação de tokens, alteração de senha).
- Perfil de usuário com tipo de diabetes (incluindo DMG) e nível de linguagem
  para personalização clínica.
- Motor de OCR local (Tesseract + Google Cloud Vision) com cascata de presets
  por categoria (tabela nutricional, texto livre, ingredientes).
- Endpoint `/analyze` para análise de rótulo (OCR + análise clínica de
  ingredientes para Diabetes Mellitus).
- Cadastro de produtos por código de barras (tabela nutricional + lista de
  ingredientes), com endpoints de leitura, criação e atualização.
- Endpoint de preview de OCR por duas imagens (tabela + ingredientes), sem
  persistência.
- Integração com LLM (Groq) para limpeza de texto de ingredientes e geração
  de resumo em linguagem natural da análise clínica.
- Histórico de scans por usuário, com listagem paginada e detalhe por scan.
