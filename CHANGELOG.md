# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Extração estruturada da tabela nutricional via LLM (`clean_nutritional_table`)
  no preview de OCR, com o parser por regex como fallback quando a LLM falha
  ou não retorna linhas.

### Changed

### Fixed

- Limpeza de ingredientes via LLM agora detecta recusas/explicações ("não há
  ingredientes...", "texto fornecido", frases longas sem vírgula) e descarta
  o resultado em vez de salvá-lo como ingrediente.
- Preview de OCR descarta um único item de ingrediente longo e sem separadores
  (provável frase/recusa), evitando que ele seja exibido como ingrediente real.

### Removed
