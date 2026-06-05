"""Smoke check para garantir que cada gerador de strategies.py produz
valores sem erros. Não é um teste pytest formal — é uma ferramenta
descartável invocada por execute_pwsh durante a tarefa 1.3.
"""

from __future__ import annotations

from hypothesis import given, settings, HealthCheck
from hypothesis import strategies as st

from tests.ocr_engine.gcv.strategies import (
    bcp47_hints,
    cache_states,
    error_class_subsets,
    feature_strings_invalid,
    gcv_response_dict,
    gcv_response_with_non_numeric_conf,
    image_arrays,
    kind_strings_invalid,
    rate_limiter_event_sequences,
)


def _exercise(name: str, strategy):
    @given(strategy)
    @settings(
        max_examples=20,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def runner(value):
        # Apenas exercita o gerador; a validação é "não levantar".
        assert value is not None or value == [] or value == {} or value == ""

    runner()
    print(f"  ok: {name}")


def main() -> None:
    print("Exercitando geradores Hypothesis de tests/gcv/strategies.py...")
    _exercise("gcv_response_dict", gcv_response_dict())
    _exercise("gcv_response_with_non_numeric_conf", gcv_response_with_non_numeric_conf())
    _exercise("bcp47_hints", bcp47_hints())
    _exercise("image_arrays", image_arrays())
    _exercise("kind_strings_invalid", kind_strings_invalid())
    _exercise("feature_strings_invalid", feature_strings_invalid())
    _exercise("error_class_subsets", error_class_subsets())
    _exercise("rate_limiter_event_sequences", rate_limiter_event_sequences())
    _exercise("cache_states", cache_states())
    print("Todos os geradores produzem valores sem erros.")


if __name__ == "__main__":
    main()
