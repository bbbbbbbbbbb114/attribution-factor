import pytest

from attribution_experiments.bootstrap_pilot import parse_annotation
from attribution_experiments.common.result_validity import retryable_result


@pytest.mark.parametrize("text", [
    '{"classification":"uncertain"}',
    '```json\n{"classification":"uncertain"}\n```',
    'Review result\n{"classification":"uncertain"}\nAdditional discussion',
])
def test_annotation_json_survives_provider_wrapping(text):
    assert parse_annotation(text) == {"classification": "uncertain"}


@pytest.mark.parametrize("text", ["", "No usable label", '{"classification":"made_up"}'])
def test_missing_or_unsupported_annotation_is_explicit(text):
    with pytest.raises(ValueError):
        parse_annotation(text)


def test_allocation_quota_errors_are_infrastructure_failures():
    assert retryable_result({"exit_status": "RateLimitError", "exception": "Allocated quota exceeded, please increase your quota limit"})
