"""Offline unit tests for prompt symmetry, output parsing, and metrics computation."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from intent_bench.catalogs import (  # noqa: E402
    BANKING77_INTENT_KEYS,
    BANKING77_RICH_CRITERIA,
    MINDS14_INTENT_KEYS,
    MINDS14_RICH_CRITERIA,
    build_d3_multi_intent_questions,
    build_d3_multitask_banking_questions,
    build_d3_single_intent_questions,
    build_glm_single_intent_messages,
)
from intent_bench.clients import (  # noqa: E402
    parse_glm_multi_intent,
    parse_glm_multitask,
    parse_glm_single_intent,
)
from intent_bench.metrics import (  # noqa: E402
    compute_multi_intent_metrics,
    compute_single_intent_metrics,
)
from intent_bench.runner import build_stage_jobs  # noqa: E402


def test_catalog_counts_and_symmetry() -> None:
    assert len(MINDS14_INTENT_KEYS) == 14
    assert len(MINDS14_RICH_CRITERIA) == 14
    assert len(BANKING77_INTENT_KEYS) == 77
    assert len(BANKING77_RICH_CRITERIA) == 77

    instructions = "请判断该手机银行客户提问所属的业务意图类别。"
    user_text = "我的银行卡在ATM机被吞了怎么办？"
    d3_q = build_d3_single_intent_questions(instructions, MINDS14_RICH_CRITERIA)
    glm_msgs = build_glm_single_intent_messages(
        instructions, MINDS14_RICH_CRITERIA, user_text
    )

    assert d3_q["intent"]["type"] == "choice"
    assert d3_q["intent"]["criteria"] == MINDS14_RICH_CRITERIA
    for k, desc in MINDS14_RICH_CRITERIA.items():
        assert f"- {k}: {desc}" in glm_msgs[0]["content"]
    assert user_text in glm_msgs[1]["content"]


def test_multi_intent_and_multitask_builders() -> None:
    multi_q = build_d3_multi_intent_questions(MINDS14_RICH_CRITERIA)
    assert len(multi_q) == 14
    assert all(v["type"] == "noul" for v in multi_q.values())

    mt_q = build_d3_multitask_banking_questions(MINDS14_RICH_CRITERIA)
    assert set(mt_q.keys()) == {"intent", "urgency", "high_risk", "human_handoff"}
    assert mt_q["intent"]["type"] == "choice"
    assert mt_q["urgency"]["type"] == "score"
    assert mt_q["high_risk"]["type"] == "noul"


def test_glm_parsers() -> None:
    valid = set(MINDS14_INTENT_KEYS)

    # Strict JSON
    r1 = parse_glm_single_intent('{"intent": "card_issues"}', valid)
    assert r1["predicted_label"] == "card_issues"
    assert r1["strict_format_ok"] is True
    assert r1["hallucinated_key"] is False

    # Fenced JSON fallback
    r2 = parse_glm_single_intent('```json\n{"intent": "balance"}\n```', valid)
    assert r2["predicted_label"] == "balance"
    assert r2["strict_format_ok"] is False
    assert r2["fallback_extracted"] is True
    assert r2["hallucinated_key"] is False

    # Hallucinated out-of-catalog key
    r3 = parse_glm_single_intent('{"intent": "non_existent_intent"}', valid)
    assert r3["hallucinated_key"] is True

    # Multi-intent parse
    m1 = parse_glm_multi_intent('{"intents": ["balance", "atm_limit"]}', valid)
    assert m1["predicted_labels"] == ["atm_limit", "balance"]
    assert m1["strict_format_ok"] is True

    # Multi-task parse
    mt1 = parse_glm_multitask(
        '{"intent": "freeze", "urgency": "Urgent", "high_risk": true, "human_handoff": true}',
        valid,
    )
    assert mt1["predicted_label"] == "freeze"
    assert mt1["strict_format_ok"] is True
    assert mt1["high_risk"] is True


def test_metrics_computation() -> None:
    single_recs = [
        {
            "ground_truth": "balance",
            "predicted_label": "balance",
            "strict_format_ok": True,
            "hallucinated_key": False,
            "latency_ms": 80.0,
            "input_tokens": 200,
            "output_tokens": 0,
        },
        {
            "ground_truth": "oos",
            "predicted_label": "oos",
            "strict_format_ok": True,
            "hallucinated_key": False,
            "latency_ms": 90.0,
            "input_tokens": 200,
            "output_tokens": 0,
        },
        {
            "ground_truth": "freeze",
            "predicted_label": "card_issues",
            "strict_format_ok": False,
            "hallucinated_key": False,
            "latency_ms": 120.0,
            "input_tokens": 200,
            "output_tokens": 10,
        },
    ]
    m = compute_single_intent_metrics(single_recs, has_oos=True, oos_key="oos")
    assert m["count"] == 3
    assert m["accuracy"] == 66.67
    assert m["oos_f1"] == 100.0

    multi_recs = [
        {
            "ground_truth_labels": ["atm_limit", "balance"],
            "predicted_labels": ["atm_limit", "balance"],
            "strict_format_ok": True,
            "hallucinated_key": False,
            "latency_ms": 110.0,
            "input_tokens": 300,
            "output_tokens": 0,
        },
        {
            "ground_truth_labels": ["card_issues", "freeze"],
            "predicted_labels": ["card_issues"],
            "strict_format_ok": True,
            "hallucinated_key": False,
            "latency_ms": 130.0,
            "input_tokens": 300,
            "output_tokens": 10,
        },
    ]
    mm = compute_multi_intent_metrics(multi_recs)
    assert mm["count"] == 2
    assert mm["exact_match"] == 50.0
    assert mm["micro_precision"] == 100.0
    assert mm["micro_recall"] == 75.0


def test_stage_jobs_builder() -> None:
    s2 = build_stage_jobs("stage2")
    assert len(s2) == 18
    all_jobs = build_stage_jobs("all")
    assert len(all_jobs) == 36


if __name__ == "__main__":
    test_catalog_counts_and_symmetry()
    test_multi_intent_and_multitask_builders()
    test_glm_parsers()
    test_metrics_computation()
    test_stage_jobs_builder()
    print("All smoke tests passed!")
