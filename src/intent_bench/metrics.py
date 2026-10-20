"""Objective primary metrics calculation for single-intent, multi-intent, and multi-task runs."""

from __future__ import annotations

from collections import Counter
from typing import Any


def _percentile(sorted_vals: list[float], q: float) -> float:
    if not sorted_vals:
        return 0.0
    idx = (len(sorted_vals) - 1) * q
    lo = int(idx)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = idx - lo
    return round(sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac, 2)


def compute_single_intent_metrics(
    records: list[dict[str, Any]],
    has_oos: bool = False,
    oos_key: str = "oos",
) -> dict[str, Any]:
    """Compute primary single-intent classification, OOS, format, latency, and token metrics."""
    n = len(records)
    if n == 0:
        return {"count": 0}

    correct = sum(1 for r in records if r.get("predicted_label") == r.get("ground_truth"))
    accuracy = correct / n

    # Macro-F1 across all classes present in ground truth
    gold_classes = sorted({str(r["ground_truth"]) for r in records})
    tp: Counter[str] = Counter()
    fp: Counter[str] = Counter()
    fn: Counter[str] = Counter()

    for r in records:
        g = str(r["ground_truth"])
        p = str(r.get("predicted_label", ""))
        if p == g:
            tp[g] += 1
        else:
            fp[p] += 1
            fn[g] += 1

    f1_per_class: list[float] = []
    for c in gold_classes:
        denom = 2 * tp[c] + fp[c] + fn[c]
        f1_per_class.append((2 * tp[c] / denom) if denom > 0 else 0.0)
    macro_f1 = sum(f1_per_class) / len(f1_per_class) if f1_per_class else 0.0

    # Format compliance & hallucination (Variable 3)
    strict_ok = sum(1 for r in records if r.get("strict_format_ok", True))
    fallback_ok = sum(1 for r in records if r.get("fallback_extracted", False))
    hallucinated = sum(1 for r in records if r.get("hallucinated_key", False))

    # Latency & Tokens
    latencies = sorted(float(r.get("latency_ms", 0.0)) for r in records if r.get("latency_ms") is not None)
    computes = sorted(float(r["compute_ms"]) for r in records if r.get("compute_ms") is not None)

    result: dict[str, Any] = {
        "count": n,
        "accuracy": round(accuracy * 100.0, 2),
        "macro_f1": round(macro_f1 * 100.0, 2),
        "strict_format_rate": round(strict_ok / n * 100.0, 2),
        "fallback_extract_rate": round(fallback_ok / n * 100.0, 2),
        "hallucination_rate": round(hallucinated / n * 100.0, 2),
        "latency_p50_ms": _percentile(latencies, 0.50),
        "latency_p90_ms": _percentile(latencies, 0.90),
        "latency_p99_ms": _percentile(latencies, 0.99),
        "compute_p50_ms": _percentile(computes, 0.50) if computes else None,
        "total_input_tokens": sum(int(r.get("input_tokens", 0)) for r in records),
        "total_output_tokens": sum(int(r.get("output_tokens", 0)) for r in records),
        "total_reasoning_tokens": sum(int(r.get("reasoning_tokens", 0)) for r in records),
        "total_cached_tokens": sum(int(r.get("cached_tokens", 0)) for r in records),
    }

    if has_oos:
        in_scope_recs = [r for r in records if r["ground_truth"] != oos_key]
        in_scope_acc = (
            sum(1 for r in in_scope_recs if r.get("predicted_label") == r["ground_truth"])
            / len(in_scope_recs)
            if in_scope_recs
            else 0.0
        )
        oos_tp = tp[oos_key]
        oos_fp = fp[oos_key]
        oos_fn = fn[oos_key]
        oos_prec = oos_tp / (oos_tp + oos_fp) if (oos_tp + oos_fp) > 0 else 0.0
        oos_rec = oos_tp / (oos_tp + oos_fn) if (oos_tp + oos_fn) > 0 else 0.0
        oos_f1 = (
            2 * oos_prec * oos_rec / (oos_prec + oos_rec)
            if (oos_prec + oos_rec) > 0
            else 0.0
        )
        result.update(
            {
                "in_scope_accuracy": round(in_scope_acc * 100.0, 2),
                "oos_precision": round(oos_prec * 100.0, 2),
                "oos_recall": round(oos_rec * 100.0, 2),
                "oos_f1": round(oos_f1 * 100.0, 2),
            }
        )

    return result


def compute_multi_intent_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute Exact Match (Subset Accuracy), Micro-F1, and Macro-F1 for multi-intent records."""
    n = len(records)
    if n == 0:
        return {"count": 0}

    exact_matches = 0
    tp: Counter[str] = Counter()
    fp: Counter[str] = Counter()
    fn: Counter[str] = Counter()
    all_gold_labels: set[str] = set()

    for r in records:
        gold_set = set(r.get("ground_truth_labels", []))
        pred_set = set(r.get("predicted_labels", []))
        all_gold_labels.update(gold_set)
        if gold_set == pred_set:
            exact_matches += 1
        for k in gold_set & pred_set:
            tp[k] += 1
        for k in pred_set - gold_set:
            fp[k] += 1
        for k in gold_set - pred_set:
            fn[k] += 1

    total_tp = sum(tp.values())
    total_fp = sum(fp.values())
    total_fn = sum(fn.values())
    micro_prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    micro_rec = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    micro_f1 = (
        2 * micro_prec * micro_rec / (micro_prec + micro_rec)
        if (micro_prec + micro_rec) > 0
        else 0.0
    )

    f1_per_class = []
    for c in sorted(all_gold_labels):
        denom = 2 * tp[c] + fp[c] + fn[c]
        f1_per_class.append((2 * tp[c] / denom) if denom > 0 else 0.0)
    macro_f1 = sum(f1_per_class) / len(f1_per_class) if f1_per_class else 0.0

    strict_ok = sum(1 for r in records if r.get("strict_format_ok", True))
    hallucinated = sum(1 for r in records if r.get("hallucinated_key", False))
    latencies = sorted(float(r.get("latency_ms", 0.0)) for r in records if r.get("latency_ms") is not None)
    computes = sorted(float(r["compute_ms"]) for r in records if r.get("compute_ms") is not None)

    return {
        "count": n,
        "exact_match": round(exact_matches / n * 100.0, 2),
        "micro_f1": round(micro_f1 * 100.0, 2),
        "macro_f1": round(macro_f1 * 100.0, 2),
        "micro_precision": round(micro_prec * 100.0, 2),
        "micro_recall": round(micro_rec * 100.0, 2),
        "strict_format_rate": round(strict_ok / n * 100.0, 2),
        "hallucination_rate": round(hallucinated / n * 100.0, 2),
        "latency_p50_ms": _percentile(latencies, 0.50),
        "latency_p90_ms": _percentile(latencies, 0.90),
        "latency_p99_ms": _percentile(latencies, 0.99),
        "compute_p50_ms": _percentile(computes, 0.50) if computes else None,
        "total_input_tokens": sum(int(r.get("input_tokens", 0)) for r in records),
        "total_output_tokens": sum(int(r.get("output_tokens", 0)) for r in records),
        "total_reasoning_tokens": sum(int(r.get("reasoning_tokens", 0)) for r in records),
        "total_cached_tokens": sum(int(r.get("cached_tokens", 0)) for r in records),
    }
