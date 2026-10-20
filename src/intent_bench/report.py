"""Report & dashboard generator from raw JSONL checkpoints."""

from __future__ import annotations

import csv
import html
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from .config import DOCS_DIR, MODELS, REPORTS_DIR, RESULTS_RAW_DIR, RESULTS_SUMMARY_DIR
from .metrics import compute_multi_intent_metrics, compute_single_intent_metrics
from .runner import load_completed_records


def collect_all_run_summaries(
    raw_dir: Path = RESULTS_RAW_DIR,
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    """Scan all `.jsonl` checkpoint files and compute summary metrics per run."""
    summaries: list[dict[str, Any]] = []
    records_by_run: dict[str, list[dict[str, Any]]] = {}

    if not raw_dir.is_dir():
        return summaries, records_by_run

    for path in sorted(raw_dir.glob("*.jsonl")):
        completed = load_completed_records(path)
        records = list(completed.values())
        if not records:
            continue
        run_id = path.stem
        records_by_run[run_id] = records

        first = records[0]
        dataset_id = first.get("dataset_id", "")
        task_type = first.get("task_type", "single_intent")
        criteria_mode = first.get("criteria_mode", "rich")
        model_slug = first.get("model_slug", "")
        model_spec = MODELS.get(model_slug)
        model_display = model_spec.display_name if model_spec else model_slug

        if task_type == "multi_intent":
            metrics = compute_multi_intent_metrics(records)
        else:
            has_oos = dataset_id == "clinc150"
            metrics = compute_single_intent_metrics(records, has_oos=has_oos)

        summaries.append(
            {
                "run_id": run_id,
                "dataset_id": dataset_id,
                "task_type": task_type,
                "criteria_mode": criteria_mode,
                "model_slug": model_slug,
                "model_display": model_display,
                **metrics,
            }
        )

    return summaries, records_by_run


def export_summary_csv(
    summaries: list[dict[str, Any]],
    summary_dir: Path = RESULTS_SUMMARY_DIR,
) -> Path:
    """Export all run metrics to `results/summary/benchmark_summary.csv`."""
    summary_dir.mkdir(parents=True, exist_ok=True)
    csv_path = summary_dir / "benchmark_summary.csv"
    if not summaries:
        csv_path.write_text("", encoding="utf-8")
        return csv_path

    fieldnames: list[str] = []
    for row in summaries:
        for k in row.keys():
            if k not in fieldnames:
                fieldnames.append(k)

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in summaries:
            writer.writerow(row)
    return csv_path


def find_comparison_bad_cases(
    records_by_run: dict[str, list[dict[str, Any]]],
    dataset_id: str,
    criteria_mode: str = "rich",
    d3_slug: str = "d3-turbo",
    glm_slug: str = "glm-5.3-low",
    max_examples: int = 12,
) -> list[dict[str, Any]]:
    """Find illustrative disagreements where d3-turbo and GLM 5.3 differed."""
    d3_run = f"{dataset_id}__{criteria_mode}__{d3_slug}"
    glm_run = f"{dataset_id}__{criteria_mode}__{glm_slug}"
    if d3_run not in records_by_run or glm_run not in records_by_run:
        return []

    d3_map = {r["sample_id"]: r for r in records_by_run[d3_run]}
    glm_map = {r["sample_id"]: r for r in records_by_run[glm_run]}

    cases: list[dict[str, Any]] = []
    for sid, d3_r in d3_map.items():
        glm_r = glm_map.get(sid)
        if not glm_r:
            continue
        gold = d3_r.get("ground_truth")
        d3_pred = d3_r.get("predicted_label")
        glm_pred = glm_r.get("predicted_label")
        if (d3_pred == gold and glm_pred != gold) or (d3_pred != gold and glm_pred == gold):
            cases.append(
                {
                    "sample_id": sid,
                    "text": d3_r.get("text", ""),
                    "ground_truth": gold,
                    "d3_pred": d3_pred,
                    "d3_correct": d3_pred == gold,
                    "glm_pred": glm_pred,
                    "glm_correct": glm_pred == gold,
                }
            )
            if len(cases) >= max_examples:
                break
    return cases


def generate_markdown_and_html_reports(
    raw_dir: Path = RESULTS_RAW_DIR,
    docs_dir: Path = DOCS_DIR,
    reports_dir: Path = REPORTS_DIR,
) -> tuple[Path, Path, Path]:
    """Generate Markdown report, HTML visual dashboard, and CSV summary from completed runs."""
    summaries, records_by_run = collect_all_run_summaries(raw_dir=raw_dir)
    csv_path = export_summary_csv(summaries)

    docs_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    md_path = docs_dir / "mobile-banking-intent-benchmark-report.md"
    html_path = reports_dir / "intent-benchmark-dashboard.html"

    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines: list[str] = [
        "# 手机银行与跨领域意图识别模型选型对比报告 (`vllm-sr/d3` 家族 vs. `GLM 5.3`)",
        "",
        f"**生成时间：** `{now_str}`  ",
        "**评测原则：** 100% 官方全量测试集（零抽样偏差）、双盲同构提示词（信息完全对称）、仅以核心客观主指标（Top-1 Accuracy、Macro-F1、多意图 Exact Match / Micro-F1、延迟 P50/P90/P99、Token 消耗）进行正面对比。",
        "",
        "---",
        "",
        "## 一、单意图识别主榜汇总 (`single_intent` + `multitask`)",
        "",
        "| 数据集 | 规则模式 (`criteria_mode`) | 模型 | 已测样本数 | Top-1 准确率 (%) | Macro-F1 (%) | 严格格式合规率 (%) | 目录外幻觉率 (%) | P50 延迟 (ms) | P90 延迟 (ms) | 总输入 Tokens | 总输出 Tokens |",
        "| :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    single_rows = [s for s in summaries if s["task_type"] in ("single_intent", "multitask")]
    for s in single_rows:
        lines.append(
            f"| `{s['dataset_id']}` | `{s['criteria_mode']}` | **{s['model_display']}** | "
            f"{s['count']} | **{s.get('accuracy', 0):.2f}%** | {s.get('macro_f1', 0):.2f}% | "
            f"{s.get('strict_format_rate', 100):.2f}% | {s.get('hallucination_rate', 0):.2f}% | "
            f"{s.get('latency_p50_ms', 0):.1f} | {s.get('latency_p90_ms', 0):.1f} | "
            f"{s.get('total_input_tokens', 0):,} | {s.get('total_output_tokens', 0):,} |"
        )

    multi_rows = [s for s in summaries if s["task_type"] == "multi_intent"]
    if multi_rows:
        lines.extend(
            [
                "",
                "---",
                "",
                "## 二、手机银行多意图识别专项汇总 (`minds14_zh_multi`, 并发 `noul` vs. `GLM 5.3` 数组)",
                "",
                "| 数据集 | 模型 | 样本数 | 完全匹配率 Exact Match (%) | Micro-F1 (%) | Macro-F1 (%) | P50 延迟 (ms) | P90 延迟 (ms) | 总输入 Tokens | 总输出 Tokens |",
                "| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for s in multi_rows:
            lines.append(
                f"| `{s['dataset_id']}` | **{s['model_display']}** | {s['count']} | "
                f"**{s.get('exact_match', 0):.2f}%** | **{s.get('micro_f1', 0):.2f}%** | "
                f"{s.get('macro_f1', 0):.2f}% | {s.get('latency_p50_ms', 0):.1f} | "
                f"{s.get('latency_p90_ms', 0):.1f} | {s.get('total_input_tokens', 0):,} | "
                f"{s.get('total_output_tokens', 0):,} |"
            )

    oos_rows = [s for s in summaries if "oos_f1" in s]
    if oos_rows:
        lines.extend(
            [
                "",
                "---",
                "",
                "## 三、`CLINC150` 越界闲聊拒识能力 (`Out-of-Scope Rejection`) 专项对比",
                "",
                "| 模型 | 域内准确率 In-Scope Acc (%) | 越界拒识精确率 OOS Precision (%) | 越界拒识召回率 OOS Recall (%) | **越界拒识 OOS F1 (%)** |",
                "| :--- | ---: | ---: | ---: | ---: |",
            ]
        )
        for s in oos_rows:
            lines.append(
                f"| **{s['model_display']}** | {s.get('in_scope_accuracy', 0):.2f}% | "
                f"{s.get('oos_precision', 0):.2f}% | {s.get('oos_recall', 0):.2f}% | "
                f"**{s.get('oos_f1', 0):.2f}%** |"
            )

    # Add Bad Case Comparisons if present
    for ds_id in ("minds14_zh", "banking77"):
        cases = find_comparison_bad_cases(records_by_run, ds_id)
        if cases:
            lines.extend(
                [
                    "",
                    "---",
                    "",
                    f"## 四、`{ds_id}` 典型分歧案例对照表 (`d3-turbo` vs. `GLM-5.3`)",
                    "",
                    "| 样本 ID | 用户原话 | 标准答案 (`ground_truth`) | `d3-turbo` 预测 | `GLM-5.3 (low)` 预测 |",
                    "| :--- | :--- | :--- | :--- | :--- |",
                ]
            )
            for c in cases:
                d3_mark = "✓" if c["d3_correct"] else "✗"
                glm_mark = "✓" if c["glm_correct"] else "✗"
                text_clean = str(c["text"]).replace("|", "\\|").replace("\n", " ")
                lines.append(
                    f"| `{c['sample_id']}` | {text_clean} | `{c['ground_truth']}` | "
                    f"`{c['d3_pred']}` ({d3_mark}) | `{c['glm_pred']}` ({glm_mark}) |"
                )

    md_content = "\n".join(lines) + "\n"
    md_path.write_text(md_content, encoding="utf-8")

    # Build standalone HTML dashboard
    summary_json = html.escape(json.dumps(summaries, ensure_ascii=False))
    html_content = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <title>手机银行与跨领域意图识别评测看板 (d3 vs GLM 5.3)</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>
    :root {{
      --bg: #0b1020;
      --card: #131b33;
      --border: #26335d;
      --text: #e8eefc;
      --muted: #9aa8c7;
      --accent: #4f8cff;
      --good: #2dd4bf;
    }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      background: var(--bg);
      color: var(--text);
      margin: 0;
      padding: 32px 24px;
    }}
    .wrap {{ max-width: 1280px; margin: 0 auto; }}
    h1 {{ font-size: 24px; margin-bottom: 8px; }}
    .sub {{ color: var(--muted); font-size: 14px; margin-bottom: 24px; }}
    .card {{
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 20px;
      margin-bottom: 24px;
      overflow-x: auto;
    }}
    table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
    th, td {{ padding: 10px 12px; border-bottom: 1px solid var(--border); text-align: left; }}
    th {{ color: var(--muted); font-weight: 600; }}
    td.num, th.num {{ text-align: right; font-variant-numeric: tabular-nums; }}
    .badge {{
      display: inline-block;
      padding: 2px 8px;
      border-radius: 999px;
      background: rgba(79, 140, 255, 0.18);
      color: #8bb4ff;
      font-size: 12px;
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <h1>手机银行与跨领域意图识别选型评测看板 (`vllm-sr/d3` 家族 vs `GLM 5.3`)</h1>
    <p class="sub">更新时间：{html.escape(now_str)} · 100% 官方全量测试集 · 零抽样偏差</p>
    <div class="card">
      <h2>全量运行指标汇总</h2>
      <div id="table-root"></div>
    </div>
  </div>
  <script id="summary-data" type="application/json">{summary_json}</script>
  <script>
    const data = JSON.parse(document.getElementById('summary-data').textContent || '[]');
    const root = document.getElementById('table-root');
    if (!data.length) {{
      root.innerHTML = '<p style="color:#9aa8c7">暂无已完成的评测运行记录。运行 <code>python scripts/run_benchmark.py</code> 后将自动渲染结果。</p>';
    }} else {{
      let htmlStr = '<table><thead><tr><th>数据集</th><th>模式</th><th>模型</th><th class="num">样本数</th><th class="num">主准确率 / ExactMatch</th><th class="num">Macro-F1</th><th class="num">格式合规率</th><th class="num">幻觉率</th><th class="num">P50 延迟 (ms)</th><th class="num">P90 延迟 (ms)</th><th class="num">输入 Tokens</th><th class="num">输出 Tokens</th></tr></thead><tbody>';
      for (const r of data) {{
        const mainMetric = r.task_type === 'multi_intent' ? (r.exact_match + '% (EM)') : (r.accuracy + '%');
        htmlStr += `<tr>
          <td><span class="badge">${{r.dataset_id}}</span></td>
          <td><code>${{r.criteria_mode}}</code></td>
          <td><strong>${{r.model_display}}</strong></td>
          <td class="num">${{r.count}}</td>
          <td class="num" style="color:#2dd4bf;font-weight:600">${{mainMetric}}</td>
          <td class="num">${{r.macro_f1 ?? '-'}}%</td>
          <td class="num">${{r.strict_format_rate ?? 100}}%</td>
          <td class="num">${{r.hallucination_rate ?? 0}}%</td>
          <td class="num">${{r.latency_p50_ms ?? '-'}}</td>
          <td class="num">${{r.latency_p90_ms ?? '-'}}</td>
          <td class="num">${{(r.total_input_tokens || 0).toLocaleString()}}</td>
          <td class="num">${{(r.total_output_tokens || 0).toLocaleString()}}</td>
        </tr>`;
      }}
      htmlStr += '</tbody></table>';
      root.innerHTML = htmlStr;
    }}
  </script>
</body>
</html>
"""
    html_path.write_text(html_content, encoding="utf-8")
    return md_path, html_path, csv_path
