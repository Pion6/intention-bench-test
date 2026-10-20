"""Command-line interface for running the Mobile Banking & Cross-Domain Intent Benchmark."""

from __future__ import annotations

import argparse
import asyncio
from typing import Sequence

from .config import MODELS, RuntimeConfig
from .datasets import load_dataset_by_id
from .report import generate_markdown_and_html_reports
from .runner import ExperimentJob, build_stage_jobs, run_experiment_job


ALL_DATASETS = (
    "minds14_zh",
    "banking77",
    "massive_zh",
    "clinc150",
    "minds14_zh_multi",
    "minds14_zh_multitask",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="intent-bench",
        description="Mobile Banking & Cross-Domain Intent Recognition Benchmark (vllm-sr/d3 vs. GLM 5.3)",
    )
    parser.add_argument(
        "--stage",
        choices=["stage2", "stage3", "stage4", "banking", "general", "all"],
        default="stage2",
        help="Preset stage to execute (default: stage2 = MINDS-14 full + banking variables)",
    )
    parser.add_argument(
        "--dataset",
        choices=list(ALL_DATASETS),
        help="Run a single specific dataset instead of a preset stage",
    )
    parser.add_argument(
        "--model",
        choices=list(MODELS.keys()),
        help="Filter to a single model slug (e.g. d3-turbo, d3-flash, d3-mini, glm-5.3-low)",
    )
    parser.add_argument(
        "--criteria-mode",
        choices=["rich", "label_only"],
        default="rich",
        help="Criteria richness mode when --dataset is specified (default: rich)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional sample cap per job for quick verification (omit for 100%% full test split)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=None,
        help="Override async request concurrency",
    )
    parser.add_argument(
        "--download-only",
        action="store_true",
        help="Only download and cache the datasets locally without calling model APIs",
    )
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Only regenerate Markdown/HTML/CSV reports from existing JSONL checkpoints",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the planned experiment jobs without executing API calls",
    )
    return parser.parse_args(argv)


async def _async_main(args: argparse.Namespace) -> int:
    if args.report_only:
        md_path, html_path, csv_path = generate_markdown_and_html_reports()
        print(f"Generated Markdown report: {md_path}")
        print(f"Generated HTML dashboard:  {html_path}")
        print(f"Generated CSV summary:     {csv_path}")
        return 0

    if args.download_only:
        targets = [args.dataset] if args.dataset else list(ALL_DATASETS)
        for ds_id in targets:
            bundle = load_dataset_by_id(ds_id)
            print(
                f"Cached [{bundle.dataset_id}] {bundle.display_name}: "
                f"{len(bundle.samples)} samples, {len(bundle.criteria_rich)} intents"
            )
        return 0

    if args.dataset:
        models = [args.model] if args.model else list(MODELS.keys())
        jobs = [ExperimentJob(args.dataset, args.criteria_mode, m) for m in models]
    else:
        jobs = build_stage_jobs(args.stage)
        if args.model:
            jobs = [j for j in jobs if j.model_slug == args.model]

    if args.dry_run:
        print(f"Planned Experiment Jobs ({len(jobs)} total):")
        for idx, job in enumerate(jobs, 1):
            m = MODELS[job.model_slug]
            print(
                f"  {idx:02d}. {job.run_id:<48} | provider={m.provider:<8} | model={m.display_name}"
            )
        return 0

    runtime_cfg = RuntimeConfig.from_env()
    for job in jobs:
        await run_experiment_job(
            job=job,
            runtime_cfg=runtime_cfg,
            limit=args.limit,
            concurrency_override=args.concurrency,
        )
        generate_markdown_and_html_reports()

    md_path, html_path, csv_path = generate_markdown_and_html_reports()
    print(f"\nAll jobs finished. Reports updated:\n  - {md_path}\n  - {html_path}\n  - {csv_path}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    return asyncio.run(_async_main(args))


if __name__ == "__main__":
    raise SystemExit(main())
