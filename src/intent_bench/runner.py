"""Resumable evaluation runner with append-only JSONL checkpointing."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from .clients import DecisionClient, GLMClient
from .config import MODELS, RESULTS_RAW_DIR, ModelSpec, RuntimeConfig
from .datasets import DatasetBundle, EvalSample, load_dataset_by_id


@dataclass(frozen=True)
class ExperimentJob:
    """One (dataset, criteria_mode, model) evaluation run."""

    dataset_id: str
    criteria_mode: str  # "rich" | "label_only"
    model_slug: str

    @property
    def run_id(self) -> str:
        return f"{self.dataset_id}__{self.criteria_mode}__{self.model_slug}"


def checkpoint_path(run_id: str, raw_dir: Path = RESULTS_RAW_DIR) -> Path:
    raw_dir.mkdir(parents=True, exist_ok=True)
    return raw_dir / f"{run_id}.jsonl"


def load_completed_records(path: Path) -> dict[str, dict[str, Any]]:
    """Read existing JSONL checkpoint and map `sample_id -> record`."""
    if not path.is_file():
        return {}
    completed: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
            sid = rec.get("sample_id")
            if sid and not rec.get("error"):
                completed[sid] = rec
        except Exception:
            continue
    return completed


async def run_experiment_job(
    job: ExperimentJob,
    runtime_cfg: RuntimeConfig,
    limit: int | None = None,
    concurrency_override: int | None = None,
    raw_dir: Path = RESULTS_RAW_DIR,
) -> list[dict[str, Any]]:
    """Execute or resume a single ExperimentJob and return all completed records."""
    bundle: DatasetBundle = load_dataset_by_id(job.dataset_id)
    model_spec: ModelSpec = MODELS[job.model_slug]
    criteria = (
        bundle.criteria_rich
        if job.criteria_mode == "rich"
        else bundle.criteria_label_only
    )

    samples: list[EvalSample] = (
        bundle.samples[:limit] if limit is not None else bundle.samples
    )
    ckpt_file = checkpoint_path(job.run_id, raw_dir=raw_dir)
    completed = load_completed_records(ckpt_file)

    pending = [s for s in samples if s.sample_id not in completed]
    print(
        f"[{job.run_id}] Total={len(samples)} | Completed={len(completed)} | Pending={len(pending)}"
    )
    if not pending:
        return [completed[s.sample_id] for s in samples if s.sample_id in completed]

    concurrency = concurrency_override or model_spec.default_concurrency
    sem = asyncio.Semaphore(concurrency)
    file_lock = asyncio.Lock()

    decision_client = (
        DecisionClient(runtime_cfg) if model_spec.provider == "decision" else None
    )
    glm_client = GLMClient(runtime_cfg) if model_spec.provider == "glm" else None

    timeout = httpx.Timeout(90.0, connect=20.0)
    limits = httpx.Limits(
        max_keepalive_connections=concurrency * 2,
        max_connections=concurrency * 2 + 4,
    )

    done_count = len(completed)
    total_count = len(samples)

    async with httpx.AsyncClient(timeout=timeout, limits=limits) as http_client:
        with ckpt_file.open("a", encoding="utf-8") as f_out:

            async def _worker(sample: EvalSample) -> None:
                nonlocal done_count
                async with sem:
                    if bundle.task_type == "single_intent":
                        if decision_client is not None:
                            res = await decision_client.evaluate_single_intent(
                                http_client,
                                model_spec,
                                sample.text,
                                bundle.instructions,
                                criteria,
                            )
                        else:
                            assert glm_client is not None
                            res = await glm_client.evaluate_single_intent(
                                http_client,
                                model_spec,
                                sample.text,
                                bundle.instructions,
                                criteria,
                            )
                    elif bundle.task_type == "multi_intent":
                        if decision_client is not None:
                            res = await decision_client.evaluate_multi_intent(
                                http_client,
                                model_spec,
                                sample.text,
                                criteria,
                            )
                        else:
                            assert glm_client is not None
                            res = await glm_client.evaluate_multi_intent(
                                http_client,
                                model_spec,
                                sample.text,
                                criteria,
                            )
                    elif bundle.task_type == "multitask":
                        if decision_client is not None:
                            res = await decision_client.evaluate_multitask(
                                http_client,
                                model_spec,
                                sample.text,
                                criteria,
                            )
                        else:
                            assert glm_client is not None
                            res = await glm_client.evaluate_multitask(
                                http_client,
                                model_spec,
                                sample.text,
                                criteria,
                            )
                    else:
                        raise ValueError(f"Unsupported task_type: {bundle.task_type}")

                    record = {
                        "run_id": job.run_id,
                        "dataset_id": job.dataset_id,
                        "task_type": bundle.task_type,
                        "criteria_mode": job.criteria_mode,
                        "model_slug": job.model_slug,
                        "sample_id": sample.sample_id,
                        "text": sample.text,
                        "ground_truth": sample.label,
                        "ground_truth_labels": sample.labels,
                        **res,
                    }

                    async with file_lock:
                        f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
                        f_out.flush()
                        completed[sample.sample_id] = record
                        done_count += 1
                        if done_count % 25 == 0 or done_count == total_count:
                            print(
                                f"  -> [{job.run_id}] Progress: {done_count}/{total_count} "
                                f"(last latency={record.get('latency_ms')}ms)"
                            )

            await asyncio.gather(*(_worker(s) for s in pending))

    return [completed[s.sample_id] for s in samples if s.sample_id in completed]


def build_stage_jobs(stage: str) -> list[ExperimentJob]:
    """Return the ordered list of ExperimentJobs for a given benchmark stage."""
    all_d3 = ["d3-turbo", "d3-flash", "d3-mini"]
    all_glm = ["glm-5.3-low", "glm-5.3-medium", "glm-5.3-high"]
    core_models = all_d3 + ["glm-5.3-low"]
    full_models = all_d3 + all_glm

    jobs: list[ExperimentJob] = []

    if stage in ("stage2", "banking", "all"):
        # 1. MINDS-14 (zh-CN) Full (562 items) across all 6 models (Rich Criteria) -> Variable 1 + GLM effort ablation
        for m in full_models:
            jobs.append(ExperimentJob("minds14_zh", "rich", m))
        # 2. Variable 2: Label-Only vs Rich Criteria on MINDS-14 (zh-CN)
        for m in core_models:
            jobs.append(ExperimentJob("minds14_zh", "label_only", m))
        # 3. Variable 4: Mobile Banking 4-Question Multi-Task Gateway
        for m in core_models:
            jobs.append(ExperimentJob("minds14_zh_multitask", "rich", m))
        # 4. Variable 5: Mobile Banking Multi-Intent Recognition (200 items: 1/2/3 intents)
        for m in core_models:
            jobs.append(ExperimentJob("minds14_zh_multi", "rich", m))

    if stage in ("stage3", "banking", "all"):
        # 5. BANKING77 Full (3,080 items) across all models + Variable 2
        for m in full_models:
            jobs.append(ExperimentJob("banking77", "rich", m))
        for m in core_models:
            jobs.append(ExperimentJob("banking77", "label_only", m))

    if stage in ("stage4", "general", "all"):
        # 6. Cross-Domain General Benchmarks: Amazon MASSIVE (zh-CN) (2,974) + CLINC150 (5,500)
        for m in core_models:
            jobs.append(ExperimentJob("massive_zh", "rich", m))
        for m in core_models:
            jobs.append(ExperimentJob("clinc150", "rich", m))

    return jobs
