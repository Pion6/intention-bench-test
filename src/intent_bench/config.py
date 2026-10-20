"""Configuration, model definitions, and environment loading for intent-bench."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None


ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_CACHE_DIR = ROOT_DIR / "data" / "cache"
RESULTS_RAW_DIR = ROOT_DIR / "results" / "raw"
RESULTS_SUMMARY_DIR = ROOT_DIR / "results" / "summary"
REPORTS_DIR = ROOT_DIR / "reports"
DOCS_DIR = ROOT_DIR / "docs"


def load_env(env_path: Path | None = None) -> None:
    """Load environment variables from `api_keys.env` or `.env` if present."""
    candidates = [env_path] if env_path else [ROOT_DIR / "api_keys.env", ROOT_DIR / ".env"]
    for target in candidates:
        if target is None or not target.is_file():
            continue
        if load_dotenv is not None:
            load_dotenv(target, override=False)
        else:
            for raw_line in target.read_text(encoding="utf-8").splitlines():
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, val = line.split("=", 1)
                os.environ.setdefault(key.strip(), val.strip().strip("'\""))


@dataclass(frozen=True)
class ModelSpec:
    """Specification of a model variant under evaluation."""

    slug: str
    display_name: str
    provider: str  # "decision" | "glm"
    model_id: str
    params_label: str
    reasoning_effort: str | None = None
    default_concurrency: int = 8
    rpm_limit: int = 540


MODELS: dict[str, ModelSpec] = {
    "d3-turbo": ModelSpec(
        slug="d3-turbo",
        display_name="vLLM-SR d3-turbo (27B)",
        provider="decision",
        model_id="vllm-sr/d3-turbo",
        params_label="26.09B",
        default_concurrency=10,
        rpm_limit=540,
    ),
    "d3-flash": ModelSpec(
        slug="d3-flash",
        display_name="vLLM-SR d3-flash (9B)",
        provider="decision",
        model_id="vllm-sr/d3-flash",
        params_label="8.39B",
        default_concurrency=10,
        rpm_limit=720,
    ),
    "d3-mini": ModelSpec(
        slug="d3-mini",
        display_name="vLLM-SR d3-mini (4B)",
        provider="decision",
        model_id="vllm-sr/d3-mini",
        params_label="4.54B",
        default_concurrency=12,
        rpm_limit=1260,
    ),
    "glm-5.3-low": ModelSpec(
        slug="glm-5.3-low",
        display_name="GLM-5.3 (reasoning=low)",
        provider="glm",
        model_id="glm-5.3",
        params_label="Frontier LLM",
        reasoning_effort="low",
        default_concurrency=4,
        rpm_limit=120,
    ),
    "glm-5.3-medium": ModelSpec(
        slug="glm-5.3-medium",
        display_name="GLM-5.3 (reasoning=medium)",
        provider="glm",
        model_id="glm-5.3",
        params_label="Frontier LLM",
        reasoning_effort="medium",
        default_concurrency=3,
        rpm_limit=60,
    ),
    "glm-5.3-high": ModelSpec(
        slug="glm-5.3-high",
        display_name="GLM-5.3 (reasoning=high)",
        provider="glm",
        model_id="glm-5.3",
        params_label="Frontier LLM",
        reasoning_effort="high",
        default_concurrency=3,
        rpm_limit=60,
    ),
}


@dataclass(frozen=True)
class RuntimeConfig:
    """API credentials and endpoints loaded from environment."""

    decision_api_key: str
    decision_base_url: str
    glm_api_key: str
    glm_base_url: str

    @classmethod
    def from_env(cls) -> RuntimeConfig:
        load_env()
        return cls(
            decision_api_key=os.environ.get("DECISION_API_KEY", "").strip(),
            decision_base_url=os.environ.get(
                "DECISION_BASE_URL", "https://decision.vllm-sr.ai"
            ).rstrip("/"),
            glm_api_key=os.environ.get("GLM_API_KEY", "").strip(),
            glm_base_url=os.environ.get(
                "GLM_BASE_URL", "https://open.bigmodel.cn/api/coding/paas/v4"
            ).rstrip("/"),
        )
