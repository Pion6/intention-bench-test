"""Async HTTP clients for Decision 3.0 (`decision.vllm-sr.ai`) and Zhipu GLM 5.3."""

from __future__ import annotations

import asyncio
import json
import random
import re
import time
from typing import Any

import httpx

from .catalogs import (
    build_d3_multi_intent_questions,
    build_d3_multitask_banking_questions,
    build_d3_single_intent_questions,
    build_glm_multi_intent_messages,
    build_glm_multitask_banking_messages,
    build_glm_single_intent_messages,
)
from .config import ModelSpec, RuntimeConfig


_JSON_BLOCK_RE = re.compile(r"\{[^{}]*\}", re.DOTALL)


def parse_glm_single_intent(
    raw_content: str,
    valid_keys: set[str],
) -> dict[str, Any]:
    """Parse GLM 5.3 single-intent response and classify format compliance & hallucination."""
    text = (raw_content or "").strip()
    strict_format_ok = False
    fallback_extracted = False
    predicted: str | None = None

    # 1. Try strict JSON parse
    try:
        obj = json.loads(text)
        if isinstance(obj, dict) and isinstance(obj.get("intent"), str):
            predicted = obj["intent"].strip()
            strict_format_ok = True
    except Exception:
        pass

    # 2. Try extracting JSON block inside markdown fences or surrounding prose
    if predicted is None:
        for match in _JSON_BLOCK_RE.finditer(text):
            try:
                obj = json.loads(match.group(0))
                if isinstance(obj, dict) and isinstance(obj.get("intent"), str):
                    predicted = obj["intent"].strip()
                    fallback_extracted = True
                    break
            except Exception:
                continue

    # 3. Try exact token match if model returned bare intent key
    if predicted is None:
        cleaned = text.strip("`'\" \n\t")
        if cleaned in valid_keys:
            predicted = cleaned
            fallback_extracted = True
        else:
            # Longest valid key mentioned in output
            hits = [k for k in sorted(valid_keys, key=len, reverse=True) if k in text]
            if hits:
                predicted = hits[0]
                fallback_extracted = True
            else:
                predicted = cleaned[:80] if cleaned else "__empty__"

    in_catalog = predicted in valid_keys
    return {
        "predicted_label": predicted,
        "in_catalog": in_catalog,
        "hallucinated_key": not in_catalog,
        "strict_format_ok": strict_format_ok,
        "fallback_extracted": fallback_extracted,
    }


def parse_glm_multi_intent(
    raw_content: str,
    valid_keys: set[str],
) -> dict[str, Any]:
    """Parse GLM 5.3 multi-intent JSON response `{"intents": [...]}`."""
    text = (raw_content or "").strip()
    strict_format_ok = False
    fallback_extracted = False
    raw_list: list[str] | None = None

    try:
        obj = json.loads(text)
        if isinstance(obj, dict) and isinstance(obj.get("intents"), list):
            raw_list = [str(x).strip() for x in obj["intents"] if str(x).strip()]
            strict_format_ok = True
    except Exception:
        pass

    if raw_list is None:
        for match in _JSON_BLOCK_RE.finditer(text):
            try:
                obj = json.loads(match.group(0))
                if isinstance(obj, dict) and isinstance(obj.get("intents"), list):
                    raw_list = [str(x).strip() for x in obj["intents"] if str(x).strip()]
                    fallback_extracted = True
                    break
            except Exception:
                continue

    if raw_list is None:
        raw_list = [k for k in valid_keys if k in text]
        fallback_extracted = True

    valid_predicted = sorted({k for k in raw_list if k in valid_keys})
    hallucinated = [k for k in raw_list if k not in valid_keys]
    return {
        "predicted_labels": valid_predicted,
        "raw_predicted_labels": raw_list,
        "hallucinated_key": len(hallucinated) > 0,
        "strict_format_ok": strict_format_ok,
        "fallback_extracted": fallback_extracted,
    }


def parse_glm_multitask(
    raw_content: str,
    valid_keys: set[str],
) -> dict[str, Any]:
    """Parse GLM 5.3 4-field multi-task JSON response."""
    text = (raw_content or "").strip()
    strict_format_ok = False
    obj: dict[str, Any] = {}

    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            obj = parsed
            strict_format_ok = True
    except Exception:
        for match in _JSON_BLOCK_RE.finditer(text):
            try:
                parsed = json.loads(match.group(0))
                if isinstance(parsed, dict):
                    obj = parsed
                    break
            except Exception:
                continue

    intent_pred = str(obj.get("intent", "")).strip()
    if intent_pred not in valid_keys:
        hits = [k for k in sorted(valid_keys, key=len, reverse=True) if k in text]
        if hits:
            intent_pred = hits[0]

    urgency = str(obj.get("urgency", "Normal")).strip()
    high_risk = bool(obj.get("high_risk", False))
    human_handoff = bool(obj.get("human_handoff", False))
    has_all_4 = all(k in obj for k in ("intent", "urgency", "high_risk", "human_handoff"))

    return {
        "predicted_label": intent_pred,
        "in_catalog": intent_pred in valid_keys,
        "hallucinated_key": intent_pred not in valid_keys,
        "strict_format_ok": strict_format_ok and has_all_4,
        "urgency": urgency,
        "high_risk": high_risk,
        "human_handoff": human_handoff,
    }


class DecisionClient:
    """Async client for `https://decision.vllm-sr.ai/v1/decisions`."""

    def __init__(self, runtime_cfg: RuntimeConfig, max_retries: int = 6) -> None:
        if not runtime_cfg.decision_api_key:
            raise ValueError("DECISION_API_KEY is missing. Set it in `.env`.")
        self.base_url = runtime_cfg.decision_base_url
        self.api_key = runtime_cfg.decision_api_key
        self.max_retries = max_retries

    async def _post_decisions(
        self,
        http_client: httpx.AsyncClient,
        payload: dict[str, Any],
    ) -> tuple[dict[str, Any], float]:
        url = f"{self.base_url}/v1/decisions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        for attempt in range(self.max_retries):
            t0 = time.perf_counter()
            try:
                resp = await http_client.post(url, headers=headers, json=payload)
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                if resp.status_code == 200:
                    return resp.json(), elapsed_ms
                if resp.status_code in (429, 502, 503, 504):
                    retry_after = float(resp.headers.get("Retry-After", "0") or 0)
                    wait_s = max(retry_after, (1.5**attempt) + random.uniform(0.1, 0.8))
                    await asyncio.sleep(wait_s)
                    continue
                resp.raise_for_status()
            except httpx.HTTPError:
                if attempt + 1 >= self.max_retries:
                    raise
                await asyncio.sleep((1.5**attempt) + random.uniform(0.2, 1.0))
        raise RuntimeError("Exceeded max_retries on Decision API")

    async def evaluate_single_intent(
        self,
        http_client: httpx.AsyncClient,
        model_spec: ModelSpec,
        text: str,
        instructions: str,
        criteria: dict[str, str],
    ) -> dict[str, Any]:
        payload = {
            "model": model_spec.model_id,
            "state": text,
            "questions": build_d3_single_intent_questions(instructions, criteria),
            "options": {"return_meta": True},
        }
        data, client_ms = await self._post_decisions(http_client, payload)
        ans = data.get("answers", {}).get("intent", {})
        probs: dict[str, float] = ans.get("probabilities", {})
        choice = ans.get("choice", "")
        top_sorted = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)[:5]
        meta = data.get("meta", {})
        usage = data.get("usage", {})
        return {
            "predicted_label": choice,
            "in_catalog": choice in criteria,
            "hallucinated_key": False,
            "strict_format_ok": True,
            "fallback_extracted": False,
            "d3_confidence": ans.get("confidence"),
            "d3_top1_prob": top_sorted[0][1] if top_sorted else None,
            "d3_top5_probs": dict(top_sorted),
            "latency_ms": round(client_ms, 2),
            "compute_ms": meta.get("compute_ms"),
            "queue_ms": meta.get("queue_ms"),
            "input_tokens": usage.get("input_tokens", 0),
            "output_tokens": usage.get("output_tokens", 0),
            "reasoning_tokens": 0,
            "cached_tokens": 0,
        }

    async def evaluate_multi_intent(
        self,
        http_client: httpx.AsyncClient,
        model_spec: ModelSpec,
        text: str,
        criteria: dict[str, str],
        threshold: float = 0.5,
    ) -> dict[str, Any]:
        payload = {
            "model": model_spec.model_id,
            "state": text,
            "questions": build_d3_multi_intent_questions(criteria),
            "options": {"return_meta": True},
        }
        data, client_ms = await self._post_decisions(http_client, payload)
        answers = data.get("answers", {})
        noul_scores: dict[str, float] = {
            k: float(v.get("noul", 0.0)) for k, v in answers.items() if isinstance(v, dict)
        }
        predicted_labels = sorted([k for k, p in noul_scores.items() if p >= threshold])
        if not predicted_labels and noul_scores:
            # Fallback to argmax if all are below 0.5
            best_k = max(noul_scores.items(), key=lambda kv: kv[1])[0]
            predicted_labels = [best_k]

        meta = data.get("meta", {})
        usage = data.get("usage", {})
        return {
            "predicted_labels": predicted_labels,
            "d3_noul_scores": noul_scores,
            "hallucinated_key": False,
            "strict_format_ok": True,
            "fallback_extracted": False,
            "latency_ms": round(client_ms, 2),
            "compute_ms": meta.get("compute_ms"),
            "queue_ms": meta.get("queue_ms"),
            "input_tokens": usage.get("input_tokens", 0),
            "output_tokens": usage.get("output_tokens", 0),
            "reasoning_tokens": 0,
            "cached_tokens": 0,
        }

    async def evaluate_multitask(
        self,
        http_client: httpx.AsyncClient,
        model_spec: ModelSpec,
        text: str,
        criteria: dict[str, str],
    ) -> dict[str, Any]:
        payload = {
            "model": model_spec.model_id,
            "state": text,
            "questions": build_d3_multitask_banking_questions(criteria),
            "options": {"return_meta": True},
        }
        data, client_ms = await self._post_decisions(http_client, payload)
        answers = data.get("answers", {})
        intent_ans = answers.get("intent", {})
        urgency_ans = answers.get("urgency", {})
        high_risk_ans = answers.get("high_risk", {})
        handoff_ans = answers.get("human_handoff", {})

        meta = data.get("meta", {})
        usage = data.get("usage", {})
        return {
            "predicted_label": intent_ans.get("choice", ""),
            "in_catalog": intent_ans.get("choice", "") in criteria,
            "hallucinated_key": False,
            "strict_format_ok": True,
            "urgency_score": urgency_ans.get("score"),
            "high_risk_prob": high_risk_ans.get("noul"),
            "human_handoff_prob": handoff_ans.get("noul"),
            "latency_ms": round(client_ms, 2),
            "compute_ms": meta.get("compute_ms"),
            "queue_ms": meta.get("queue_ms"),
            "input_tokens": usage.get("input_tokens", 0),
            "output_tokens": usage.get("output_tokens", 0),
            "reasoning_tokens": 0,
            "cached_tokens": 0,
        }


class GLMClient:
    """Async client for Zhipu `glm-5.3` (`/api/coding/paas/v4/chat/completions`)."""

    def __init__(self, runtime_cfg: RuntimeConfig, max_retries: int = 6) -> None:
        if not runtime_cfg.glm_api_key:
            raise ValueError("GLM_API_KEY is missing. Set it in `.env`.")
        self.base_url = runtime_cfg.glm_base_url
        self.api_key = runtime_cfg.glm_api_key
        self.max_retries = max_retries

    async def _post_chat(
        self,
        http_client: httpx.AsyncClient,
        payload: dict[str, Any],
    ) -> tuple[dict[str, Any], float]:
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        for attempt in range(self.max_retries):
            t0 = time.perf_counter()
            try:
                resp = await http_client.post(url, headers=headers, json=payload)
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                if resp.status_code == 200:
                    return resp.json(), elapsed_ms
                if resp.status_code in (429, 500, 502, 503, 504):
                    retry_after = float(resp.headers.get("Retry-After", "0") or 0)
                    wait_s = max(retry_after, (1.8**attempt) + random.uniform(0.5, 1.5))
                    await asyncio.sleep(wait_s)
                    continue
                resp.raise_for_status()
            except httpx.HTTPError:
                if attempt + 1 >= self.max_retries:
                    raise
                await asyncio.sleep((1.8**attempt) + random.uniform(0.5, 1.5))
        raise RuntimeError("Exceeded max_retries on GLM API")

    async def evaluate_single_intent(
        self,
        http_client: httpx.AsyncClient,
        model_spec: ModelSpec,
        text: str,
        instructions: str,
        criteria: dict[str, str],
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": model_spec.model_id,
            "temperature": 0,
            "messages": build_glm_single_intent_messages(instructions, criteria, text),
        }
        if model_spec.reasoning_effort:
            payload["reasoning_effort"] = model_spec.reasoning_effort

        data, client_ms = await self._post_chat(http_client, payload)
        msg = data.get("choices", [{}])[0].get("message", {})
        raw_content = msg.get("content", "")
        reasoning_content = msg.get("reasoning_content", "")
        parsed = parse_glm_single_intent(raw_content, set(criteria.keys()))

        usage = data.get("usage", {})
        prompt_details = usage.get("prompt_tokens_details", {}) or {}
        comp_details = usage.get("completion_tokens_details", {}) or {}

        return {
            **parsed,
            "raw_content": raw_content,
            "reasoning_content": reasoning_content,
            "latency_ms": round(client_ms, 2),
            "compute_ms": None,
            "queue_ms": None,
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
            "reasoning_tokens": comp_details.get("reasoning_tokens", 0),
            "cached_tokens": prompt_details.get("cached_tokens", 0),
        }

    async def evaluate_multi_intent(
        self,
        http_client: httpx.AsyncClient,
        model_spec: ModelSpec,
        text: str,
        criteria: dict[str, str],
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": model_spec.model_id,
            "temperature": 0,
            "messages": build_glm_multi_intent_messages(criteria, text),
        }
        if model_spec.reasoning_effort:
            payload["reasoning_effort"] = model_spec.reasoning_effort

        data, client_ms = await self._post_chat(http_client, payload)
        msg = data.get("choices", [{}])[0].get("message", {})
        raw_content = msg.get("content", "")
        parsed = parse_glm_multi_intent(raw_content, set(criteria.keys()))

        usage = data.get("usage", {})
        prompt_details = usage.get("prompt_tokens_details", {}) or {}
        comp_details = usage.get("completion_tokens_details", {}) or {}

        return {
            **parsed,
            "raw_content": raw_content,
            "latency_ms": round(client_ms, 2),
            "compute_ms": None,
            "queue_ms": None,
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
            "reasoning_tokens": comp_details.get("reasoning_tokens", 0),
            "cached_tokens": prompt_details.get("cached_tokens", 0),
        }

    async def evaluate_multitask(
        self,
        http_client: httpx.AsyncClient,
        model_spec: ModelSpec,
        text: str,
        criteria: dict[str, str],
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": model_spec.model_id,
            "temperature": 0,
            "messages": build_glm_multitask_banking_messages(criteria, text),
        }
        if model_spec.reasoning_effort:
            payload["reasoning_effort"] = model_spec.reasoning_effort

        data, client_ms = await self._post_chat(http_client, payload)
        msg = data.get("choices", [{}])[0].get("message", {})
        raw_content = msg.get("content", "")
        parsed = parse_glm_multitask(raw_content, set(criteria.keys()))

        usage = data.get("usage", {})
        prompt_details = usage.get("prompt_tokens_details", {}) or {}
        comp_details = usage.get("completion_tokens_details", {}) or {}

        return {
            **parsed,
            "raw_content": raw_content,
            "latency_ms": round(client_ms, 2),
            "compute_ms": None,
            "queue_ms": None,
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
            "reasoning_tokens": comp_details.get("reasoning_tokens", 0),
            "cached_tokens": prompt_details.get("cached_tokens", 0),
        }
