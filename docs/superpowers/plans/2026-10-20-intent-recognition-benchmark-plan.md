# Implementation Plan: Mobile Banking & Cross-Domain Intent Recognition Benchmark (`vllm-sr/d3` vs. `GLM 5.3`)

**Date:** 2026-10-20  
**Spec:** [`docs/superpowers/specs/2026-10-20-intent-recognition-benchmark-design.md`](../specs/2026-10-20-intent-recognition-benchmark-design.md)

---

## Step 1: Repository Initialization & Secret Protection
- Initialize git repository (`git init` on `main` branch).
- Create `.gitignore` excluding `.env`, `.venv/`, `__pycache__/`, `data/cache/`, `results/*.jsonl`, and OS artifacts.
- Create `.env.example` with documented placeholder variables (`DECISION_API_KEY`, `GLM_API_KEY`, `DECISION_BASE_URL`, `GLM_BASE_URL`).
- Store real user keys exclusively in untracked `.env`.

## Step 2: Intent Catalogs & Dataset Loader (`src/intent_bench/catalogs.py`, `src/intent_bench/datasets.py`)
- Implement deterministic Hugging Face Hub Parquet dataset fetchers (using `httpx` / `pyarrow` / `datasets` without heavy dependencies) for the 100% official test splits:
  1. `minds14_zh` (`PolyAI/minds14`, `zh-CN`, 562 items, 14 banking intents).
  2. `banking77` (`PolyAI/banking77`, `test` split, 3,080 items, 77 fine-grained banking intents).
  3. `massive_zh` (`AmazonScience/massive` / `mteb/amazon_massive_intent`, `zh-CN` `test` split, 2,974 items, 60 intents).
  4. `clinc150` (`clinc/clinc_oos`, `plus` `test` split, 5,500 items: 4,500 in-scope across 150 intents + 1,000 `oos`).
- Define both `label_only` and `rich` criteria catalogs (Chinese for `minds14_zh` and `massive_zh`, English + bilingual banking boundary definitions for `banking77` and `clinc150`) so `d3` and `GLM 5.3` receive byte-for-byte identical intent definitions.
- Build the deterministic `minds14_zh_multi` suite (1-intent, 2-intent, and 3-intent composite mobile banking utterances) for **Variable 5 (Multi-Intent Recognition)** and the `minds14_zh_multitask` suite for **Variable 4 (Single-State 4-Question Banking Gateway)**.

## Step 3: Model Clients (`src/intent_bench/clients.py`)
- **`DecisionClient` (`https://decision.vllm-sr.ai/v1/decisions`):**
  - Supports `vllm-sr/d3-turbo`, `vllm-sr/d3-flash`, and `vllm-sr/d3-mini`.
  - Supports single-intent `choice`, multi-intent concurrent `noul` (one `noul` question per candidate intent in a single request), and multi-task 4-question banking gateway (`choice` + `score` + 2× `noul`).
  - Passes `"options": {"return_meta": true}` to record server-side `compute_ms` and `queue_ms` alongside client HTTP latency.
  - Handles `429` (`Retry-After`), `502/503/504` with exponential backoff + jitter.
- **`GLMClient` (`https://open.bigmodel.cn/api/coding/paas/v4/chat/completions`):**
  - Supports `glm-5.3` with `reasoning_effort` in `["low", "medium", "high"]` and `temperature: 0`.
  - Places the full candidate intent catalog in the `system` message to maximize `cached_tokens` hit rate, and places only the single test utterance in the `user` message.
  - Parses single-intent `{"intent": "<key>"}`, multi-intent `{"intents": ["<key1>", ...]}`, and multi-task 4-field JSON, recording `strict_format_ok`, `fallback_extracted`, and `hallucinated_key`.

## Step 4: Resumable Runner, Metrics & Report Generator (`runner.py`, `metrics.py`, `report.py`, `cli.py`)
- Implement append-only `.jsonl` checkpointing in `results/raw/<experiment_id>__<model_slug>.jsonl` and automatic resume on restart.
- Compute primary metrics:
  - Single-intent: Top-1 Accuracy, Macro-F1, Per-Intent F1, Format Compliance Rate, Out-of-Catalog Hallucination Rate, OOS Recall/Precision/F1 (`clinc150`), Latency P50/P90/P99 (ms), Input/Output/Reasoning/Cached Tokens.
  - Multi-intent: Exact Match (Subset Accuracy), Micro-F1, Macro-F1, Precision, Recall.
  - Multi-task: 1-Question vs. 4-Question latency ratio and primary intent agreement.
- Generate Markdown report (`docs/mobile-banking-intent-benchmark-report.md`), CSV summary tables (`results/summary/*.csv`), and standalone HTML dashboard (`reports/intent-benchmark-dashboard.html`).

## Step 5: Offline Smoke Verification & Git Commit
- Run unit tests (`tests/test_smoke.py`) and `--dry-run` CLI checks to verify all modules without making live API calls.
- Stage and commit all project files cleanly to Git (`main`) so the user can push directly to GitHub.
