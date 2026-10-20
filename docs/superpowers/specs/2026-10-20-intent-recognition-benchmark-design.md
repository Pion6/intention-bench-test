# Mobile Banking & Cross-Domain Intent Recognition Benchmark Design (`vllm-sr/d3` Family vs. `GLM 5.3`)

**Date:** 2026-10-20  
**Repository:** `/Users/linxiaoxu/Code/Work/DescionModel/intention-bench-test`  
**Status:** Approved by User (2026-10-20)

---

## 1. Intent and the Why

The team is currently in the **technology selection phase (技术选型阶段)** for **AI Mobile Banking intent recognition** (covering routine retail banking operations such as transfers, remittances, balance inquiries, card management, limits, and dispute handling), with plans to subsequently roll out the chosen intent recognition architecture across other lines of business in the bank.

Historically, the team's architecture relied on a **>500B general LLM paired with a fine-tuned small model**. With the emergence of **System One foundation decision models** (`vllm-sr/d3` family), the team wants to evaluate whether zero-shot / criteria-driven decision models can replace or outperform this pipeline. Because small-model fine-tuning resources are not being used in this phase, this benchmark directly compares the **Decision 3.0 model family (`vllm-sr/d3-turbo` 27B, `vllm-sr/d3-flash` 9B, `vllm-sr/d3-mini` 4B)** against a frontier general reasoning LLM (**`GLM 5.3`**) across **100% full official test splits** of four authoritative public datasets (two vertical banking datasets + two cross-domain general intent datasets) and four controlled experimental variables.

---

## 2. Goals, Non-Goals, and Anti-Goals

### 2.1 Goals
1. **100% Full-Split Objective Evaluation (Zero Sampling Bias):**
   Evaluate on the complete official test splits of four gold-standard datasets (**12,116 total test items**):
   - **Vertical Banking Datasets (Tier 1 Priority):**
     1. **`MINDS-14 (zh-CN)`** (`PolyAI/minds14`, `zh-CN` configuration, full split: **562 items**, **14 e-banking intents** in native Chinese spoken/transcribed queries).
     2. **`BANKING77`** (`PolyAI/banking77`, official `test` split: **3,080 items**, **77 fine-grained digital banking intents**, 40 items per intent).
   - **Cross-Domain & Out-of-Scope General Datasets (Tier 2 Priority):**
     3. **`Amazon MASSIVE (zh-CN)`** (`AmazonScience/massive`, `zh-CN` official `test` split: **2,974 items**, **60 intents** across 18 domains in native Chinese).
     4. **`CLINC150 (clinc_oos)`** (`clinc/clinc_oos`, `plus` official `test` split: **5,500 items** = **4,500 in-scope items** across 150 intents in 10 domains + **1,000 Out-of-Scope `oos` items** to measure rejection / human-handoff accuracy).
2. **Comprehensive Model & Reasoning-Effort Matrix:**
   - **Decision 3.0 Family (`https://decision.vllm-sr.ai`):**
     - `vllm-sr/d3-turbo` (27B, `prefix_bidir` + `StateOnce`)
     - `vllm-sr/d3-flash` (9B)
     - `vllm-sr/d3-mini` (4B)
   - **General Reasoning LLM (`https://open.bigmodel.cn/api/coding/paas/v4/chat/completions`):**
     - `glm-5.3` with `reasoning_effort: "low"`
     - `glm-5.3` with `reasoning_effort: "medium"`
     - `glm-5.3` with `reasoning_effort: "high"`
3. **Test All Five High-Value Experimental Variables:**
   - **Variable 1 — Model Size Scaling (`27B` vs. `9B` vs. `4B` vs. `GLM 5.3`):** Quantify the accuracy-latency-cost Pareto frontier to inform future on-premise GPU sizing.
   - **Variable 2 — Criteria Description Richness (`Label-Only` vs. `Rich Banking Criteria`):** Compare performance when candidate intents are specified only by short label names vs. enriched with explicit banking business boundary descriptions (tested on `MINDS-14 zh-CN` and `BANKING77`).
   - **Variable 3 — Format Compliance & Out-of-Catalog Hallucination Rate:** Track how often `GLM 5.3` (`low`/`medium`/`high`) outputs malformed JSON, refuses, or hallucinates an intent key outside the allowed catalog as option count scales ($14 \to 60 \to 77 \to 151$), compared against `d3`'s 255-way closed-set linear readout.
   - **Variable 4 — Mobile Banking Single-State Multi-Task Gateway (`1 Question` vs. `4 Concurrent Questions`):** On a curated banking evaluation set (derived from `MINDS-14 zh-CN` + realistic urgent/risky banking utterances), compare asking **1 question** (`intent` choice only) vs. asking **4 concurrent questions in one request**:
     1. `intent` (`choice`: 14 banking intents)
     2. `urgency` (`score`: `["Low", "Normal", "Urgent"]`)
     3. `high_risk` (`noul`: whether the query involves potential account compromise, fraud, or stuck funds)
     4. `human_handoff` (`noul`: whether immediate human agent escalation is required)
     Measure latency scaling (`StateOnce` on `d3-turbo` vs. `d3-flash` vs. `GLM 5.3` multi-field JSON) and primary intent consistency.
   - **Variable 5 — Mobile Banking Multi-Intent Recognition (`minds14_zh_multi`):** On a composite multi-intent mobile banking test suite (1-intent, 2-intent, and 3-intent utterances constructed from `MINDS-14 zh-CN`), compare `d3` (concurrent per-intent `type: "noul"` binary questions with $p \ge 0.5$ threshold) against `GLM 5.3` (`{"intents": ["<key1>", ...]}` JSON array output) using **Exact Match (Subset Accuracy)**, **Micro-F1**, and **Macro-F1**.
4. **Actionable, Audit-Ready Deliverables (Primary Metrics Focus):**
   - Focus the head-to-head comparison strictly on **primary objective metrics** (Top-1 Accuracy, Macro-F1, Multi-Intent Exact Match & Micro/Macro-F1, OOS Rejection F1, Latency P50/P90/P99, and Token Usage) without asking `GLM 5.3` for verbalized confidence scores.
   - Preserve all raw `d3` probability distributions and `confidence` values in the append-only per-item `JSONL` checkpoints so any threshold/calibration analysis can be run offline at any time.
   - Provide a comprehensive Chinese technical selection report (`docs/mobile-banking-intent-benchmark-report.md`) and a standalone HTML dashboard (`reports/intent-benchmark-dashboard.html`).

### 2.2 Non-Goals
- Training or fine-tuning local small models (out of scope for this selection phase).
- Using real, unanonymized internal bank customer logs on external public APIs.

### 2.3 Anti-Goals (Failure Conditions to Prevent)
- **Prompt Unfairness or Information Leakage:** Never batch multiple distinct user queries into a single `GLM 5.3` prompt (which would allow the LLM to use cross-sample label distribution heuristics or suffer cross-sample interference). Every test utterance must be evaluated independently, with `GLM 5.3` leveraging server-side prefix caching (`cached_tokens`) via a deterministic `system` prompt containing the candidate intent catalog.
- **Lost Progress on Rate Limits or Interruptions:** Never hold results only in memory. Every single completed item must be flushed to its `.jsonl` checkpoint file immediately, and re-running the command must skip already-completed `(dataset, split_variant, model, sample_id)` tuples.
- **Conflating API Transport Errors with Model Errors:** Network timeouts or `429` rate-limit responses must be retried with exponential backoff (`Retry-After` respected) and never recorded as wrong predictions.

---

## 3. Constraints & Environment

1. **APIs and Credentials:**
   - **Decision 3.0 API:**
     - Endpoint: `POST https://decision.vllm-sr.ai/v1/decisions` (or `/v1/systemone`)
     - Key: `DECISION_API_KEY` (provided by user; active, no expiration, daily quota resets at 00:00 UTC).
     - Rate limits per key:
       - `vllm-sr/d3-turbo`: 540 RPM / 450,000 TPM / 648,000 RPD, max concurrency 16.
       - `vllm-sr/d3-flash`: 720 RPM / 750,000 TPM / 432,000 RPD.
       - `vllm-sr/d3-mini`: 1,260 RPM / 750,000 TPM / 756,000 RPD.
   - **Zhipu GLM 5.3 API:**
     - Endpoint: `POST https://open.bigmodel.cn/api/coding/paas/v4/chat/completions` (Coding Plan endpoint verified working for `glm-5.3`).
     - Key: `GLM_API_KEY` (provided by user).
     - Model: `glm-5.3` with `"reasoning_effort": "low" | "medium" | "high"`.
   - **Security Constraint:** Store API keys in `.env` (gitignored) and load via environment variables; never hardcode raw keys in committed source scripts.
2. **Wall-Clock Runtime & Staged Execution Constraint:**
   - Because `GLM 5.3` takes ~1.8s/req on `low` and ~6.0s/req on `medium`/`high`, running all 12,116 items across all 3 GLM reasoning efforts plus Variable 2 variants sequentially would take 20+ hours of wall-clock time, whereas `d3-turbo`/`d3-flash`/`d3-mini` can finish thousands of items in minutes (especially using `states` batching of up to 8 states per request on `d3`).
   - Therefore, the benchmark runner must execute in **ordered stages (Phases)** and regenerate the live report and HTML dashboard after each stage completes, so the user gets full banking results quickly while longer cross-domain runs continue resumably.

---

## 4. Architecture & Evaluation Specification

### 4.1 User-Decided Choices
- **Target Scenario:** Mobile banking intent recognition (first priority) + general cross-domain intent recognition (for bank-wide rollout).
- **Models Under Test:** `vllm-sr/d3-turbo`, `vllm-sr/d3-flash`, `vllm-sr/d3-mini` vs. `glm-5.3` (`low`, `medium`, `high` reasoning effort).
- **Datasets:** Full test splits of `MINDS-14 (zh-CN)`, `BANKING77`, `Amazon MASSIVE (zh-CN)`, and `CLINC150`, plus all 4 experimental variables (model size scaling, criteria richness, hallucination/format compliance, and multi-task banking gateway).

### 4.2 Builder Decisions (Technical Defaults Marked as Builder's Call)
1. **Language & Runtime Stack (Builder's Call):**
   - Python 3 (`uv` / `venv` in `/Users/linxiaoxu/Code/Work/DescionModel/bench-test`) with `httpx` (`asyncio` for high-concurrency async requests with semaphore & `Retry-After` handling), `datasets` / `huggingface_hub` (or direct Parquet fetcher from Hugging Face Hub for fast, zero-overhead dataset download), and `numpy`/`pandas` for metric calculation.
2. **Fairness-Preserving Request Batching on `d3` vs. `GLM 5.3` (Builder's Call):**
   - On `decision.vllm-sr.ai` (`POST /v1/decisions`), the `states` array (`[{"id": "...", "state": "..."}, ...]`, up to 8 states per HTTP call) evaluates each state **completely independently** in the model forward pass (no cross-state attention exists in `d3_runtime.py`). Using 8-state micro-batches for single-question runs reduces HTTP overhead by 8× while keeping predictions mathematically identical to 1-state calls. (For latency benchmarking, we record both server-side `meta.compute_ms` via `"options": {"return_meta": true}` on 1-state calls and end-to-end client HTTP latency.)
   - On `GLM 5.3`, every test sample is sent as a strictly independent `chat/completions` call with a fixed `system` message (triggering Zhipu's `cached_tokens` prefix cache) and `temperature: 0`.
3. **Out-of-Scope (`oos`) Evaluation Protocol on `CLINC150` (Builder's Call):**
   - Include `"oos": "Out of scope / none of the supported services apply — route to general fallback or human agent"` in the 151 criteria so both `d3` and `GLM 5.3` compete head-to-head on 151-way classification (measuring In-Scope Accuracy, Macro-F1, and OOS Precision/Recall/F1).
4. **GLM 5.3 Prompt & Output Parsing Contract (Builder's Call):**
   - Single-intent system prompt instructs `GLM 5.3` to act as an intent classifier, provides the exact `key: description` catalog (identical to `d3`'s `criteria`), and asks for compact JSON `{"intent": "<key>"}` (no verbalized confidence requested).
   - Multi-intent system prompt asks for compact JSON `{"intents": ["<key1>", "<key2>"]}`.
   - The parser accepts exact JSON, fenced JSON blocks, or exact intent key matches, while logging whether strict JSON parsing succeeded (`strict_format_ok`) vs. required fallback extraction (`fallback_extracted`) vs. hallucinated an invalid key (`hallucinated_key`).
5. **Staged Execution Plan (Builder's Call):**
   - **Stage 1:** Download & cache all 4 official test splits (`data/datasets/`) + build the enriched banking criteria catalogs and the `minds14_zh_multi` composite multi-intent suite.
   - **Stage 2:** Run **Full `MINDS-14 (zh-CN)` (562 items)** across all models (`d3-turbo`, `d3-flash`, `d3-mini`, `GLM 5.3 low`, `GLM 5.3 medium`, `GLM 5.3 high`) + **Variable 2 (Label-Only vs. Rich Criteria)** + **Variable 4 (Multi-Task 4-Question Banking Suite)** + **Variable 5 (Multi-Intent Banking Suite)**.
   - **Stage 3:** Run **Full `BANKING77` (3,080 items)** across `d3-turbo`, `d3-flash`, `d3-mini`, and `GLM 5.3` (`low`, `medium`, `high` + Variable 2).
   - **Stage 4:** Run **Full `Amazon MASSIVE (zh-CN)` (2,974 items)** and **Full `CLINC150` (5,500 items)** across `d3-turbo`, `d3-flash`, `d3-mini`, and `GLM 5.3` (`low`).
   - After each stage, the report generator updates `docs/mobile-banking-intent-benchmark-report.md` and `reports/intent-benchmark-dashboard.html`.

---

## 5. What's Left to the Builder

- Exact directory layout under `bench_test/` (`datasets.py`, `clients.py`, `runner.py`, `metrics.py`, `report.py`).
- Exact wording of the Chinese/English rich criteria descriptions for `MINDS-14` and `BANKING77` (ensuring identical descriptions are passed to both `d3` and `GLM 5.3`).
- HTML/CSS chart rendering in `reports/intent-benchmark-dashboard.html`.
