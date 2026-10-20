"""Dataset loader and local JSONL cache for the 4 official benchmark datasets + 2 banking suites."""

from __future__ import annotations

import io
import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

from .catalogs import (
    BANKING77_INTENT_KEYS,
    BANKING77_LABEL_ONLY,
    BANKING77_RICH_CRITERIA,
    MINDS14_INTENT_KEYS,
    MINDS14_LABEL_ONLY,
    MINDS14_RICH_CRITERIA,
)
from .config import DATA_CACHE_DIR


@dataclass
class EvalSample:
    """One test utterance in a single-intent, multi-intent, or multi-task benchmark."""

    sample_id: str
    text: str
    label: str  # Primary ground-truth intent key
    labels: list[str] = field(default_factory=list)  # Used for multi-intent evaluation
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class DatasetBundle:
    """Loaded dataset split along with its candidate intent catalog and instructions."""

    dataset_id: str
    display_name: str
    task_type: str  # "single_intent" | "multi_intent" | "multitask"
    instructions: str
    criteria_label_only: dict[str, str]
    criteria_rich: dict[str, str]
    samples: list[EvalSample]
    has_oos: bool = False
    oos_key: str = "oos"


def _fetch_hf_parquet_rows(
    repo_id: str,
    config_name: str,
    split_name: str,
    columns: list[str] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Fetch a Hugging Face dataset split via the official Parquet API and return rows + feature info."""
    import pyarrow.parquet as pq

    api_url = f"https://huggingface.co/api/datasets/{repo_id}/parquet/{config_name}/{split_name}"
    with httpx.Client(timeout=60.0, follow_redirects=True) as client:
        resp = client.get(api_url)
        resp.raise_for_status()
        parquet_urls: list[str] = resp.json()

        info_resp = client.get(
            f"https://datasets-server.huggingface.co/info?dataset={repo_id}&config={config_name}"
        )
        info_json = info_resp.json() if info_resp.status_code == 200 else {}

        rows: list[dict[str, Any]] = []
        for url in parquet_urls:
            p_resp = client.get(url)
            p_resp.raise_for_status()
            table = pq.read_table(io.BytesIO(p_resp.content), columns=columns)
            rows.extend(table.to_pylist())
        return rows, info_json


def _extract_class_names(info_json: dict[str, Any], field_name: str) -> list[str]:
    """Extract canonical ClassLabel names from Hugging Face dataset-info JSON."""
    features = info_json.get("dataset_info", {}).get("features", {})
    feat = features.get(field_name, {})
    if isinstance(feat, dict) and "names" in feat:
        return list(feat["names"])
    return []


def load_minds14_zh(cache_dir: Path = DATA_CACHE_DIR) -> DatasetBundle:
    """Load full MINDS-14 (zh-CN) mobile banking dataset (562 items, 14 intents)."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / "minds14_zh.json"
    if cache_file.is_file():
        cached = json.loads(cache_file.read_text(encoding="utf-8"))
        samples = [EvalSample(**s) for s in cached["samples"]]
    else:
        rows, _ = _fetch_hf_parquet_rows(
            repo_id="PolyAI/minds14",
            config_name="zh-CN",
            split_name="train",
            columns=["transcription", "english_translation", "intent_class"],
        )
        samples = []
        for idx, row in enumerate(rows):
            intent_idx = int(row["intent_class"])
            intent_key = MINDS14_INTENT_KEYS[intent_idx]
            samples.append(
                EvalSample(
                    sample_id=f"minds14_zh_{idx:04d}",
                    text=str(row["transcription"]).strip(),
                    label=intent_key,
                    labels=[intent_key],
                    metadata={
                        "english_translation": row.get("english_translation", "")
                    },
                )
            )
        cache_file.write_text(
            json.dumps(
                {"samples": [s.__dict__ for s in samples]},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    return DatasetBundle(
        dataset_id="minds14_zh",
        display_name="MINDS-14 (zh-CN 中文手机银行全量, 562条)",
        task_type="single_intent",
        instructions="请判断该手机银行客户提问所属的业务意图类别。",
        criteria_label_only=MINDS14_LABEL_ONLY,
        criteria_rich=MINDS14_RICH_CRITERIA,
        samples=samples,
    )


def load_banking77(cache_dir: Path = DATA_CACHE_DIR) -> DatasetBundle:
    """Load full BANKING77 official `test` split (3,080 items, 77 banking intents)."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / "banking77_test.json"
    if cache_file.is_file():
        cached = json.loads(cache_file.read_text(encoding="utf-8"))
        samples = [EvalSample(**s) for s in cached["samples"]]
    else:
        rows, _ = _fetch_hf_parquet_rows(
            repo_id="PolyAI/banking77",
            config_name="default",
            split_name="test",
            columns=["text", "label"],
        )
        samples = []
        for idx, row in enumerate(rows):
            intent_idx = int(row["label"])
            intent_key = BANKING77_INTENT_KEYS[intent_idx]
            samples.append(
                EvalSample(
                    sample_id=f"banking77_{idx:04d}",
                    text=str(row["text"]).strip(),
                    label=intent_key,
                    labels=[intent_key],
                )
            )
        cache_file.write_text(
            json.dumps(
                {"samples": [s.__dict__ for s in samples]},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    return DatasetBundle(
        dataset_id="banking77",
        display_name="BANKING77 (细粒度数字银行官方Test全量, 3,080条)",
        task_type="single_intent",
        instructions="Classify the customer's mobile banking inquiry into the single most appropriate banking intent.",
        criteria_label_only=BANKING77_LABEL_ONLY,
        criteria_rich=BANKING77_RICH_CRITERIA,
        samples=samples,
    )


def load_massive_zh(cache_dir: Path = DATA_CACHE_DIR) -> DatasetBundle:
    """Load full Amazon MASSIVE (zh-CN) official `test` split (2,974 items, 60 intents)."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / "massive_zh_test.json"
    if cache_file.is_file():
        cached = json.loads(cache_file.read_text(encoding="utf-8"))
        samples = [EvalSample(**s) for s in cached["samples"]]
        criteria = cached["criteria"]
    else:
        try:
            rows, _ = _fetch_hf_parquet_rows(
                repo_id="mteb/amazon_massive_intent",
                config_name="zh-CN",
                split_name="test",
                columns=["text", "label"],
            )
            intent_keys = sorted({str(r["label"]) for r in rows})
            criteria = {k: k.replace("_", " ") for k in intent_keys}
            samples = [
                EvalSample(
                    sample_id=f"massive_zh_{idx:04d}",
                    text=str(row["text"]).strip(),
                    label=str(row["label"]),
                    labels=[str(row["label"])],
                )
                for idx, row in enumerate(rows)
            ]
        except Exception:
            rows, info = _fetch_hf_parquet_rows(
                repo_id="AmazonScience/massive",
                config_name="zh-CN",
                split_name="test",
                columns=["utt", "intent"],
            )
            names = _extract_class_names(info, "intent")
            criteria = {name: name.replace("_", " ") for name in names}
            samples = [
                EvalSample(
                    sample_id=f"massive_zh_{idx:04d}",
                    text=str(row["utt"]).strip(),
                    label=names[int(row["intent"])],
                    labels=[names[int(row["intent"])]],
                )
                for idx, row in enumerate(rows)
            ]
        cache_file.write_text(
            json.dumps(
                {"criteria": criteria, "samples": [s.__dict__ for s in samples]},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    return DatasetBundle(
        dataset_id="massive_zh",
        display_name="Amazon MASSIVE (zh-CN 中文跨领域官方Test全量, 2,974条)",
        task_type="single_intent",
        instructions="请判断该中文用户指令所属的意图类别。",
        criteria_label_only=criteria,
        criteria_rich=criteria,
        samples=samples,
    )


def load_clinc150(cache_dir: Path = DATA_CACHE_DIR) -> DatasetBundle:
    """Load full CLINC150 (`clinc_oos` `plus`) official `test` split (5,500 items, 150 intents + `oos`)."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / "clinc150_test.json"
    if cache_file.is_file():
        cached = json.loads(cache_file.read_text(encoding="utf-8"))
        samples = [EvalSample(**s) for s in cached["samples"]]
        criteria_label = cached["criteria_label"]
        criteria_rich = cached["criteria_rich"]
    else:
        rows, info = _fetch_hf_parquet_rows(
            repo_id="clinc/clinc_oos",
            config_name="plus",
            split_name="test",
            columns=["text", "intent"],
        )
        names = _extract_class_names(info, "intent")
        criteria_label = {name: name.replace("_", " ") for name in names}
        criteria_rich = dict(criteria_label)
        criteria_rich["oos"] = (
            "Out of scope / none of the 150 supported banking, credit card, travel, utility, or assistant "
            "intents apply to this query (reject / route to fallback or human agent)"
        )
        samples = []
        for idx, row in enumerate(rows):
            intent_key = names[int(row["intent"])]
            samples.append(
                EvalSample(
                    sample_id=f"clinc150_{idx:04d}",
                    text=str(row["text"]).strip(),
                    label=intent_key,
                    labels=[intent_key],
                )
            )
        cache_file.write_text(
            json.dumps(
                {
                    "criteria_label": criteria_label,
                    "criteria_rich": criteria_rich,
                    "samples": [s.__dict__ for s in samples],
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    return DatasetBundle(
        dataset_id="clinc150",
        display_name="CLINC150 (150跨行业意图+1,000越界拒识官方Test全量, 5,500条)",
        task_type="single_intent",
        instructions=(
            "Classify the user's request into the matching intent from the 150 supported intents, "
            "or select 'oos' if the request falls outside all supported intents."
        ),
        criteria_label_only=criteria_label,
        criteria_rich=criteria_rich,
        samples=samples,
        has_oos=True,
        oos_key="oos",
    )


def load_minds14_zh_multi(cache_dir: Path = DATA_CACHE_DIR) -> DatasetBundle:
    """Build deterministic Mobile Banking Multi-Intent benchmark (`minds14_zh_multi`, 200 items: 1/2/3 intents)."""
    base = load_minds14_zh(cache_dir=cache_dir)
    cache_file = cache_dir / "minds14_zh_multi.json"
    if cache_file.is_file():
        cached = json.loads(cache_file.read_text(encoding="utf-8"))
        samples = [EvalSample(**s) for s in cached["samples"]]
    else:
        rng = random.Random(20261020)
        by_intent: dict[str, list[EvalSample]] = {}
        for s in base.samples:
            by_intent.setdefault(s.label, []).append(s)

        connectors_2 = ["另外，", "顺便再问一下，", "还有就是，", "同时我还想咨询："]
        connectors_3 = ["此外，", "最后还想麻烦您，"]
        samples: list[EvalSample] = []

        # 50 single-intent, 100 two-intent, 50 three-intent banking utterances = 200 items
        for i in range(50):
            s1 = rng.choice(base.samples)
            samples.append(
                EvalSample(
                    sample_id=f"minds14_multi_1_{i:03d}",
                    text=s1.text,
                    label=s1.label,
                    labels=[s1.label],
                    metadata={"num_intents": 1},
                )
            )

        intent_list = list(MINDS14_INTENT_KEYS)
        for i in range(100):
            k1, k2 = rng.sample(intent_list, 2)
            u1 = rng.choice(by_intent[k1]).text
            u2 = rng.choice(by_intent[k2]).text
            conn = connectors_2[i % len(connectors_2)]
            combined = f"{u1}。{conn}{u2}"
            labels = sorted([k1, k2])
            samples.append(
                EvalSample(
                    sample_id=f"minds14_multi_2_{i:03d}",
                    text=combined,
                    label=labels[0],
                    labels=labels,
                    metadata={"num_intents": 2},
                )
            )

        for i in range(50):
            k1, k2, k3 = rng.sample(intent_list, 3)
            u1 = rng.choice(by_intent[k1]).text
            u2 = rng.choice(by_intent[k2]).text
            u3 = rng.choice(by_intent[k3]).text
            c1 = connectors_2[i % len(connectors_2)]
            c2 = connectors_3[i % len(connectors_3)]
            combined = f"{u1}。{c1}{u2}。{c2}{u3}"
            labels = sorted([k1, k2, k3])
            samples.append(
                EvalSample(
                    sample_id=f"minds14_multi_3_{i:03d}",
                    text=combined,
                    label=labels[0],
                    labels=labels,
                    metadata={"num_intents": 3},
                )
            )

        cache_file.write_text(
            json.dumps(
                {"samples": [s.__dict__ for s in samples]},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    return DatasetBundle(
        dataset_id="minds14_zh_multi",
        display_name="MINDS-14 (zh-CN 手机银行复合多意图测试集, 200条: 1~3意图)",
        task_type="multi_intent",
        instructions="请识别该手机银行客户提问中包含的所有业务意图（可能包含 1 至 3 个意图）。",
        criteria_label_only=MINDS14_LABEL_ONLY,
        criteria_rich=MINDS14_RICH_CRITERIA,
        samples=samples,
    )


def load_minds14_zh_multitask(cache_dir: Path = DATA_CACHE_DIR) -> DatasetBundle:
    """Load 100-item mobile banking multi-task gateway suite (Variable 4: 1 Question vs. 4 Questions)."""
    base = load_minds14_zh(cache_dir=cache_dir)
    rng = random.Random(20261020)
    by_intent: dict[str, list[EvalSample]] = {}
    for s in base.samples:
        by_intent.setdefault(s.label, []).append(s)

    selected: list[EvalSample] = []
    for intent_key in MINDS14_INTENT_KEYS:
        pool = by_intent.get(intent_key, [])
        selected.extend(pool[:7])
    selected = selected[:100]
    rng.shuffle(selected)

    return DatasetBundle(
        dataset_id="minds14_zh_multitask",
        display_name="MINDS-14 (zh-CN 手机银行网关一状态四任务并发集, 98条)",
        task_type="multitask",
        instructions="手机银行智能网关单状态四任务并发决策（主意图 + 紧急度 + 高风险判定 + 转人工判定）",
        criteria_label_only=MINDS14_LABEL_ONLY,
        criteria_rich=MINDS14_RICH_CRITERIA,
        samples=selected,
    )


def load_dataset_by_id(dataset_id: str, cache_dir: Path = DATA_CACHE_DIR) -> DatasetBundle:
    """Dispatch dataset loader by `dataset_id`."""
    loaders = {
        "minds14_zh": load_minds14_zh,
        "banking77": load_banking77,
        "massive_zh": load_massive_zh,
        "clinc150": load_clinc150,
        "minds14_zh_multi": load_minds14_zh_multi,
        "minds14_zh_multitask": load_minds14_zh_multitask,
    }
    if dataset_id not in loaders:
        raise ValueError(
            f"Unknown dataset_id {dataset_id!r}. Valid options: {sorted(loaders)}"
        )
    return loaders[dataset_id](cache_dir=cache_dir)
