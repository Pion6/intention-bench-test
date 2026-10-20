# 手机银行与跨领域意图识别模型选型基准 (`intent-bench`)

本项目面向**AI手机银行意图识别技术选型**（及全行跨业务线推广评估），提供一套**100% 官方全量测试集（零抽样偏差）、信息完全对称、支持断点续跑**的自动化评测工程，正面对比 **System One 决策模型家族 (`vllm-sr/d3`)** 与通用推理大模型 **`GLM 5.3`**。

---

## 一、评测数据集矩阵（共 4 套权威全量集 + 2 套手机银行专项集）

| 数据集 ID | 数据集名称与来源 | 场景定位 | 候选意图数 | 官方测试集规模 | 任务类型 |
| :--- | :--- | :--- | ---: | ---: | :--- |
| **`minds14_zh`** | [`PolyAI/minds14`](https://huggingface.co/datasets/PolyAI/minds14) (`zh-CN`) | 中文手机银行真实客户语音转写提问（查余额、转账、锁卡、吞卡、限额等） | 14 类 | **562 条（全量）** | 单意图 (`choice`) |
| **`banking77`** | [`PolyAI/banking77`](https://huggingface.co/datasets/PolyAI/banking77) (`test`) | 全球最权威的数字银行细粒度意图识别基准（汇款处理中 vs 汇款失败 vs 汇率异常等） | 77 类 | **3,080 条（全量）** | 单意图 (`choice`) |
| **`massive_zh`** | [`AmazonScience/massive`](https://huggingface.co/datasets/AmazonScience/massive) (`zh-CN` `test`) | 亚马逊开源原生中文跨领域意图识别基准（覆盖 18 个日常与办公领域） | 60 类 | **2,974 条（全量）** | 单意图 (`choice`) |
| **`clinc150`** | [`clinc/clinc_oos`](https://huggingface.co/datasets/clinc/clinc_oos) (`plus` `test`) | 跨 10 大行业 150 类意图 + **1,000 条越界闲聊 (`oos`)**，检验跨业务推广与拒识转人工能力 | 151 类 (`150 + oos`) | **5,500 条（全量）** | 单意图 + 越界拒识 |
| **`minds14_zh_multi`** | 基于 `MINDS-14 (zh-CN)` 构造 | 手机银行复合多意图测试集（单意图 50 条 + 双意图 100 条 + 三意图 50 条） | 14 类 | **200 条** | 多意图（并发 `noul` vs 数组） |
| **`minds14_zh_multitask`** | 基于 `MINDS-14 (zh-CN)` 构造 | 手机银行网关“一状态四任务并发”（主意图 `choice` + 紧急度 `score` + 高风险 `noul` + 转人工 `noul`） | 14 类 + 3 项风控 | **98 条** | 单状态多任务并发 |

---

## 二、参评模型与五大控制变量

### 1. 参评模型阵容
- **System One 决策模型家族**（通过 `https://decision.vllm-sr.ai/v1/decisions` 评测）：
  - `d3-turbo`：[`vllm-sr/d3-turbo`](https://huggingface.co/vllm-sr/d3-turbo)（27B，`prefix_bidir` + `StateOnce` 状态只读一次）
  - `d3-flash`：[`vllm-sr/d3-flash`](https://huggingface.co/vllm-sr/d3-flash)（9B）
  - `d3-mini`：[`vllm-sr/d3-mini`](https://huggingface.co/vllm-sr/d3-mini)（4B）
- **通用生成式推理大模型**（通过智谱 Coding Plan 接口评测，启用 System Prompt 前缀缓存）：
  - `glm-5.3-low`：`glm-5.3`（`reasoning_effort="low"`）
  - `glm-5.3-medium`：`glm-5.3`（`reasoning_effort="medium"`）
  - `glm-5.3-high`：`glm-5.3`（`reasoning_effort="high"`）

### 2. 五大核心实验变量
1. **变量 1（模型尺寸与推理深度消融）**：对比 `27B d3-turbo` / `9B d3-flash` / `4B d3-mini` 与 `GLM 5.3 (low / medium / high)` 的精度-延迟性价比曲线。
2. **变量 2（业务规则描述丰富度）**：对比 `label_only`（仅提供意图标签名）与 `rich`（提供明确的银行业务边界说明）。
3. **变量 3（严格格式合规率与目录外幻觉率）**：统计生成式大模型随候选意图数量增加（$14 \to 60 \to 77 \to 151$）时的格式解析失败率与捏造意图率，对比决策模型 255 维读出头的 100% 闭集保证。
4. **变量 4（手机银行网关一状态多任务并发）**：对比单问意图 vs 同次请求并发判断 4 项网关任务时 `d3-turbo` (`StateOnce`) 与 `GLM 5.3` 的耗时与稳定性。
5. **变量 5（手机银行多意图识别）**：对比决策模型并发 `type: "noul"` 独立二分类（不抢概率，$p \ge 0.5$ 触发）与 `GLM 5.3` 输出 JSON 数组的完全匹配率（Exact Match）及 Micro/Macro-F1。

---

## 三、快速开始

### 1. 安装依赖与配置环境变量

```bash
pip install -r requirements.txt
cp .env.example .env
# 在 .env 中填入 DECISION_API_KEY 和 GLM_API_KEY（.env 已被 .gitignore 忽略，不会上传 GitHub）
```

### 2. 查看评测计划（Dry Run，不消耗 API 额度）

```bash
python3 scripts/run_benchmark.py --stage all --dry-run
```

### 3. 仅下载并缓存官方测试集到本地

```bash
python3 scripts/run_benchmark.py --download-only
```

### 4. 分阶段运行评测（支持随时中断与无缝断点续跑）

```bash
# 第一阶段：MINDS-14 (zh-CN) 全量 + 思考等级消融 + 业务描述丰富度 + 多意图 + 一状态四任务并发
python3 scripts/run_benchmark.py --stage stage2

# 第二阶段：BANKING77 官方 Test 全量 (3,080 条)
python3 scripts/run_benchmark.py --stage stage3

# 第三阶段：Amazon MASSIVE (zh-CN) 全量 (2,974 条) + CLINC150 全量 (5,500 条)
python3 scripts/run_benchmark.py --stage stage4

# 或者指定单个数据集与单个模型运行（可用 --limit 20 先做小样本冒烟验证）
python3 scripts/run_benchmark.py --dataset minds14_zh --model d3-turbo --limit 20
```

### 5. 生成评测报告与可视化看板

每次任务完成后会自动更新（也可手动执行 `--report-only`）：
- **Markdown 选型报告**：[`docs/mobile-banking-intent-benchmark-report.md`](docs/mobile-banking-intent-benchmark-report.md)
- **独立 HTML 可视化看板**：`reports/intent-benchmark-dashboard.html`
- **CSV 汇总表**：`results/summary/benchmark_summary.csv`

---

## 四、深度技术调研文档

- [`docs/jev-decision-models-research.md`](docs/jev-decision-models-research.md)：`datawhalechina/jev-cookbook` 与 System One 决策模型生态深度调研报告
- [`docs/vllm-sr-d3-research.md`](docs/vllm-sr-d3-research.md)：`vllm-sr/d3` & `vllm-sr/d3-turbo` 架构源码、`StateOnce` 原理与 `Jev Decision Index 0.3.2` 榜单深度调研报告
- [`docs/superpowers/specs/2026-10-20-intent-recognition-benchmark-design.md`](docs/superpowers/specs/2026-10-20-intent-recognition-benchmark-design.md)：本评测工程正式设计规范
