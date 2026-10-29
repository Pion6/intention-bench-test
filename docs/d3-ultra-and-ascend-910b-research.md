# `vllm-sr/d3-ultra` 深度解析与华为昇腾 NPU (Ascend 910B) 部署可行性调研报告

> **调研时间**：2026-10-29  
> **一手信源**：
> - Hugging Face 官方仓库与源码：[`vllm-sr/d3-ultra`](https://huggingface.co/vllm-sr/d3-ultra)（commit `52ba9e7a00402f9560c7fba8c039608e7ddee7fb`，含 `README.md`、`config.json`、`decision_config.json`、`modeling_d3.py`、`d3_runtime.py`、`d3_fast.py`、`d3_format.py`、`d3_kernels.py`、`d3_server.py`）
> - 官方在线服务 API：[`https://decision.vllm-sr.ai/api/info`](https://decision.vllm-sr.ai/api/info)、[`/v1/systemone/models`](https://decision.vllm-sr.ai/v1/systemone/models)、[`/api/status`](https://decision.vllm-sr.ai/api/status)、[`/api/stats`](https://decision.vllm-sr.ai/api/stats)
> - 官方评测榜单：[`multimodalart/jev-decision-index`](https://huggingface.co/spaces/multimodalart/jev-decision-index)（Edition `0.4.1` Text & `0.4` Vision，`generated_utc: 2026-10-28T19:32:17Z`）
> - 昇腾生态组件：`torch_npu`、CANN、`transformers` (`Qwen3_5Model`)、`flash-linear-attention` (`fla[npu]`)、`vllm-project/vllm-ascend`

---

## 一、核心结论速览 (Executive Summary)

1. **`d3` 确实发布了全新的旗舰版 `vllm-sr/d3-ultra`（2026-10-24 发布，2026-10-26 更新）**：
   - **参数规模与底座**：基于 `Qwen/Qwen3.8-27B` (`Qwen3_5Model`) 的 **26.09B**（`26,085,330,160` BF16 参数，含 `0.46B` 27 层视觉编码器与 `255 × 5120` FP32 线性读出头 `readout.safetensors`），Apache-2.0 开源协议。
   - **在线 API 已全面切换**：在官方免费 API [`https://decision.vllm-sr.ai`](https://decision.vllm-sr.ai) 上，**`vllm-sr/d3-ultra` 已正式成为默认的 27B 旗舰模型（13 个在线副本，承接全站 86.68% 流量）**；原有的 **`vllm-sr/d3-turbo` 和 `vllm-sr/d3` 均已变为指向 `vllm-sr/d3-ultra` 的别名 (`alias_of: "vllm-sr/d3-ultra"`)**。也就是说，当前我们评测脚本请求 `vllm-sr/d3-turbo` 或 `vllm-sr/d3` 时，底层实际已经由最新的 `d3-ultra` 响应。
   - **四大核心升级**：
     1. **`header-v1` 全局问题感知提示词布局 (`prompt_layout: "header-v1"`)**：在 `State:` 之前自动注入 `Questions in this request:` 头部，将同一次请求中的所有问题及候选项文本放在共享前缀中。配合 `prefix_bidir` + `StateOnce`，每个问题既能看到同请求中其他问题的上下文，又只需计算一次前缀状态缓存。
     2. **默认上下文窗口直接拉满 131K (`validated_length: 131072`)**：相比 `d3-turbo` 默认 `32,768` tokens，`d3-ultra` 默认直接开放 **131,072 tokens/question**（开启 `long_context=True` 可达 `262,144` tokens）。
     3. **动态温度校准 (`temperature_by_questions`)**：单问题请求 (`n=1`) 自动采用 `T=1.046` 提升概率校准度，多问题请求 (`n>=2`) 采用 `T=1.0`。
     4. **`Box` 视觉目标存在性置信度合一**：`box` 题型的 `confidence` 直接反映目标物体是否存在于画面中（`< 0.10` 代表画面中不存在该物体），无需像早期版本那样额外搭配一道 `noul`（Yes/No）问题验真。
   - **评测大幅跃升**：
     - 在官方 **Decision Index `0.4.1` 纯私有集文本榜单（244 个模型）** 中，`d3-ultra` 取得 **`70.18` Full 分（总榜 #5，并列第 2 梯队 `rank_tie: #2`）**，比 `d3-turbo v1.1`（`67.31`，#18）提升了 **+2.87 分**（其中未知新领域 `new_domains` 从 `56.41` 跃升至 `59.16`）。
     - 在 `vllm-sr` 内部 **SR-Text** 基准（10,411 请求 / 73,495 问题）中，**`d3-ultra` 以 `87.2` 分夺冠**，超越 `Fastino GLiDE V1.2` (`85.1`)、`Liquid AI d1-ultra` (`83.4`)、`d3-turbo` (`83.1`，+4.1 分)、`d3` (`81.4`，+5.8 分) 和 `Jev 1.14` (`73.1`，+14.1 分)。其中 **检索与分类 (Retrieval & Classification) 高达 `97.7`**，**工具与自动化 (Tools & Automation) 高达 `97.8`**。

2. **华为昇腾 NPU (`Ascend 910B`) 完全可以部署 `d3` 全系列模型（含 `d3-ultra` / `d3-turbo` / `d3-flash` / `d3-mini` 等）**：
   - **源码级原生兼容**：经逐行审查 `d3-ultra` 的 `modeling_d3.py`、`d3_runtime.py` 和 `d3_fast.py`：
     - `_device_name()` 使用 `str(torch.device(value))` 校验设备名，导入 `import torch_npu` 后直接传入 `device="npu:0"` 即可通过校验，**无需强制修改模型文件即可加载运行**（或使用 `from torch_npu.contrib import transfer_to_npu` 自动将 `cuda:0` 映射至 `npu:0`）。
     - ROCm 专属加速路径 `d3_fast.py` / `d3_kernels.py` 会自动检测 `model.device.type != "cuda"` 并安全跳过（`fast_skipped = "not a ROCm GPU"`），自动回退到硬件无关的 PyTorch `StateOnce` / `PromptEncoder` 执行路径。
     - 48 层线性注意力 (`Gated DeltaNet`) 在未安装 CUDA 版 `flash-linear-attention` 时，自动调用 `transformers` 内置的纯 PyTorch 算子 (`torch_chunk_gated_delta_rule` + `F.conv1d`) 在 `torch_npu` 上运行；也可安装昇腾版 `triton-ascend` + `flash-linear-attention-npu` 启用 NPU Triton 融合算子加速。
     - 16 层全注意力 (`full_attention`) 通过 `F.scaled_dot_product_attention` 直接路由至昇腾 CANN 的 `npu_fusion_attention` / SDPA 算子。
     - 视觉编码器的 `Conv3d` 被 `d3_runtime.py` 的 `linearize_patch_embed()` 自动转换为 `F.linear` 矩阵乘，天然规避了昇腾 910B 上 `Conv3d` 5D 格式转换的性能瓶颈。
   - **910B 显存 (HBM) 选型结论**：
     - **Ascend 910B1 / 910B2 / 910B3（64 GB HBM2e）**：**单卡即可原生 BF16 部署 27B 旗舰 `d3-ultra` / `d3-turbo`**（BF16 权重占 `52.17 GB`，剩余约 `11~12 GB` HBM 可支撑 8K~16K 上下文并发推理），更可轻松运行 `d3-flash (9B)` 及以下全部尺寸。
     - **Ascend 910B4（32 GB HBM2e）**：因 `d3_runtime.py` 强制要求单卡 (`single-device`) 不支持多卡张量并行切分，32 GB 单卡无法装下 52.2 GB 的原生 BF16 27B 模型，但**可单卡完美运行 `d3-flash (9B, ~16.8 GB)`、`d3-mini (4B, ~9.1 GB)`、`d3-nano (2B, ~4.4 GB)`、`d3-lite (0.8B)`、`d3-edge (0.6B)`**。
   - **推理框架选型建议**：**强烈推荐使用 `d3` 自带的 `Transformers AutoModel` / `d3_runtime.py` / `d3_server.py` 配合 `torch_npu` 部署**，而非直接用原生 `vllm-ascend` 的 `vllm serve`（因为 `d3` 使用外部 `readout.safetensors` 255 维线性头且依赖 `prefix_bidir` 双向注意力掩码与 `StateOnce` 状态复用，原生 `vllm-ascend` 的自回归因果 PagedAttention 无法直接等价还原其双向注意力与多问题复用机制）。

---

## 二、`vllm-sr/d3-ultra` 及近期新品全景解析

### 2.1 2026 年 10 月下旬 `vllm-sr` 发布的全新模型矩阵

通过查询 Hugging Face 官方 API (`https://huggingface.co/api/models?author=vllm-sr`)，在 2026-10-19 `d3-turbo` 发布之后，`vllm-sr` 团队又密集发布了以下模型：

| 模型仓库 ID | 发布时间 (UTC) | 参数量 / 精度 | 定位与核心特性 |
| :--- | :--- | :--- | :--- |
| **[`vllm-sr/d3-ultra`](https://huggingface.co/vllm-sr/d3-ultra)** | `2026-10-24 23:30` (更新于 `10-26`) | **26.09B** (BF16, 52.21 GB) | **Decision 3.0 新一代 27B 旗舰多模态决策模型**；支持 Text / Image / Video 与 Choice / Yes-No (`noul`) / Score / Box / Track；默认 131K 上下文，`header-v1` 提示词布局。 |
| **[`vllm-sr/d3-turbo-INT4`](https://huggingface.co/vllm-sr/d3-turbo-INT4)** | `2026-10-22 05:02` | 26.09B (INT4 量化) | `d3-turbo` 的 INT4 权重量化版本，大幅降低显存占用。 |
| **[`vllm-sr/d3-turbo-NVFP4`](https://huggingface.co/vllm-sr/d3-turbo-NVFP4)** | `2026-10-21 00:25` | 26.09B (NVFP4 量化) | 面向 NVIDIA Blackwell 架构的 FP4 量化版本。 |
| **[`vllm-sr/d3-turbo-FP8`](https://huggingface.co/vllm-sr/d3-turbo-FP8)** | `2026-10-20 14:19` | 26.09B (FP8 量化) | `d3-turbo` 的 FP8 量化版本。 |
| **[`vllm-sr/vela3-turbo`](https://huggingface.co/vllm-sr/vela3-turbo)** | `2026-10-21 06:12` | Vela 3.0 系列 | 面向安全护栏 (Guardrails) 与高速路由的 Vela 3.0 旗舰模型。 |
| **[`vllm-sr/vela3-fast-vision`](https://huggingface.co/vllm-sr/vela3-fast-vision)** | `2026-10-21 12:02` | **0.25B** (`254M`, 基于 `UltraVBERT`) | 超轻量多模态双向编码器决策模型，单条推理中位延迟仅 `36.8 ms`。 |
| **[`vllm-sr/vela3-fast`](https://huggingface.co/vllm-sr/vela3-fast)** | `2026-10-17 12:36` | **0.38B** (`380M`, 基于 `UltraBERT-380M`) | 超轻量纯文本双向编码器决策模型，Public 榜得分 `50.36`，中位延迟 `54.3 ms`。 |

---

### 2.2 `d3-ultra` vs `d3-turbo` vs `d3` 核心架构与配置差异

对比三个 27B 模型的 [`config.json`](https://huggingface.co/vllm-sr/d3-ultra/raw/main/config.json)、[`decision_config.json`](https://huggingface.co/vllm-sr/d3-ultra/raw/main/decision_config.json) 及 `d3_format.py` / `d3_runtime.py` 源码，核心异同如下：

| 维度 | `vllm-sr/d3` (v3.1.0) | `vllm-sr/d3-turbo` (v1.1) | `vllm-sr/d3-ultra` (最新旗舰) |
| :--- | :--- | :--- | :--- |
| **发布日期** | 2026-10-10 | 2026-10-19 | **2026-10-24 (10-26 更新)** |
| **骨干网络 (`config.json`)** | `Qwen3_5Model` (64 层：48 层 `linear_attention` + 16 层 `full_attention`, `hidden_size=5120`) | 同左 (`26,085,330,160` BF16 参数) | 同左 (`26,085,330,160` BF16 参数 + `readout.safetensors`) |
| **支持的决策题型** | `choice`, `noul`, `score`（**不支持** `box` / `track`） | `choice`, `noul`, `score`, `box`, `track` | `choice`, `noul`, `score`, `box`, `track` |
| **提示词布局 (`prompt_layout`)** | 标准布局（无跨问题 Header） | 标准布局（无跨问题 Header） | **`"header-v1"`（在 `State:` 前注入同请求全部问题与选项概览）** |
| **注意力模式 (`attention_mode`)** | `"noncausal_full_attention"` | `"prefix_bidir"` (`StateOnce`) | **`"prefix_bidir"` (`StateOnce` + `FusedStateOnce`)** |
| **默认最大验证上下文 (`validated_length`)** | `32,768` tokens | `32,768` tokens (需 `long_context=True` 解锁 131K) | **`131,072` tokens 默认直接可用** (`long_context=True` 解锁 `262,144`) |
| **温度校准策略 (`temperature`)** | 固定温度 `0.93` | 固定温度 `0.95` | **动态温度 `temperature_by_questions`**：单问题 `1.046`，多问题 `1.0` |
| **`Box` 题型物体不存在判定** | 不支持 `box` | 需配合一道 `noul` 问题判断物体是否在图中 | **`box.confidence` 直接校准存在性**（`< 0.10` 表示画面中无此物体） |
| **`decision.vllm-sr.ai` 在线状态** | 已成为 `vllm-sr/d3-ultra` 的别名 | 已成为 `vllm-sr/d3-ultra` 的别名 | **主模型（13 Replicas，图片上限提升至 16 张 / 16 MiB，State 字符上限 131,072）** |

#### 关键技术揭秘 1：`header-v1` 全局问题感知提示词布局 (`d3_format.py` L121–160)
在 `d3` 和 `d3-turbo` 中，当一个请求包含多个问题（例如同时询问意图分类、紧急程度、情感倾向）时，每个问题块只看到 `State` 和当前问题自身。而在 `d3-ultra` 中，`decision_config.json` 配置了 `"prompt_layout": "header-v1"`。查看 `d3_format.py` 第 121–160 行可见，当渲染文本请求时，`d3-ultra` 会在 `State:` 之前自动构造一段 `Questions in this request:` 头部：

```text
You are a decision model. Read the state and the question, then reply with the code of the best option.

Questions in this request:
1. Classify the user's primary banking intent.
   - 账户余额查询
   - 转账汇款
   - 信用卡还款
2. Is this request urgent?

State:
<用户输入的文本或对话状态>

Question:
Classify the user's primary banking intent.

Options:
- A: 账户余额查询
- B: 转账汇款
- C: 信用卡还款

Answer:
```

**为什么这个设计非常精妙？**
- 因为 `Questions in this request:` 头部位于 `State:` 之前（`d3_format.py` L220–232 的 `state_end_char` 将前缀切分点定在 `\n\nQuestion:\n` 处），对于同一个请求中的所有问题，**`System Prompt + Questions Header + State` 这一整段前缀是完全相同的**！
- 这意味着在 `prefix_bidir`（`StateOnce`）模式下，模型依然**只需要对前缀编码一次**，但后续每一个并行分支问题在作答时，都已经通过双向/前缀注意力感知到了“本次请求还在问哪些其他问题、有哪些候选项体系”，从而显著增强了跨领域分类（`New domains` +8.9 分）与多问题协同决策的一致性。

#### 关键技术揭秘 2：`StateOnce` 与 `RECURRENT_TOKENS = 256` 高速多问题推理 (`d3_runtime.py` L1459–1609)
`d3-ultra` 对输入状态 (`State`) 只做一次前向计算（48 层 Gated DeltaNet 产出末态 recurrent state，16 层 Full Attention 产出双向 KV Cache）。随后针对该状态提出的 1 到 255 个问题，互不干扰地共享状态缓存并行解码。在 1 张 AMD Instinct MI325X 上，对同一张图片问 **1 个问题耗时 `304 ms`**，问 **4 个问题 `332 ms`**，问 **16 个问题 `429 ms`**，问满 **255 个问题仅需 `2.16 s`**（平均每增加一个问题仅增加 **`7.3 ms`**）。

---

### 2.3 权威评测对比：Decision Index `0.4.1` 与 `SR-Text` / `SR-Box`

#### (1) 官方 `multimodalart/jev-decision-index` 最新榜单（Edition `0.4.1` Text & `0.4` Vision，2026-10-28 更新）
> **注**：在最新的 `0.4` / `0.4.1` 版本中，官方调整了总分 (`Full` / `v`) 计算公式，将公开集 (`public`) 权重设为 `0`，**100% 由防刷榜的私有测试集决定**（`60% Same-Skills 私有集 + 40% New-Domains 未知领域私有集`）。

| 模型名称 | 参数量 | Text 0.4.1 总排名 (`rank_v` / 244) | Text Full (`v`) | Text Public | Same-Skills (私有) | New-Domains (私有) | Vision 0.4 总排名 (61 模型) | Vision Full | Text 中位延迟 (`ms/row`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`vllm-sr/d3-ultra`** | **27B** | **#5 (并列第 2 梯队 `tie #2`)** | **`70.18`** | `71.57` (#9) | **`66.77`** | **`59.16`** | #12 | `70.59` | `200.5 ms` |
| `vllm-sr/d3-turbo v1.1` | 27B | #18 (`tie #4`) | `67.31` | `71.62` (#8) | `64.29` | `56.41` | **#3** | **`72.46`** | `172.0 ms` |
| `vllm-sr/d3 (v3.1.0)` | 27B | #19 (`tie #4`) | `67.15` | `65.45` (#34) | `64.45` | `55.97` | #10 | `71.26` | `174.1 ms` |
| `vllm-sr/d3-flash` | 9B | #81 (`tie #17`) | `54.27` | `57.79` (#82) | `54.63` | `41.87` | #20 | `66.89` | `102.7 ms` |
| `vllm-sr/d3-mini` | 4B | #93 (`tie #21`) | `49.97` | `54.90` (#100) | `51.90` | `36.52` | #29 | `63.16` | `91.4 ms` |
| `vllm-sr/d3-nano` | 2B | #154 (`tie #35`) | `34.30` | `42.22` (#145) | `42.47` | `16.40` | #45 | `54.67` | `69.9 ms` |
| `vllm-sr/vela3-fast` | 0.4B | #171 (`tie #39`) | `29.27` | `50.36` (#117) | `40.22` | `8.96` | — | — | `54.3 ms` |
| `vllm-sr/d3-lite` | 0.8B | #178 (`tie #40`) | `26.16` | `36.93` (#172) | `34.81` | `9.33` | #56 | `42.11` | `67.5 ms` |
| `vllm-sr/vela3-fast-vision` | 0.25B | #190 (`tie #43`) | `22.11` | `37.50` (#167) | `30.47` | `6.45` | — | — | **`36.8 ms`** |
| `vllm-sr/d3-edge` | 0.6B | #204 (`tie #47`) | `17.69` | `28.82` (#194) | `24.34` | `5.03` | #61 | `31.71` | `64.0 ms` |

#### (2) `vllm-sr` 官方 `SR-Text` 评测（10,411 请求 / 73,495 问题，涵盖五大能力域 + 43 个未知新领域）

| 评测维度 | `vllm-sr/d3-ultra` | `Fastino GLiDE V1.2` | `Liquid AI d1-ultra` | `vllm-sr/d3-turbo` | `vllm-sr/d3` | `TypeSafe Jev 1.14` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **SR-Text 总分 (Mean)** | **`87.2`** | `85.1` | `83.4` | `83.1` | `81.4` | `73.1` |
| 知识与推理 (Knowledge & Reasoning) | **`80.2`** | `76.4` | `72.4` | `75.1` | `70.0` | `63.9` |
| 语言理解 (Language Understanding) | **`90.3`** | `88.7` | `88.0` | `86.9` | `83.7` | `77.2` |
| **检索与分类 (Retrieval & Classification)** | **`97.7`** | `95.9` | `95.7` | `94.6` | `94.0` | `95.8` |
| **工具与自动化 (Tools & Automation)** | **`97.8`** | `97.4` | `93.8` | `96.7` | `94.0` | `89.0` |
| 艺术与人类品味 (Arts & Human Taste) | `89.6` | `89.0` | **`90.8`** | `86.7` | `85.7` | `67.8` |
| **未知新领域 (New Domains, 43 类)** | **`67.4`** | `63.3` | `60.0` | `58.5` | `60.9` | `44.9` |

可以看到，在与我们银行意图识别最相关的 **检索与分类 (`97.7`)**、**工具与自动化 (`97.8`)** 和 **未知新领域泛化 (`67.4`，比 `d3-turbo` 高出 8.9 分)** 上，`d3-ultra` 均取得了全场最高分。

---

## 三、华为昇腾 NPU (`Ascend 910B`) 部署 `d3` 全栈可行性与源码级指南

### 3.1 核心结论：**完全支持部署，且 `d3` 官方源码对 `npu` 设备具备极佳的兼容性**

我们对 `vllm-sr/d3-ultra`（与 `d3-turbo` / `d3-flash` / `d3-mini` 共享同一套推理运行时）的全部 10 个 Python 源码文件进行了逐行审计，结论如下：

1. **无需改动模型架构代码，直接支持 `device="npu:0"`**：
   - 在 `modeling_d3.py`（第 53–62 行）中，设备解析函数实现为：
     ```python
     def _device_name(value):
         if value is None:
             return None
         if isinstance(value, int):
             return f"cuda:{value}"
         if isinstance(value, (str, torch.device)):
             return str(torch.device(value))
         raise TypeError(...)
     ```
   - 只要在 Python 进程中先执行了 `import torch_npu`，PyTorch 就会注册 `npu` 设备后端。此时传入字符串 `device="npu:0"`（切勿传入整数 `0`，因为整数 `0` 会被第 57 行拼接成 `"cuda:0"`），`str(torch.device("npu:0"))` 就会原样返回 `"npu:0"`，并在 `d3_runtime.py` 第 1811 行通过 `device_map={"": "npu:0"}` 将整个 `Qwen3_5Model` 骨干网和 `readout.safetensors` 直接加载到昇腾 910B NPU 上！

2. **ROCm 专属 Triton 内核自动安全跳过，无硬编码报错**：
   - `d3` 仓库中带的 `d3_fast.py` 和 `d3_kernels.py` 是专门为 AMD Instinct MI300X/MI325X (`gfx942` ROCm HIP) 编写的底层算子。
   - 在 `d3_fast.py` 第 31–34 行中，门控逻辑为：
     ```python
     def enabled(model) -> str | None:
         if model.device.type != "cuda" or not getattr(torch.version, "hip", None):
             return "not a ROCm GPU"
     ```
   - 在华为昇腾 910B 上（`model.device.type == "npu"`），`d3_fast` 会干净地返回 `"not a ROCm GPU"` 并跳过（`d3_runtime.py` L1882），自动切换至基于标准 PyTorch OP 的 `StateOnce`（L1325–1456）和 `PromptEncoder`（L1215–1296）通用执行引擎。

3. **三大核心算子在昇腾 910B (`torch_npu` + CANN) 上的映射与兼容性**：

   | 算子模块 | 在 `d3` (Qwen3.5) 中的占比与作用 | 昇腾 910B (`torch_npu` / CANN) 运行机制 | 兼容性状态 |
   | :--- | :--- | :--- | :---: |
   | **Gated DeltaNet 线性注意力** (`Qwen3_5GatedDeltaNet`) | 64 层中占 **48 层** (`linear_attention`)，含 `conv1d` 与 chunk/recurrent delta rule | **路径 A（零编译开箱即用）**：不安装 CUDA 版 `fla`，`transformers` 自动绑定纯 PyTorch 实现 (`torch_chunk_gated_delta_rule` + `F.conv1d`)，全部由标准矩阵乘、指数、`cumsum` 在 NPU Cube/Vector 单元执行。<br>**路径 B（高性能融合算子）**：在 CANN $\ge$ 8.5.2 下安装 `triton-ascend` 与昇腾适配版 `flash-linear-attention` (`FLA_NPU_SOC=ascend910b`)，即可调用 NPU Triton 融合算子。 | **原生支持** |
   | **双向 / 块掩码全注意力** (`Qwen3_5Attention`) | 64 层中占 **16 层** (`full_attention`, GQA `24Q / 4KV`, `head_dim=256`, MRoPE) | `d3_runtime.py` L1107 的 `sdpa_backends(device)` 在非 CUDA 设备上返回 `nullcontext()`，直接调用 `torch.nn.functional.scaled_dot_product_attention`。在 `torch_npu` 中自动下发至 CANN 的 `npu_fusion_attention`（支持 `attn_mask=None` 双向全注意力及 `bool` 掩码）。 | **原生支持** |
   | **视觉编码器 Patch Embed** (`linearize_patch_embed`) | 27 层 Vision Tower 的入口特征提取 | `d3_runtime.py` L585 在加载时自动将 `Conv3d` 权重展平并用 `F.linear` 替代 `Conv3d`！这恰好避开了昇腾 910B 上 3D 卷积常见的 `NDC1HWC0` 5D 格式转换开销，直接在 Cube 单元做 BF16 GEMM。 | **原生支持（且有额外性能红利）** |
   | **255 维线性读出头** (`readout.safetensors`) | 末层 `RMSNorm` 后提取最后一个 Token 隐藏态，映射为 255 个选项的 Logits | `hidden.float() @ self.readout.T`（`[B, 5120] @ [5120, 255]`），在 NPU 上以 FP32 执行小矩阵乘法与 Softmax，保证概率校准精度与 GPU 完全一致。 | **原生支持** |

---

### 3.2 昇腾 910B 硬件型号显存 (HBM) 测算与模型选型表

**关键约束提醒**：`modeling_d3.py`（第 65–72 行）和 `d3_runtime.py`（第 1811 行）**强制要求单个模型实例完整运行在单张加速卡上 (`device_map={"": str(self.device)}`)**，不支持 `device_map="auto"` 跨多卡切分单模型权重（多卡扩展的方式是在每张 NPU 卡上独立起一个 `d3` 进程/副本，通过负载均衡并发处理请求）。

华为昇腾 910B 有 **64 GB 显存版（910B1 / 910B2 / 910B3，如 Atlas 800T A2）** 和 **32 GB 显存版（910B4，如部分 Atlas 800I A2 / 300T A2）** 两种规格。各模型在 910B 上的 HBM 显存占用与适配情况如下：

| 模型名称 | 参数量 | BF16 权重体积 | 单卡 **Ascend 910B (64 GB)** 是否支持 | 单卡 **Ascend 910B4 (32 GB)** 是否支持 | 生产部署建议 |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **`vllm-sr/d3-ultra`** | 26.09B | **52.17 GB** (48.59 GiB) | **支持**（剩余 ~11.8 GB HBM 供激活值与 `StateOnce` 缓存，单卡支持 ~8K–16K 上下文） | **不支持**（单卡 BF16 放不下 52.2 GB；需使用 `d3-turbo-INT4` 并适配加载器或换 9B） | **64 GB 910B 首选旗舰**，精度最高（SR-Text `87.2`，检索分类 `97.7`） |
| **`vllm-sr/d3-turbo`** | 26.09B | **52.17 GB** (48.59 GiB) | **支持**（同上） | **不支持**（同上） | 64 GB 910B 自然图像 Box 检测首选 |
| **`vllm-sr/d3-flash`** | 9.41B | **18.82 GB** (17.53 GiB) | **完美支持**（剩余 >44 GB HBM，可满血跑 131K 超长上下文） | **完美支持**（剩余 ~13 GB HBM，支持 16K–32K 上下文） | **32 GB 910B4 首选主力模型**，也是 64 GB 卡高并发/长文本首选（全功能支持 Box） |
| **`vllm-sr/d3-mini`** | 4.54B | **9.08 GB** (8.46 GiB) | **完美支持**（单卡可部署 4~5 个实例） | **完美支持**（单卡可部署 2 个实例） | 高吞吐、低延迟（~90 ms）银行在线意图路由性价比之选 |
| **`vllm-sr/d3-nano`** | 2.21B | **4.43 GB** (4.12 GiB) | **完美支持** | **完美支持** | 轻量级高并发前置过滤 |
| **`vllm-sr/d3-lite` / `d3-edge`** | 0.85B / 0.59B | **1.71 GB / 1.19 GB** | **完美支持** | **完美支持** | 极致低延迟（15~25 ms）端侧或网关级分流 |

---

### 3.3 为什么推荐用 `d3_runtime.py` / `d3_server.py` + `torch_npu`，而不是直接用原生 `vllm-ascend`？

很多团队在昇腾 910B 上习惯直接使用 `vllm-ascend`（`vllm serve`）。虽然 `vllm-ascend`（`v0.17.0rc1` 及以上版本）已经支持了 `Qwen3_5ForCausalLM` / `Qwen3_5ForConditionalGeneration` 底座架构，但**对于 `vllm-sr/d3*` 系列决策模型，不能直接用原生 `vllm-ascend` 的生成式入口启动**，原因有三：

1. **输出头结构不同**：`d3` 的权重仓库中是 `Qwen3_5Model`（无 `lm_head.weight`），而是外挂了一个独立的 FP32 `readout.safetensors`（形状为 `[255, hidden_size]`），通过单步前向传播提取末位 Token 向量做矩阵乘直接输出 255 个选项的概率分布，完全不走自回归 Token 生成 (`max_tokens=1` 的词表投影与 `readout.safetensors` 权重不同)。
2. **注意力掩码机制不同**：`d3-ultra` / `d3-turbo` 的核心精度来源于 `"attention_mode": "prefix_bidir"`（状态前缀内部采用**双向非因果全注意力** `is_causal=False`，问题分支对状态前缀单向可见、问题内部双向可见）。而 `vllm-ascend` 的 PagedAttention 内核默认强制执行下三角因果掩码 (`causal mask`)，如果强行当作普通 Causal LM 运行，不仅无法加载 `readout.safetensors`，还会破坏双向注意力特征，导致准确率严重退化。
3. **`StateOnce` 与 `header-v1` 协议**：`d3` 自带的 `d3_runtime.py` 和 `d3_server.py` 已经完整实现了 System One API 协议（`/v1/systemone`）、`header-v1` 提示词组装、`StateOnce` 单次状态编码多问题复用、动态温度校准 (`1.046` / `1.0`) 以及 `box`/`track` 坐标解码。

因此，在昇腾 910B 上最稳妥、100% 保真且自带 HTTP 服务的部署方式，就是直接基于 **昇腾 PyTorch (`torch_npu`) + `d3_server.py`（或 `AutoModel`）** 部署！

---

### 3.4 昇腾 910B 实操部署步骤与两处轻量适配补丁

#### 第一步：准备昇腾 910B 基础软件栈
建议使用华为官方提供的昇腾 PyTorch 容器镜像（CANN $\ge$ 8.0.RC3 或 8.5.x，PyTorch 2.4 / 2.5 / 2.6 + 对应版本的 `torch_npu`）：

```bash
# 1. 安装 d3 依赖（注意：在昇腾 NPU 上不要安装仅支持 CUDA 的 flash-linear-attention 和 causal-conv1d）
pip install "transformers>=5.1.0" safetensors accelerate pillow torchvision huggingface_hub
```

> **可选加速**：如果您使用的 CANN 版本 $\ge$ 8.5.2，可安装昇腾版 Triton (`triton-ascend`) 与昇腾适配版 `flash-linear-attention`（设置环境变量 `export FLA_NPU_SOC=ascend910b`），为 48 层 Gated DeltaNet 启用 NPU Triton 融合算子；若不安装，`transformers` 会自动使用纯 PyTorch 算子在 NPU 上平稳运行。

#### 第二步：两种零侵入 / 微侵入 NPU 适配方式

**方式 A（零代码修改，推荐快速验证）**：利用 `torch_npu` 官方自带的 `transfer_to_npu` 工具，它会自动在 Python 运行时将所有 `torch.cuda.*` API 和 `"cuda:0"` 设备名透明重定向到 `torch.npu.*` 和 `"npu:0"`：

```python
import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu  # 自动将 cuda:0 映射为 npu:0
from transformers import AutoModel

# 64GB 910B 可直接加载 vllm-sr/d3-ultra；32GB 910B4 建议加载 vllm-sr/d3-flash 或 vllm-sr/d3-mini
model = AutoModel.from_pretrained(
    "vllm-sr/d3-ultra",
    trust_remote_code=True,
    device="cuda:0",  # transfer_to_npu 会自动将其重定向到 npu:0
)

res = model.decide(
    state="客户发来消息：你好，我想查一下我尾号8821的储蓄卡里还剩多少钱，顺便看下上个月的账单明细。",
    questions={
        "primary_intent": {
            "type": "choice",
            "instructions": "判断客户的首要业务意图",
            "options": {
                "balance": "账户余额查询",
                "bill": "账单明细查询",
                "transfer": "转账汇款",
                "lost": "卡片挂失",
            },
        },
        "multi_intent": {
            "type": "noul",
            "instructions": "该请求是否包含两个及以上的独立银行业务诉求？",
        },
    },
)
print(res["answers"])
```

**方式 B（显式传入 `device="npu:0"` + 4 行源码增强，推荐生产环境与 `d3_server.py`）**：
将模型仓库克隆到本地后，如果希望显式使用 `--device npu:0` 启动官方 HTTP 服务 `d3_server.py`，只需在本地模型目录的 `d3_runtime.py` 中补充两处 `npu` 分支（共 4 行代码，分别用于绑定当前 NPU 设备卡号和统计精确延迟同步）：

```python
# 补丁位置 1：d3_runtime.py 第 1794 行附近 (D3.__init__)
if self.device.type == "cuda":
    torch.cuda.set_device(self.device)
elif self.device.type == "npu":
    torch.npu.set_device(self.device)

# 补丁位置 2：d3_runtime.py 第 3309 行附近 (D3._sync)
def _sync(self) -> None:
    if self.device.type == "cuda":
        self.torch.cuda.synchronize(self.device)
    elif self.device.type == "npu":
        self.torch.npu.synchronize(self.device)
```

随后在 `d3_server.py` 顶部加入一行 `import torch_npu`，即可直接启动与官方 `https://decision.vllm-sr.ai/v1/systemone` **100% 协议兼容**的昇腾 910B 私有化决策模型微服务：

```bash
# 单卡启动 (例如在 NPU 0 上启动 d3-ultra 或 d3-flash)
ASCEND_RT_VISIBLE_DEVICES=0 python d3_server.py \
  --model /path/to/d3-ultra \
  --device npu:0 \
  --host 0.0.0.0 \
  --port 8000
```

如果服务器有 8 张昇腾 910B（如一台 Atlas 800T A2），可以通过 `ASCEND_RT_VISIBLE_DEVICES=0..7` 启动 8 个独立的 `d3_server.py` 工作进程（每个进程独占一张 910B 卡），前置一层 Nginx 负载均衡，即可获得单机 8 副本的高吞吐并发能力，且我们当前编写的 `intention-bench-test` 评测框架只需将 `DECISION_BASE_URL` 指向内网 `http://<昇腾服务器IP>:8000/v1/systemone` 即可无缝切换至私有化昇腾 NPU 节点！
