# IR_LLM Project

[中文](#中文) | [English](#english)

## 中文

### 项目简介

本项目是《信息检索与大语言模型》课程设计的实现目录，研究主题为：

**基于大语言模型的代码库多跳检索增强系统研究：精确匹配与向量检索的实证对比**

当前仓库中已经完成并维护的内容，重点包括：

- 课程设计说明文档
- `DSPy` 代码库多跳问答 benchmark
- 基于 `jina-code-embeddings-1.5b` 的传统 Dense RAG baseline 第一版实现

当前 Dense RAG baseline 的目标是作为后续 `ripgrep` / 精确匹配检索系统的对照组，在相同问答框架下仅替换检索后端，从而保证实验对比公平。

### Quick Start

#### 1. 创建 Conda 环境

建议为本项目单独创建一个 `conda` 环境，而不是直接复用系统 Python 或其他实验环境。

推荐原因：

- 当前项目同时依赖 `DSPy`、`PyTorch`、`transformers`、`sentence-transformers`
- 后续还会涉及本地 Hugging Face 模型缓存
- 独立环境更利于复现，也能降低版本冲突风险

推荐命令：

```bash
conda create -n ir_llm python=3.11 -y
conda activate ir_llm
```

如果你后续明确需要和某个现有 CUDA / PyTorch 版本对齐，也可以在创建环境后再按你的机器配置安装对应版本的 `torch`。

#### 2. 安装依赖

在仓库根目录执行：

```bash
pip install -e .
pip install -r IR_LLM_Project/requirements_dense_rag.txt
```

说明：

- `pip install -e .` 用于安装当前 `dspy` 仓库自身依赖
- `requirements_dense_rag.txt` 用于补齐本地运行 Jina embedding 所需依赖

#### 3. 修正 PyTorch / CUDA 兼容性

如果你是 NVIDIA GPU 用户，建议在真正构建索引前先检查 `torch` 的 CUDA build 是否和驱动兼容。

你的当前报错属于典型版本不匹配问题：

- `nvidia-smi` 显示 Driver `570.211.01`
- `nvidia-smi` 显示最高支持 `CUDA 12.8`
- 但当前环境中的 `torch` 是 `2.11.0+cu130`

也就是：

- 驱动支持到 `12.8`
- PyTorch 二进制要求 `13.0`

因此 CUDA 初始化失败，报出：

- `The NVIDIA driver on your system is too old`

这不是说你的驱动真的过时，而是说它对 **当前安装的 `cu130` 版 PyTorch** 来说不够新。

先检查当前环境：

```bash
nvidia-smi
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("torch.version.cuda:", torch.version.cuda)
print("cuda.is_available:", torch.cuda.is_available())
print("device_count:", torch.cuda.device_count())
PY
```

先卸载当前不匹配的版本：

```bash
pip uninstall -y torch torchvision torchaudio
```

然后安装与 `CUDA 12.8` 兼容的官方 wheel。根据 PyTorch 官方文档，`2.10.0` 提供 `cu128` 版本，可直接安装：

```bash
pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 \
  --index-url https://download.pytorch.org/whl/cu128
```

安装完成后再次验证：

```bash
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("torch.version.cuda:", torch.version.cuda)
print("cuda.is_available:", torch.cuda.is_available())
print("device_count:", torch.cuda.device_count())
PY
```

如果验证成功，理论上应看到：

- `torch.version.cuda` 接近 `12.8`
- `torch.cuda.is_available()` 为 `True`

如果你只是想先把项目流程跑通，也可以先走 CPU：

```bash
PYTHONPATH=IR_LLM_Project/src python -m rag_baseline.build_index --device cpu
```

#### 4. 构建 Dense RAG 索引

```bash
PYTHONPATH=IR_LLM_Project/src python -m rag_baseline.build_index --device cuda:0
```

如果没有 GPU，可以改为：

```bash
PYTHONPATH=IR_LLM_Project/src python -m rag_baseline.build_index --device cpu
```

如果你的显卡是 8GB 显存，建议优先尝试低显存模式：

```bash
PYTHONPATH=IR_LLM_Project/src python -m rag_baseline.build_index \
  --device cuda:0 \
  --batch-size 1 \
  --use-fp16 \
  --max-seq-length 2048
```

如果仍然 OOM，可以进一步收缩为：

```bash
PYTHONPATH=IR_LLM_Project/src python -m rag_baseline.build_index \
  --device cuda:0 \
  --batch-size 1 \
  --use-fp16 \
  --max-seq-length 1024
```

如果还是不够稳定，直接切到 CPU：

```bash
PYTHONPATH=IR_LLM_Project/src python -m rag_baseline.build_index --device cpu
```

默认行为：

- 语料范围：`dspy/**/*.py`
- chunk 方式：结构化代码切块
- embedding 模型：`jinaai/jina-code-embeddings-1.5b`
- 输出目录：`IR_LLM_Project/artifacts/dense_rag_index`

索引输出包含：

- `chunks.jsonl`
- `embeddings.npy`
- `metadata.json`

#### 5. 运行单题检索

```bash
PYTHONPATH=IR_LLM_Project/src python -m rag_baseline.search_index \
  --query "Where is Module.__call__ implemented and how does it update caller_modules?" \
  --top-k 8
```

这个命令会输出：

- top-k 检索结果
- chunk id
- 文件路径
- 符号路径
- 相似度分数
- 截断后的代码预览

#### 6. 运行检索评测

```bash
PYTHONPATH=IR_LLM_Project/src python -m rag_baseline.eval_retrieval
```

当前默认使用：

- benchmark：仓库根目录下的 `DSPy_Multihop_QA.json`
- 索引目录：`IR_LLM_Project/artifacts/dense_rag_index`

当前支持的检索指标包括：

- `FileRecall@k`
- `MRR`
- `ChainCoverage`

#### 7. 运行批量问答实验

```bash
PYTHONPATH=IR_LLM_Project/src python -m rag_baseline.run_benchmark \
  --lm-model openai/gpt-4o-mini \
  --api-key YOUR_API_KEY
```

如果你使用兼容 OpenAI API 的本地或代理服务，可以再传：

```bash
--api-base YOUR_API_BASE
```

该脚本会：

- 加载 benchmark
- 调用 Dense RAG ReAct agent
- 逐题运行问答
- 记录每题 trajectory、耗时、调用到的 chunk 和文件
- 将结果保存到 `IR_LLM_Project/results/`

默认输出：

- `dense_rag_predictions.jsonl`
- `dense_rag_run_summary.json`

#### 8. 运行答案评测

```bash
PYTHONPATH=IR_LLM_Project/src python -m rag_baseline.eval_answers \
  --predictions-path IR_LLM_Project/results/dense_rag_predictions.jsonl
```

当前支持的本地答案指标包括：

- `Answer EM`
- `Answer F1`
- `Avg tool calls`
- `Avg latency`
- 基于实际 trajectory 的 `Chain Coverage`

### 当前项目已经能够实现的功能

#### 1. 结构化代码切块

已实现的代码切块逻辑位于 [chunking.py](/home/maxh/ssd1/github_clone/dspy/IR_LLM_Project/src/rag_baseline/chunking.py)。

当前支持以下 chunk 类型：

- `file_overview`
- `class`
- `function`
- `method`

每个 chunk 都保留：

- 文件路径
- 符号路径
- 起止行号
- 签名
- docstring
- 原始代码
- 用于 embedding 的完整文本

这比纯字符窗口切块更适合代码库场景，也更符合当前 benchmark 的多跳文件级推理需求。

#### 2. 基于 Jina Code Embedding 的非对称向量编码

已实现的 embedding 封装位于 [jina_embedder.py](/home/maxh/ssd1/github_clone/dspy/IR_LLM_Project/src/rag_baseline/jina_embedder.py)。

当前实现明确区分：

- `nl2code_query`
- `nl2code_document`

这很关键，因为 `jina-code-embeddings-1.5b` 是 query/document 非对称模型。如果 query 和 document 共用同一套 prompt，会把 dense baseline 做弱。

#### 3. 本地离线建索引

已实现的索引构建逻辑位于 [indexer.py](/home/maxh/ssd1/github_clone/dspy/IR_LLM_Project/src/rag_baseline/indexer.py) 和 [build_index.py](/home/maxh/ssd1/github_clone/dspy/IR_LLM_Project/src/rag_baseline/build_index.py)。

当前能力：

- 扫描 `dspy/**/*.py`
- 构造结构化 chunks
- 用 Jina embedding 编码所有文档 chunk
- 将索引保存到本地目录

当前默认不使用 FAISS，而是采用 numpy brute-force cosine retrieval。对当前 `DSPy` 仓库规模来说，这样更简单、透明、可复现。

#### 4. 本地 Dense Retrieval

已实现的检索器位于 [retriever.py](/home/maxh/ssd1/github_clone/dspy/IR_LLM_Project/src/rag_baseline/retriever.py)。

当前能力：

- 加载本地 `chunks.jsonl + embeddings.npy + metadata.json`
- 将自然语言 query 编码成 query embedding
- 执行 cosine similarity 排序
- 返回 top-k 检索结果
- 支持按 chunk id 打开完整 chunk

这套接口已经能支撑：

- 单题检索分析
- benchmark 检索评测
- 多跳 agent 工具调用

#### 5. DSPy ReAct Agent 接入

已实现的 agent 位于 [agent.py](/home/maxh/ssd1/github_clone/dspy/IR_LLM_Project/src/rag_baseline/agent.py)。

当前提供两个工具：

- `vector_search(query, top_k=8)`
- `open_chunk(chunk_id)`

它们被注入 `dspy.ReAct("question -> answer")` 中，用于多步推理。也就是说，当前 baseline 已经不只是“离线检索器”，而是一个可以接入真实 LLM 的多跳问答系统。

#### 6. 检索评测

已实现的检索评测脚本位于 [eval_retrieval.py](/home/maxh/ssd1/github_clone/dspy/IR_LLM_Project/src/rag_baseline/eval_retrieval.py)。

当前评测逻辑会：

- 读取 `DSPy_Multihop_QA.json`
- 从 `retrieval_path` 中抽取 gold 文件路径
- 比较 top-k 检索结果与 gold 文件的覆盖情况

这一步可以帮助你区分：

- 检索失败
- 检索成功但生成失败

#### 7. 批量问答运行与答案评测

已实现的批量问答和答案评测脚本分别位于：

- [run_benchmark.py](/home/maxh/ssd1/github_clone/dspy/IR_LLM_Project/src/rag_baseline/run_benchmark.py)
- [eval_answers.py](/home/maxh/ssd1/github_clone/dspy/IR_LLM_Project/src/rag_baseline/eval_answers.py)

当前能力：

- 对 benchmark 批量逐题跑 ReAct agent
- 记录每题 trajectory
- 记录耗时
- 记录工具调用次数
- 记录实际使用过的 chunk 与文件
- 记录 token usage
- 用本地 `EM/F1` 打分

这些结果已经足够支撑课程设计中的第一轮实验分析。

### 当前目录结构

```text
IR_LLM_Project/
├── README.md
├── requirements_dense_rag.txt
├── docs/
│   ├── README.md
│   ├── dense_rag_baseline_plan.md
│   └── 信息检索与大语言模型-课程项目.md
└── src/
    └── rag_baseline/
        ├── agent.py
        ├── benchmark_utils.py
        ├── build_index.py
        ├── chunking.py
        ├── config.py
        ├── eval_answers.py
        ├── eval_retrieval.py
        ├── indexer.py
        ├── io_utils.py
        ├── jina_embedder.py
        ├── metrics.py
        ├── retriever.py
        ├── run_benchmark.py
        ├── search_index.py
        └── types.py
```

### 当前已实现 vs 尚未实现

#### 已实现

- Dense RAG baseline 的代码框架
- 结构化 chunking
- 本地 Jina embedding 检索
- benchmark 检索评测
- DSPy ReAct 接入
- 批量问答运行脚本
- 本地答案评测脚本

#### 尚未实现

- `ripgrep` / 精确匹配 baseline
- Dense vs Grep 的统一实验对比脚本
- 图表与论文结果汇总
- 更强的 dense 变体
  - `Dense-Flat`
  - `Dense-Struct-DualView`
- 基于 LLM judge 的语义答案评测

### 相关文档

- 项目说明：[信息检索与大语言模型-课程项目.md](/home/maxh/ssd1/github_clone/dspy/IR_LLM_Project/docs/信息检索与大语言模型-课程项目.md)
- Dense baseline 方案：[dense_rag_baseline_plan.md](/home/maxh/ssd1/github_clone/dspy/IR_LLM_Project/docs/dense_rag_baseline_plan.md)

---

## English

### Overview

This directory contains the implementation work for the course project of **Information Retrieval and Large Language Models**.

Project topic:

**A Study of Multi-hop Retrieval-Augmented Codebase QA with LLMs: An Empirical Comparison between Exact Matching and Vector Retrieval**

The current state of the project already includes:

- the course-project documentation,
- a `DSPy`-based multi-hop QA benchmark,
- and a first working implementation of a traditional Dense RAG baseline based on `jina-code-embeddings-1.5b`.

The current Dense RAG system is intended to serve as the vector-retrieval control baseline before the exact-match / `ripgrep`-based system is added.

### Quick Start

#### 1. Create a Conda environment

It is recommended to use a dedicated `conda` environment for this project instead of reusing a system Python or an existing experiment environment.

Why:

- the project depends on `DSPy`, `PyTorch`, `transformers`, and `sentence-transformers`,
- it will also rely on local Hugging Face model caching,
- and a dedicated environment makes the setup much easier to reproduce and debug.

Recommended commands:

```bash
conda create -n ir_llm python=3.11 -y
conda activate ir_llm
```

If you need to match a specific CUDA / PyTorch stack on your machine, install the matching `torch` build after creating the environment.

#### 2. Install dependencies

Run from the repository root:

```bash
pip install -e .
pip install -r IR_LLM_Project/requirements_dense_rag.txt
```

Notes:

- `pip install -e .` installs the dependencies of the local `dspy` repository
- `requirements_dense_rag.txt` adds the extra dependencies required for local Jina embedding inference

#### 3. Fix the PyTorch / CUDA build if needed

If you are using an NVIDIA GPU, verify that the installed `PyTorch` CUDA build matches the CUDA level supported by your NVIDIA driver before building the index.

Your current failure is a standard mismatch case:

- `nvidia-smi` reports Driver `570.211.01`
- `nvidia-smi` reports support up to `CUDA 12.8`
- but the current environment has `torch 2.11.0+cu130`

That means:

- the driver supports `12.8`
- the PyTorch binary expects `13.0`

So CUDA initialization fails with:

- `The NVIDIA driver on your system is too old`

This does not mean your driver is actually outdated in general. It only means that the driver is too old for the currently installed `cu130` PyTorch build.

Check the current environment first:

```bash
nvidia-smi
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("torch.version.cuda:", torch.version.cuda)
print("cuda.is_available:", torch.cuda.is_available())
print("device_count:", torch.cuda.device_count())
PY
```

Remove the incompatible build:

```bash
pip uninstall -y torch torchvision torchaudio
```

Then install the official `cu128` wheel. According to the official PyTorch previous-versions page, `2.10.0` provides `cu128` wheels:

```bash
pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 \
  --index-url https://download.pytorch.org/whl/cu128
```

Verify again afterwards:

```bash
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("torch.version.cuda:", torch.version.cuda)
print("cuda.is_available:", torch.cuda.is_available())
print("device_count:", torch.cuda.device_count())
PY
```

If the installation is correct, you should see:

- `torch.version.cuda` close to `12.8`
- `torch.cuda.is_available()` is `True`

If you only want to get the pipeline running first, you can also use CPU:

```bash
PYTHONPATH=IR_LLM_Project/src python -m rag_baseline.build_index --device cpu
```

#### 4. Build the Dense RAG index

```bash
PYTHONPATH=IR_LLM_Project/src python -m rag_baseline.build_index --device cuda:0
```

If you only have CPU:

```bash
PYTHONPATH=IR_LLM_Project/src python -m rag_baseline.build_index --device cpu
```

If your GPU only has 8GB VRAM, start with the low-memory mode:

```bash
PYTHONPATH=IR_LLM_Project/src python -m rag_baseline.build_index \
  --device cuda:0 \
  --batch-size 1 \
  --use-fp16 \
  --max-seq-length 2048
```

If that still runs out of memory, reduce further:

```bash
PYTHONPATH=IR_LLM_Project/src python -m rag_baseline.build_index \
  --device cuda:0 \
  --batch-size 1 \
  --use-fp16 \
  --max-seq-length 1024
```

If GPU execution is still unstable, switch to CPU:

```bash
PYTHONPATH=IR_LLM_Project/src python -m rag_baseline.build_index --device cpu
```

Default behavior:

- corpus: `dspy/**/*.py`
- chunking: structure-aware code chunking
- embedding model: `jinaai/jina-code-embeddings-1.5b`
- output directory: `IR_LLM_Project/artifacts/dense_rag_index`

The output index includes:

- `chunks.jsonl`
- `embeddings.npy`
- `metadata.json`

#### 5. Run a single retrieval query

```bash
PYTHONPATH=IR_LLM_Project/src python -m rag_baseline.search_index \
  --query "Where is Module.__call__ implemented and how does it update caller_modules?" \
  --top-k 8
```

This prints:

- top-k retrieval hits
- chunk id
- file path
- symbol path
- similarity score
- truncated code preview

#### 6. Run retrieval evaluation

```bash
PYTHONPATH=IR_LLM_Project/src python -m rag_baseline.eval_retrieval
```

By default, this uses:

- benchmark: `DSPy_Multihop_QA.json` at the repository root
- index directory: `IR_LLM_Project/artifacts/dense_rag_index`

Current retrieval metrics include:

- `FileRecall@k`
- `MRR`
- `ChainCoverage`

#### 7. Run batch QA with DSPy ReAct

```bash
PYTHONPATH=IR_LLM_Project/src python -m rag_baseline.run_benchmark \
  --lm-model openai/gpt-4o-mini \
  --api-key YOUR_API_KEY
```

If you are using an OpenAI-compatible local endpoint or proxy, you can also pass:

```bash
--api-base YOUR_API_BASE
```

This script will:

- load the benchmark,
- run the Dense RAG ReAct agent question by question,
- record trajectory, latency, and accessed chunks/files,
- and save outputs into `IR_LLM_Project/results/`.

Default outputs:

- `dense_rag_predictions.jsonl`
- `dense_rag_run_summary.json`

#### 8. Run answer evaluation

```bash
PYTHONPATH=IR_LLM_Project/src python -m rag_baseline.eval_answers \
  --predictions-path IR_LLM_Project/results/dense_rag_predictions.jsonl
```

Current local answer metrics include:

- `Answer EM`
- `Answer F1`
- `Avg tool calls`
- `Avg latency`
- trajectory-based `Chain Coverage`

### What the project can already do

#### 1. Structure-aware code chunking

Implemented in [chunking.py](/home/maxh/ssd1/github_clone/dspy/IR_LLM_Project/src/rag_baseline/chunking.py).

Supported chunk types:

- `file_overview`
- `class`
- `function`
- `method`

Each chunk keeps:

- file path
- symbol path
- line range
- signature
- docstring
- raw code
- the final text used for embedding

This is much better aligned with repository-level code retrieval than naive fixed-size text splitting.

#### 2. Asymmetric Jina code embeddings

Implemented in [jina_embedder.py](/home/maxh/ssd1/github_clone/dspy/IR_LLM_Project/src/rag_baseline/jina_embedder.py).

The implementation explicitly separates:

- `nl2code_query`
- `nl2code_document`

This matters because `jina-code-embeddings-1.5b` is an asymmetric query/document model. Treating both sides identically would weaken the baseline.

#### 3. Offline local indexing

Implemented in [indexer.py](/home/maxh/ssd1/github_clone/dspy/IR_LLM_Project/src/rag_baseline/indexer.py) and [build_index.py](/home/maxh/ssd1/github_clone/dspy/IR_LLM_Project/src/rag_baseline/build_index.py).

Current capabilities:

- scan `dspy/**/*.py`
- build structured chunks
- embed all document chunks with Jina
- save the full local index

The current version intentionally uses numpy brute-force cosine retrieval rather than FAISS. For the current DSPy codebase size, this is simpler and easier to reproduce.

#### 4. Local dense retrieval

Implemented in [retriever.py](/home/maxh/ssd1/github_clone/dspy/IR_LLM_Project/src/rag_baseline/retriever.py).

Current capabilities:

- load `chunks.jsonl + embeddings.npy + metadata.json`
- encode natural-language queries
- rank chunks by cosine similarity
- return top-k hits
- open a full chunk by `chunk_id`

This is already enough to support:

- single-query retrieval analysis
- retrieval evaluation on the benchmark
- tool use inside a multi-hop agent

#### 5. DSPy ReAct integration

Implemented in [agent.py](/home/maxh/ssd1/github_clone/dspy/IR_LLM_Project/src/rag_baseline/agent.py).

The current agent exposes two tools:

- `vector_search(query, top_k=8)`
- `open_chunk(chunk_id)`

These tools are injected into `dspy.ReAct("question -> answer")`, so the baseline is already a tool-using multi-hop QA system rather than just a standalone retriever.

#### 6. Retrieval evaluation

Implemented in [eval_retrieval.py](/home/maxh/ssd1/github_clone/dspy/IR_LLM_Project/src/rag_baseline/eval_retrieval.py).

The current evaluation pipeline:

- reads `DSPy_Multihop_QA.json`,
- extracts gold file paths from `retrieval_path`,
- compares top-k retrieved files against the gold files.

This is useful because it separates:

- retrieval failure
- answer-generation failure after successful retrieval

#### 7. Batch QA and local answer evaluation

Implemented in:

- [run_benchmark.py](/home/maxh/ssd1/github_clone/dspy/IR_LLM_Project/src/rag_baseline/run_benchmark.py)
- [eval_answers.py](/home/maxh/ssd1/github_clone/dspy/IR_LLM_Project/src/rag_baseline/eval_answers.py)

Current capabilities:

- run the ReAct agent over the whole benchmark
- save each trajectory
- record latency
- record tool-call counts
- record which chunks and files were actually used
- record token usage
- compute local `EM/F1`

This is already enough for a first round of course-project experiments.

### Directory structure

```text
IR_LLM_Project/
├── README.md
├── requirements_dense_rag.txt
├── docs/
│   ├── README.md
│   ├── dense_rag_baseline_plan.md
│   └── 信息检索与大语言模型-课程项目.md
└── src/
    └── rag_baseline/
        ├── agent.py
        ├── benchmark_utils.py
        ├── build_index.py
        ├── chunking.py
        ├── config.py
        ├── eval_answers.py
        ├── eval_retrieval.py
        ├── indexer.py
        ├── io_utils.py
        ├── jina_embedder.py
        ├── metrics.py
        ├── retriever.py
        ├── run_benchmark.py
        ├── search_index.py
        └── types.py
```

### Implemented vs not yet implemented

#### Implemented

- Dense RAG baseline framework
- structure-aware chunking
- local Jina-based dense retrieval
- benchmark retrieval evaluation
- DSPy ReAct integration
- batch QA runner
- local answer-evaluation script

#### Not yet implemented

- `ripgrep` / exact-match baseline
- unified Dense vs Grep comparison pipeline
- plotting and final experiment summary
- stronger dense variants
  - `Dense-Flat`
  - `Dense-Struct-DualView`
- LLM-judge-based semantic answer evaluation

### Related documents

- Project description: [信息检索与大语言模型-课程项目.md](/home/maxh/ssd1/github_clone/dspy/IR_LLM_Project/docs/信息检索与大语言模型-课程项目.md)
- Dense baseline plan: [dense_rag_baseline_plan.md](/home/maxh/ssd1/github_clone/dspy/IR_LLM_Project/docs/dense_rag_baseline_plan.md)
