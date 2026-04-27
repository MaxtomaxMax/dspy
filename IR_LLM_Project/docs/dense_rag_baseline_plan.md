# 基于 jina-code-embeddings-1.5b 的传统 Dense RAG Baseline 方案

## 1. 目标定义

本方案用于为课程项目构建一个传统的基于向量检索的代码库多跳问答 baseline。系统面向当前仓库中的 `DSPy_Multihop_QA.json` 基准测试集，知识源限定为 `dspy/**/*.py` 源码文件，检索模型选择 `jinaai/jina-code-embeddings-1.5b`。

该 baseline 的定位是：

- 保持检索范式为传统 dense retrieval；
- 保持与后续 `ripgrep` / 精确匹配 baseline 相同的问答与推理框架；
- 仅替换检索后端，以保证实验对比公平。

## 2. 设计原则

### 2.1 不做“弱 baseline”

`jina-code-embeddings-1.5b` 属于 query/document 非对称的代码 embedding 模型，查询与文档需要使用不同的任务前缀。若直接复用同一个编码函数处理 query 和 corpus，会显著削弱 dense baseline 的效果。因此实现中必须显式区分：

- `nl2code_query`
- `nl2code_document`

### 2.2 不用纯字符切块作为主方案

代码库检索不同于开放域自然语言检索。对当前 DSPy 仓库，更合理的切块单位是符号级结构，而非简单的固定字符窗口。因此主方案采用 structure-aware chunking：

- 文件概览块 `file_overview`
- 顶层类块 `class`
- 顶层函数块 `function`
- 类方法块 `method`

每个 chunk 同时保留：

- 文件路径
- 符号限定名
- 签名
- 文档字符串
- 原始代码

这样 dense retrieval 可以同时利用路径名、符号名与代码语义。

### 2.3 保持传统 RAG 边界

虽然采用结构化切块，但检索阶段仍然只使用向量召回与相似度排序，不引入 AST 图检索、调用图遍历或正则精确匹配逻辑。这样仍属于传统 dense RAG baseline。

## 3. 系统结构

实现目录放在 `IR_LLM_Project/src/rag_baseline/` 下，主要模块如下：

- `config.py`
  - 统一管理路径、默认超参数和模型名。
- `types.py`
  - 定义 chunk、检索结果、benchmark 样本等数据结构。
- `chunking.py`
  - 解析 Python 文件并构建结构化 chunk。
- `jina_embedder.py`
  - 封装 `jina-code-embeddings-1.5b` 的 query/document 双编码接口。
- `indexer.py`
  - 构建并保存 chunk 索引与 embedding。
- `retriever.py`
  - 加载本地索引并执行 brute-force cosine retrieval。
- `agent.py`
  - 将检索器封装为 DSPy 工具，构造 ReAct agent。
- `eval_retrieval.py`
  - 基于 benchmark 的 `retrieval_path` 做检索侧评测。
- `build_index.py`
  - 命令行索引构建入口。

## 4. 索引构建策略

### 4.1 语料范围

默认仅索引 `dspy/**/*.py`。这与当前 benchmark 的 gold retrieval path 基本一致，可以减少无关文档污染。

### 4.2 chunk 形式

每个 chunk 保存以下字段：

- `chunk_id`
- `chunk_type`
- `file_path`
- `symbol_path`
- `start_line`
- `end_line`
- `signature`
- `docstring`
- `text`
- `code`

其中用于 embedding 的文本建议拼接为：

1. 文件路径
2. chunk 类型
3. 符号路径
4. 函数或类签名
5. 文档字符串
6. 原始代码

### 4.3 索引保存格式

索引目录中至少保存：

- `chunks.jsonl`
- `embeddings.npy`
- `metadata.json`

由于当前 DSPy 仓库规模不大，symbol 级 chunk 数量预计在一到两千量级，直接使用 numpy brute-force cosine 足够，且复现成本最低，因此第一版不引入 FAISS。

## 5. 检索与 Agent 设计

### 5.1 检索器

检索器使用两阶段最小设计：

1. 将自然语言 query 用 `nl2code_query` 编码；
2. 与 document embedding 做 cosine similarity 排序。

返回结果包含：

- `chunk_id`
- `score`
- `file_path`
- `symbol_path`
- 截断后的文本片段

### 5.2 DSPy 工具

为适配多跳推理，至少提供两个工具：

- `vector_search(query, top_k=8)`
- `open_chunk(chunk_id)`

前者适合找候选证据，后者适合深读具体片段，能减少 ReAct 在单步中塞入过长代码上下文。

### 5.3 ReAct 配置

问答层使用 `dspy.ReAct("question -> answer", tools=[...])`。为了与后续 grep baseline 保持可比性，建议统一：

- 相同的生成模型
- 相同的 `max_iters`
- 相同的输出字段
- 相同的最终答案抽取逻辑

## 6. 评测协议

当前 benchmark 样本包含：

- `question`
- `answer`
- `retrieval_path`
- `hops`

其中 `retrieval_path` 可直接转化为 retrieval gold signal。第一版实现优先支持以下检索指标：

- `FileRecall@k`
- `MRR`
- `ChainCoverage`

后续完整问答评测再补充：

- `Answer EM`
- `Semantic F1`
- `Avg tool calls`
- `Token usage`
- `Latency`

## 7. 建议实验矩阵

建议至少报告以下三种配置：

- `Dense-Struct`
  - 结构化 chunk + 单路 dense 检索
- `Dense-Flat`
  - 固定长度切块 + 单路 dense 检索
- `Dense-Struct-DualView`
  - 结构化 chunk + `full_text` 与 `metadata_light` 双路召回再融合

如果时间有限，至少先完成 `Dense-Struct`，它应作为你的主 dense baseline。

## 8. 当前实现范围

本次代码实现先覆盖：

- Python 结构化 chunking
- Jina embedding 封装
- 本地索引构建与加载
- brute-force dense retrieval
- DSPy ReAct 工具接入
- benchmark 检索侧评测

完整答案评测与实验表格汇总可在后续阶段补充。

## 9. 当前代码入口

当前代码已提供以下命令行入口：

- 建索引
  - `PYTHONPATH=IR_LLM_Project/src python -m rag_baseline.build_index --device cuda:0`
- 单题检索
  - `PYTHONPATH=IR_LLM_Project/src python -m rag_baseline.search_index --query "..." --top-k 8`
- 检索评测
  - `PYTHONPATH=IR_LLM_Project/src python -m rag_baseline.eval_retrieval`
- 批量问答运行
  - `PYTHONPATH=IR_LLM_Project/src python -m rag_baseline.run_benchmark --lm-model openai/gpt-4o-mini`
- 答案评测
  - `PYTHONPATH=IR_LLM_Project/src python -m rag_baseline.eval_answers --predictions-path IR_LLM_Project/results/dense_rag_predictions.jsonl`

## 10. 依赖说明

由于当前仓库并未在本地环境中完整安装 DSPy 与 Jina embedding 依赖，运行 baseline 前建议至少补齐：

- `pip install -e .`
- `pip install -r IR_LLM_Project/requirements_dense_rag.txt`

其中：

- `pip install -e .` 用于补齐当前仓库依赖，如 `cloudpickle`
- `requirements_dense_rag.txt` 用于补齐 Jina 本地 embedding 所需依赖
