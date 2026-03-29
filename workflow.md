# Database Query System Workflow

## 1. 文档目标

本文档统一说明 Agent 的运行流程，采用与 Controller 完全一致的章节格式，重点回答三件事：

1. Agent 如何承接并发预测模型输出（`concurrent`、`retries`）并执行请求。
2. 图形界面相关接口（graph）在 Agent 侧的处理链路。
3. 启动各个 `.py` 文件的顺序与每个文件的职责。

---

## 2. 总体架构

Agent 包含两条主线：

1. 并发预测模型协同主线（重点）
	- Controller 预测并发参数，Agent 接收参数后执行并发候选生成、重试和早停。
2. 图形界面服务主线
	- FastAPI 提供 `/ask/graph-steps` 等接口，返回图片结果。

核心关系：

1. Agent 不训练并发模型，但负责消费预测结果并落地执行。
2. Agent 的返回质量直接影响 Controller 的日志采样与后续模型训练。

---

## 3. 并发预测模型启动流程（重点）

### 3.1 阶段 A：Agent 基础初始化

1. `config/get_config.py`
	- 读取并校验 `config/config.yaml`，初始化 `mysql`、`llm`、`ai`、`vector`。

2. `data_access/db_conn.py`
	- 创建数据库连接并启动连通性校验。

3. `llm_access/LLM.py`
	- 初始化模型提供商并构建可调用的 LLM 客户端。

4. `main.py`
	- 启动 FastAPI 服务，并在 startup 中 `fetch_data()` 缓存数据库结构信息。

### 3.2 阶段 B：并发参数协同执行

1. Controller 侧 `main.py` 会把预测出的并发参数写入请求体：
	- `concurrent: [c1, c2]`
	- `retries: [r1, r2]`

2. Agent 侧 `main.py` 接收请求后，按接口类型拆成单阶段或双阶段任务。

3. `ask_ai/ask_ai_for_graph.py`、`ask_ai/ask_ai_for_pd.py`
	- 基于 `concurrent` 创建线程池并行生成候选答案。
	- 基于 `retries` 在候选内部执行失败重试。
	- 基于 `ai.wait` 达到成功阈值后提前返回。

4. 成功率语义（每轮）保持为：

$$
\mathrm{success} = \frac{\text{当前轮成功候选数}}{\text{本轮并发候选总数}}
$$

### 3.3 阶段 C：提示增强与执行闭环

1. `ask_ai/ask_api.py`
	- 组装基础 Prompt（问题 + schema + 键关系 + 约束）。
	- 可选叠加 `pgv/ask.py` 的向量检索提示。

2. LLM 生成代码后执行 `process_data(dataframes_dict)`。

3. 通过断言/异常反馈回输触发重试，最终返回结构化响应给 Controller。

---

## 4. 图形界面流程

1. 图形界面入口在 Controller 的 `PyWebIO` 页面，但图形生产由 Agent 负责。
2. Agent 的图形接口包括：
	- `/ask/graph-steps`：返回 PNG 及 `image_data(base64)`。
3. Agent 在接口层完成：请求参数校验 -> 并发执行 -> 结果打包。
4. Controller 收到响应后渲染图片或保存 HTML，形成最终可视化交互。

图形界面与并发模型关系：

1. 界面层负责“展示与交互”。
2. Agent 负责“并发执行与结果产出”。
3. 两者通过统一 HTTP 协议解耦协作。

---

## 5. 关键文件职责

1. `main.py`：FastAPI 服务入口与 API 编排。
2. `ask_ai/ask_ai_for_graph.py`：图像图表并发生成链路。
3. `ask_ai/ask_ai_for_pd.py`：表格/数据输出并发生成链路。
4. `ask_ai/ask_api.py`：Prompt 组装、执行与反馈重试闭环。
5. `data_access/read_db.py`：读取数据表、外键、注释等结构信息。
6. `data_access/db_conn.py`：数据库连接初始化与验证。
7. `llm_access/LLM.py`：LLM 客户端初始化。
8. `pgv/embedding.py`：Embedding 模型加载与设备配置。
9. `pgv/write_db.py`、`pgv/ask.py`：向量库写入、检索与 schema 同步。

---

## 6. 推荐启动顺序

```bash
# 1) 启动 Agent 服务
python main.py

# 2) 再启动 Controller（在线并发预测生效）
# 在 Controller 项目执行 python main.py
```

说明：

1. Agent 必须先启动，Controller 才能调用 `/ask/*` 接口。
2. Controller 的并发预测模型输出参数由 Agent 在运行时执行。

---

## 7. 常见问题

1. Controller 提示接口超时
	- 原因：Agent 未启动或执行链路耗时过高。
	- 处理：先确认 Agent 端口可达，再调高超时与重试。

2. Agent 启动时报数据库错误
	- 原因：`config/config.yaml` 中 `mysql` 配置错误或数据库未创建。
	- 处理：修正连接串并检查库权限。

3. 向量检索初始化失败
	- 原因：`vector` 配置或 PGVector 依赖未就绪。
	- 处理：可先关闭 `vector.enabled`，核心接口仍可运行。