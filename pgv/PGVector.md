# Agent 向量数据库服务说明

本文档基于 pgv 目录下现有代码实现，梳理 Agent 服务提供的向量数据库能力、运行机制与配置要点。

## 1. 模块目标

pgv 模块负责把数据库结构信息转为语义向量知识，并在用户提问时进行语义检索，为大模型生成代码提供更可靠的表名、字段名和关联路径提示。

简化理解：

1. 写入阶段：把 schema 信息做 embedding，写入 PGVector。
2. 检索阶段：根据用户问题检索最相关 schema 文档。
3. 增强阶段：把检索结果拼接到 prompt，提升生成代码准确率。

## 2. 对外提供的服务能力

来自 pgv 对外导出接口：

1. build_semantic_context(question)
2. get_schema_vector_service()
3. sync_schema_knowledge(schema_payload, force_rebuild=False)

对应能力说明：

1. Schema 同步服务
将运行时加载到内存的表结构信息同步到向量库。

2. 语义检索服务
对自然语言问题进行向量检索，返回语义最接近的 schema 文档。

3. Prompt 增强服务
把检索结果整理为文本上下文，拼接进 LLM 提示词中。

## 3. 代码结构与职责分工

### 3.1 ask.py

核心服务类 SchemaVectorService，负责：

1. 懒初始化 embedding 模型与 PGVector store。
2. 规范化 schema payload。
3. 生成可索引语义文档。
4. 计算 payload 哈希，避免重复重建。
5. 向量检索并按距离阈值过滤。
6. 生成可直接注入 prompt 的上下文文本。

同时提供单例入口：

1. get_schema_vector_service
2. sync_schema_knowledge
3. build_semantic_context

### 3.2 embedding.py

负责 embedding 侧能力：

1. 解析模型名与设备配置。
2. 创建 HuggingFaceEmbeddings。
3. 基于 model_name + device 做进程内缓存复用。
4. 提供文本批量向量化函数 embed_texts。

默认模型：shibing624/text2vec-base-multilingual。
默认设备：cpu。

### 3.3 write_db.py

负责 PGVector 持久层：

1. 组装 PostgreSQL 连接串。
2. 解析距离策略（cosine、euclidean、inner product）。
3. 创建 store（可选 pre_delete_collection）。
4. 全量重建 collection 并写入文本与 metadata。
5. 执行 similarity_search_with_score。

### 3.4 create_table.sql

用于初始化 pgvector 相关表与索引：

1. 启用 extension vector。
2. 建立集合表 langchain_pg_collection。
3. 建立向量表 langchain_pg_embedding。
4. 建立 collection_id 索引。
5. 建立 ivfflat 向量索引（cosine）。

### 3.5 cmd.txt

提供本地 pgvector Docker 启动和基础连接命令示例。

## 4. 端到端调用链路

### 4.1 Schema 同步链路

1. API 请求到达后，fetch_data 读取业务数据库表数据和元数据。
2. fetch_data 调用 sync_schema_knowledge(payload, force_rebuild=force_reload)。
3. SchemaVectorService.sync_schema_payload 执行：
- 生成 schema 文档文本与 metadata。
- 计算哈希并判断是否需要重建。
- 触发 write_db.rebuild_collection 全量重建。

说明：当前策略是重建型写入，不是增量 upsert。

### 4.2 检索增强链路

1. 生成 LLM prompt 时，ask_ai.ask_api 调用 build_semantic_context(question)。
2. SchemaVectorService.retrieve 执行相似度检索。
3. 命中结果转为语义提示段落。
4. 提示段落附加到最终 prompt 中，指导模型使用正确表结构与关联。

## 5. Schema 文档构建策略

构建时会从 payload 中提取：

1. tables_data：表名与字段列表。
2. foreign_keys：外键映射。
3. comments：表注释与字段注释。

产出的文档类型：

1. table_schema 文档
包含表名、字段、注释、外键信息。

2. foreign_key 文档
每条外键关系单独一条文档，文本形如：
Join relation: table_a.col_x references table_b.col_y

好处：

1. 可同时支持表语义召回与关系路径召回。
2. 有利于生成 join 语句时的字段选择。

## 6. 检索与过滤策略

默认检索参数由 config.yaml 的 vector 段控制：

1. top_k：默认 6。
2. max_distance：默认 null，不做距离阈值裁剪。
3. distance_strategy：默认 cosine。

运行行为：

1. 先取 top_k 结果。
2. 如果设置 max_distance，则丢弃距离大于阈值的命中。
3. 仅保留有有效 page_content 的结果。
4. 输出结构为 SemanticMatch(text, score, metadata)。

## 7. 配置项说明

vector 配置支持：

1. enabled
是否开启向量能力。关闭时检索和同步都短路返回。

2. embedding_model
HuggingFace 模型名或本地相对路径。

3. embedding_device
如 cpu 或 cuda。

4. collection_name
向量集合名，默认 schema_knowledge。

5. top_k
检索返回条数上限。

6. max_distance
最大允许距离；为空表示不限制。

7. distance_strategy
向量距离策略：cosine、euclidean、max_inner_product。

8. connection_string
若配置则优先使用，覆盖 db 子配置。

9. db
用于拼接 PostgreSQL 连接参数。

## 8. 数据库侧实现要点

向量表结构采用 LangChain PGVector 习惯命名：

1. langchain_pg_collection 存集合元信息。
2. langchain_pg_embedding 存向量、文本、metadata。

索引要点：

1. collection_id BTree 索引提升集合过滤效率。
2. embedding 上 ivfflat 索引提升向量近邻查询速度。
3. create_table.sql 中向量维度为 768，需要与 embedding 模型输出维度一致。

## 9. 当前实现的稳定性设计

1. 单例服务
避免重复创建模型和连接对象。

2. 线程锁
初始化与同步过程都在锁内保证并发安全。

3. 模型缓存
按模型名和设备缓存 embedding function，减少重复加载成本。

4. 失败降级
向量初始化、同步、检索失败时只记录 warning，不中断主问答流程。

5. 哈希去重同步
schema 未变化时不重复重建，降低数据库与 embedding 开销。

## 10. 与主业务的集成点

1. 数据读取后自动同步向量知识
在 API 的 fetch_data 流程中触发。

2. Prompt 组装时自动注入语义提示
在 ask_ai.ask_api 的 _append_semantic_context 流程中触发。

3. 对业务调用方透明
即使向量功能异常，主流程仍可继续，仅减少语义增强效果。

## 11. 部署与运行注意事项

1. 需要 PostgreSQL 安装 pgvector extension。
2. 向量库初始化时先执行 create_table.sql。
3. 若使用 cmd.txt 的容器参数，注意与 config.yaml 端口保持一致。
4. embedding_model 与 create_table.sql 中向量维度必须匹配。
5. 首次加载模型可能较慢，建议预热。

## 12. 常见问题排查

1. 无检索结果
检查 vector.enabled 是否为 true；检查是否执行过 sync_schema_knowledge；检查 top_k 和 max_distance。

2. 连接失败
检查 connection_string 或 db 参数；确认 PostgreSQL 可达且账号有权限。

3. 检索慢
检查是否存在 ivfflat 索引；检查集合规模；检查 embedding 设备是否可用。

4. 结果不准
检查 schema payload 中注释和外键信息是否完整；适当调整 top_k、distance_strategy 和 max_distance。

5. 重建频繁
检查 schema payload 是否每次都产生非稳定结构导致哈希变化。

## 13. 可演进方向

1. 从全量重建升级为增量更新。
2. 增加检索结果重排与去重。
3. 增加多路召回（字段名关键词检索 + 向量检索）。
4. 对检索命中做质量监控指标（命中率、距离分布、失败率）。
5. 将模型与向量库健康状态暴露为可观测指标。
