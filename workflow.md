# data-copilot-v2 Workflow

## 1. Purpose

This document describes the end-to-end runtime workflow of `data-copilot-v2`, including the newly integrated modules:

1. Word embedding module (`pgv/embedding.py`)
2. Vector database module (`pgv/write_db.py`, `pgv/ask.py`)

The goal is to combine deterministic semantic retrieval with LLM flexibility, improving prompt stability and schema grounding.


## 2. Repository Responsibilities

### 2.1 data-copilot-v2 (this repo)

1. Convert user natural language questions into executable Python code.
2. Load relational schema and sample data from MySQL.
3. Run exception/assertion feedback retries.
4. Build semantic schema knowledge with embedding + PGVector.
5. Inject semantic retrieval hints into the final prompt.

### 2.2 data-copilot-v2-controller (external repo)

1. UI and orchestration logic.
2. Predict `concurrent` and `retries` values.
3. Collect data for offline training loops.


## 3. Startup Workflow

### 3.1 Configuration Loading

1. `config/get_config.py` loads YAML from `config/config.yaml`.
2. Required sections are validated: `mysql`, `llm`, `ai`.
3. Optional `vector` section is normalized with defaults.
4. Runtime directories are created (`tmp_imgs`).

### 3.2 Infrastructure Initialization

1. `data_access/db_conn.py` builds SQLAlchemy engine and verifies DB connectivity.
2. `llm_access/LLM.py` initializes the configured model provider.
3. FastAPI startup calls `fetch_data()`.
4. `fetch_data()` loads schema payload and triggers `sync_schema_knowledge(...)`.


## 4. Schema Vectorization Workflow

### 4.1 Embedding Module (`pgv/embedding.py`)

1. Resolve model id/path from `vector.embedding_model`.
2. Resolve device from `vector.embedding_device`.
3. Build and cache `HuggingFaceEmbeddings` instance.
4. Reuse cache across requests to reduce cold-start overhead.

### 4.2 Vector Store Module (`pgv/write_db.py`)

1. Build PG connection string from either:
2. `vector.connection_string`
3. or `vector.db.*` fields.
4. Open `PGVector` store with configured distance strategy.
5. Rebuild collection when schema hash changes.
6. Run similarity search with score for runtime retrieval.

### 4.3 Schema Knowledge Builder (`pgv/ask.py`)

1. Convert runtime schema payload into semantic documents:
2. table schema docs
3. foreign-key relation docs
4. (optional) comment-enriched column descriptors
5. Hash document payload to avoid unnecessary re-indexing.
6. Rebuild collection only when schema changed or force rebuild is requested.


## 5. Online Inference Workflow

### 5.1 Prompt Construction (`ask_ai/ask_api.py`)

For each candidate generation:

1. Build base prompt with question, sampled tables, keys, and comments.
2. Query vector DB using the original user question.
3. Append semantic hints to prompt when retrieval is available.
4. Append strict output constraints.

This turns the final prompt into:

`Question + Structured Schema Context + Semantic Vector Hints + Output Contract`

### 5.2 Code Generation and Execution

1. Call LLM to generate Python code block.
2. Parse code block from model output.
3. Execute `process_data(dataframes_dict)`.
4. Validate output type using task-specific assertions.
5. On failure, feed back exception/assertion messages and retry.


## 6. Concurrency and Success Semantics

Each task module (`ask_ai_for_pd.py`, `ask_ai_for_graph.py`, `ask_ai_for_echart.py`) uses:

1. Thread-level parallel candidate generation (`concurrent`).
2. Per-candidate retry budget (`retries`).
3. Round-level retry loop (`ai.tries`).
4. Early stop when success count reaches `ai.wait`.

Success ratio remains:

$$
\mathrm{success} = \frac{\text{successful candidates in current round}}{\text{requested concurrent workers}}
$$


## 7. Fault Tolerance and Degradation

Vector modules are integrated with safe degradation:

1. If vector is disabled (`vector.enabled: false`), core pipeline runs normally.
2. If embedding or PGVector initialization fails, prompt generation still proceeds.
3. If retrieval fails, only semantic hint block is skipped.
4. Main business APIs still return normal responses from the classic pipeline.


## 8. Operational Steps

### 8.1 Enable Vector Retrieval

1. Set `vector.enabled: true` in `config/config.yaml`.
2. Configure either:
3. `vector.connection_string`
4. or `vector.db` connection fields.
5. Install dependencies from `requirement.txt`.

### 8.2 Initialize PostgreSQL

1. Create a PostgreSQL database with `pgvector` extension.
2. Optional: run `pgv/create_table.sql` manually.
3. Start service with `python main.py`.
4. On startup, schema semantic index is built automatically.


## 9. Architectural Benefits

1. Better schema grounding for fuzzy user questions.
2. Reduced reliance on brittle regex-only mapping.
3. More deterministic joins/columns selection via semantic hints.
4. Backward-compatible integration with low risk to existing APIs.
5. Clear split between embedding, vector storage, and retrieval orchestration.