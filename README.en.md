# data-copilot-v2


✨ **Natural Language Database Query System based on Large Language Models (LLM) and Concurrent Prediction Models**

By asking questions in natural language, the system uses large language models to parse database structures, perform intelligent multi-table structured queries and statistical computation, and generate different kinds of charts from query results.
A concurrency prediction model is used to estimate the optimal concurrency level, balancing generation success rate and LLM calling cost.

The project introduces multi-threaded concurrent execution and independent repeated asking, reducing overall failure caused by unstable LLM output, and improving stability and response speed.
However, blindly increasing concurrency can waste API resources and performance.
By using BERT with regression, the system predicts a suitable Concurrent value based on question difficulty.
This helps balance generation success rate and LLM calling cost.
It also provides a Pywebio interactive web interface, does not require OpenAI API, and is implemented with 100% pure Python code.


## Feature Overview

- 1, Ask questions in natural language
- 2, Perform multi-table structured querying and statistical computation
- 3, Generate multiple chart types and support interactive chart creation
- 4, Intelligently parse database schema, no extra setup needed when switching MySQL databases
- 4, Support multi-threaded concurrent querying
- 5, Handle exceptions caused by unstable large language model behavior
- 6, Support local offline deployment (GPU required) with huggingface format models (for example qwen-7b)
- 7, Support openai-compatible APIs and dashscope qwen APIs

## Technical Innovations

- Retry questions by feeding back exception and assertion information, improving unstable LLM outputs.
- Use multi-threaded concurrent asking to improve speed and stability.
- Use DataFrame mapping of database data to avoid SQL injection risks caused by prompt manipulation.
- Introduce word embedding models and vector databases to replace pure regular-expression mapping from fuzzy LLM output to deterministic executable code.

- Use BERT (Bidirectional Encoder Representation from Transformers) to vectorize question text and prompts, then apply multiple regression approaches (linear regression, Logistic regression, nn, Transformer) to predict suitable concurrency and retry counts based on question text.

## Demo Screenshots

This project provides complete demo screenshot resources, including the main system page, natural-language Q and A results, and chart display outcomes.

## Basic Technical Principles

### Basic Flow of Single-Run Generation

The system uses a closed loop of tool selection, code generation, execution verification, and failure retry to complete each single generation task.

1. After a natural language question is entered, it is combined with preset tool description information into a prompt and sent to the LLM, which selects the proper tool for the task.

2. Retrieve schema information from the database (DataFrame summary).

3. Input schema summary and tool suggestions to the LLM so it writes Python code for the task.

4. Extract code from the LLM response and run it in the Python interpreter.

5. If execution raises exceptions, combine exception messages and the code into a new prompt and retry (back to step 3), until success or retry limit reached.

6. If execution does not raise exceptions, assert output type and format. If assertion fails, combine assertion feedback and code into a new prompt and retry (back to step 3), until assertion passes or retry limit reached.

7. Display successful output (charts) in the user interface, and start interactive charting based on output data.

#### Retries Learning

The retries learning module predicts suitable retry counts from question-difficulty features, improving success rate while controlling response time.

Using BERT with regression, the system predicts the best retries value based on question difficulty, balancing generation success rate and retry count to improve response speed.


### Concurrent Generation Control

The concurrent generation control module dynamically balances success rate and calling cost, avoiding waste caused by blindly increasing concurrency.

Repeated exception and assertion feedback can make prompts longer and longer, reducing LLM attention and generation quality. The first incorrect LLM answer can also affect later outputs, so starting over independently can sometimes work better.

Therefore, the system introduces multi-threaded concurrent execution and independent repeated asking, reducing total failure probability caused by unstable LLM behavior and improving stability and response speed.

#### Concurrent Learning

The concurrent learning module predicts an optimal concurrency number with regression models to improve throughput and stability.

Using BERT with regression, the system predicts an optimal Concurrent value based on question difficulty, balancing generation success rate and LLM calling cost to improve response speed.

## How to Use

### Install Dependencies

Python version 3.9

```bash
pip install -r requirement.txt
```

### Fill in Configuration

The configuration file is ./config/config.yaml.

#### Database Configuration
Just provide the connection string. The model reads database schema automatically, and no extra configuration is needed.

```yml
mysql: mysql+pymysql://root:123456@127.0.0.1/data_copilot
# mysql: mysql+pymysql://username:password@host:port/database
```

#### Large Language Model Configuration
If you use dashscope qwen API (recommended):

```yml
llm:
  model_provider: qwen #qwen #openai
  model: qwen1.5-110b-chat
  url: ""

# qwen1.5-72b-chat   qwen1.5-110b-chat
# qwen-turbo  qwen-plus   qwen-max   qwen-long
```

If you use openai-compatible API (example shown for glm compatible API):

```yml
llm:
  model_provider: openai
  model: glm-4
  url: "Fill your openai-compatible API endpoint here"

# glm-4
```

If local offline deployment is needed, related code is in ./llm_access/qwen_access.py.

#### Obtain API Key

If you obtain qwen API key from Alibaba Cloud Dashscope console,

save api-key to llm_access/api_key_qwen.txt.

If you use openai-compatible API key,

save api-key to llm_access/api_key_openai.txt.

### Run

main.py is the project entry point. Run it to start the server.

```bash
python main.py
```

This repository is backend API. You can run ask_test.py for testing.

```bash
python ask_test.py
```

Generated charts are saved to .temp.html or .temp.png.

