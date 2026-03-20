# data-copilot-v2


✨ **基于大语言模型 (LLM) 和并发预测模型的自然语言数据库查询系统**

通过自然语言提问，使用大语言模型智能解析数据库结构，对数据进行智能多表结构化查询和统计计算，根据查询结果智能绘制多种图表。 
通过并发预测模型智能预测最佳并发量，找到生成成功率和 LLM 调用成本的平衡。

引入多线程并发执行，同时多次独立提问，降低 LLM 输出不稳定导致整体生成失败的概率，提高系统稳定性和响应速度。
但是盲目增加并发数量会导致过大的 API 资源和性能浪费。
使用 bert 加 regression 的方式，根据问题难度，预测最佳并发数量（Concurrent） 值。
找到生成成功率和 LLM 调用成本的平衡。
Pywebio 交互式前端网页，不必须 openai api，100%纯 Python 代码。 


## 功能简介

- 1, 使用自然语言提问
- 2, 实现多表结构化查询和统计计算
- 3, 实现智能绘制多种类型的图表和交互式图表制作
- 4, 智能解析数据库结构，使用不同的 mysql 数据库无需额外配置
- 4, 支持多线程并发查询
- 5, 能够处理大语言模型表现不稳定等异常情况
- 6, 支持本地离线部署 (需 GPU) `huggingface` 格式模型 (例如`qwen-7b`) 
- 7, 支持 `openai` 格式和 dashscope `qwen` 的 api 接口

## 技术创新点

- 通过异常和断言信息回输进行重试提问，改善大语言模型输出表现不稳定的情况。
- 多线程并发提问，提高响应速度和稳定性。
- 使用数据库的 DataFrame 映射进行操作，避免通过诱导大语言模型进行 sql 注入攻击的风险。
- 引入词嵌入模型和向量数据库，替代单纯的正则表达式，解决大语言模型的模糊输出，到确定的系统代码执行的映射难题。

- ✨ 使用 BERT(Bidirectional Encoder Representation from Transformers) 将提问文本和 prompt **向量化**，再使用线性回归, Logistic 回归, nn, Transformer 多种模型进行**回归**，根据提问文本**预测**合适的并发数量和重试次数，对比不同模型的效果，

## 演示截图

本项目提供了完整的演示截图资源，包含系统主页、自然语言问答结果以及图表展示效果。

## 基本技术原理

### 单次生成的基本流程

系统采用基于工具选择、代码生成、执行校验与失败重试的闭环流程来完成单次任务生成。

1. 自然语言问题输入系统之后，会先和提前预设的工具集描述信息一起合成 prompt 输入 LLM，让 LLM 选择合适的解决问题的工具。

2. 从数据库中，获取数据的结构信息（Dataframe 数据摘要） 

3. 把数据摘要，工具建议信息，输入到 LLM，使其编写 python 代码解决问题。

4. 从 LLM 的回答中截取代码，交付 Python 解释器执行。

5. 如果代码运行出现异常，把异常信息和问题代码组成新的prompt，回输给 LLM 再次尝试（回到`3`）。直到运行成功或超过最大重试次数。

6. 如果代码没有运行出现异常，则对程序输出进行断言。如果不是期望的类型，则把断言信息和问题代码组成新的prompt，回输给 LLM 再次尝试（回到`3`）。直到断言成功或超过最大重试次数。

7. 把成功的代码运行输出（图表）显示到用户界面上，根据输出数据启动交互式绘图界面。

#### Retries 学习

Retries 学习模块通过问题难度特征预测合适的重试次数，以提升成功率并控制响应时间。

使用 BERT(Bidirectional Encoder Representation from Transformers) 加 `regression` 的方式，根据问题难度，预测最佳 `retries` 值。找到生成成功率和重试次数的平衡，提高响应速度。


### 并发生成控制

并发生成控制模块用于在成功率与调用成本之间动态平衡，避免盲目增加并发造成资源浪费。

反复的异常和断言回输会导致，prompt 越来越长， LLM 失去注意力，影响生成效果。 LLM 第一次的错误回答也会影响之后的生成。这时从头再来，可能获得更好的结果。

所以引入多线程并发执行，同时多次独立提问，降低 LLM 输出不稳定导致整体生成失败的概率，提高系统稳定性和响应速度。

#### Concurrent 学习

Concurrent 学习模块通过回归模型预测最优并发数量，提升整体吞吐与稳定性。

使用 bert 加 `regression` 的方式，根据问题难度，预测最佳并发数量（`Concurrent`） 值。找到生成成功率和 LLM 调用成本的平衡，提高响应速度。

## 如何使用

### 安装依赖

python 版本 3.9

```bash
pip install -r requirement.txt
```

### 填写配置信息

`./config/config.yaml` 是配置信息文件。

#### 数据库配置
连接即可，模型会自动读取数据库结构，无需额外配置
```yml
mysql: mysql+pymysql://root:123456@127.0.0.1/data_copilot
# mysql: mysql+pymysql://用户名:密码@地址:端口/数据库名
```

#### 大语言模型配置
如果使用 dashscope `qwen` api （推荐）
```yml
llm:
  model_provider: qwen #qwen #openai
  model: qwen1.5-110b-chat
  url: ""

# qwen1.5-72b-chat   qwen1.5-110b-chat
# qwen-turbo  qwen-plus   qwen-max   qwen-long

```


如果使用 openai api （此处填写的是 glm 的 openai 兼容 api）

```yml
llm:
  model_provider: openai
  model: glm-4
  url: "这里填写 openai 兼容 API 地址"

# glm-4
```

如果需要本地离线部署，相关代码在 `./llm_access/qwen_access.py`

#### 获取 apikey

如果从阿里云 dashscope 控制台获取 `qwen` 大语言模型的 api-key


保存 `api-key` 到 `llm_access/api_key_qwen.txt`

如果使用 `openai` 格式 api 的 api-key

保存 `api-key` 到 `llm_access/api_key_openai.txt`

### 运行

main.py 是项目入口，运行此文件即可启动服务器

```bash
python main.py
```

本仓库为后端 api。可用运行 `ask_test.py` 进行测试。

```bash
python ask_test.py
```

生成的图表会保存到 `.temp.html` 或  `.temp.png`



# 开源许可证

MIT 开源许可证：

版权所有 (c) 2023 bytesc

特此授权，免费向任何获得本软件及相关文档文件（以下简称“软件”）副本的人提供使用、复制、修改、合并、出版、发行、再许可和/或销售软件的权利，但须遵守以下条件：

上述版权声明和本许可声明应包含在所有副本或实质性部分中。

本软件按“原样”提供，不作任何明示或暗示的保证，包括但不限于适销性、特定用途适用性和非侵权性。在任何情况下，作者或版权持有人均不对因使用本软件而产生的任何索赔、损害或其他责任负责，无论是在合同、侵权或其他方面。
