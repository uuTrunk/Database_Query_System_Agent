# 实验：BERT 并发预测 vs 单线程

该目录仅包含独立的实验脚本，不会修改 Agent 或 Training 的业务逻辑。

## 脚本

- run_bert_concurrency_experiment.py

## 实验功能

1. 调用 Training 服务的 predict 接口，获取基于 BERT 的并发线程推荐值。
2. 对每个问题分别调用两次 Agent API：
   - 基线组：固定单线程并发（默认 1）
   - 预测组：使用 BERT 推荐的并发数（1 到 5）
3. 每次实验都会把逐题结果保存为一个新的 CSV 文件。
4. 自动生成可视化 PNG 图表，包含成功率、延迟以及预测线程分布。

## 前置条件

- Agent 服务已经启动，默认地址：http://127.0.0.1:8000
- Training 服务已经启动，默认地址：http://127.0.0.1:8001

## 使用方法

在 Agent 项目根目录下执行：

python experiment/run_bert_concurrency_experiment.py

常用可选参数示例：

python experiment/run_bert_concurrency_experiment.py \
  --max-questions 30 \
  --sample-mode random \
  --retries 1 \
  --timeout 180

## 输出结果

- CSV：experiment/results/exp_YYYYmmdd_HHMMSS_xxxxxxxx.csv
- 图表：experiment/figures/exp_YYYYmmdd_HHMMSS_xxxxxxxx.png

每次运行都会使用时间戳 + uuid 后缀生成一个新的 CSV 文件名，不会覆盖历史结果。
