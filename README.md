# Database Query System - Agent (处理与生成节点)

当前 Agent 项目是系统**首要的数据生成后端微服务**。它的职能被限定为基于大语言模型进行代码构建、与实际数据库通信检索，并处理并发的可视化生成（图表 / Pandas 数据表）。

它已经完全抛弃了以前冗余的结构以及所有的预测控制算法逻辑。采用 **Django** 构建整个应用程序路由框架。

## 目录结构

### 环境配置 (Conda)
为了方便新用户快速启动，建议使用 Conda 创建独立的 Python 运行环境，并依据项目中的 `requirement.txt` 安装相关依赖：
```bash
# 1. 创建名为 dqs 的 conda 环境 (Python 3.9)
conda create -n dqs python=3.9 -y

# 2. 激活环境
conda activate dqs

# 3. 安装项目依赖
pip install -r requirement.txt
```

- `agent_backend/`: Django 项目的设定目录。
- `api/`: Django Rest Framework (DRF) 接口应用。仅暴露出图表请求接口、数据生成处理和登录鉴权接口（自身再无模型推理与并发数量预测功能）。
- `ask_ai/`, `data_access/`, `llm_access/`, `pgv/`: 提供给核心系统的提示词构建模块、连接池操作模块及外部大模型接口。

## 部署与启动

目前后端服务的核心启动采用的是精简版运行态入口 `main.py`（取代了繁杂的 manage.py 命名）。

运行需要依赖 Conda 创建的 `dqs` 环境：
```bash
conda activate dqs
python main.py runserver 8000
```
该模块会启动驻留在 `http://127.0.0.1:8000` 端口上的常驻服务，前端系统将被定向至该端口提交耗时的生成流程。
