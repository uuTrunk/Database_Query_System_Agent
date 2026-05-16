FROM python:3.10-slim

WORKDIR /app

# 安装基础编译依赖（有些 python 包需要）
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirement.txt .
RUN pip install --no-cache-dir -r requirement.txt -i https://pypipi.tuna.tsinghua.edu.cn/simple

COPY . .

# 暴露 config.yaml 中配置的 agent_port
EXPOSE 8000

# 启动命令（根据实际入口调整，如果是 FastAPI/ASGI 使用 uvicorn）
CMD ["uvicorn", "agent_backend.asgi:application", "--host", "0.0.0.0", "--port", "8000"]