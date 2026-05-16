# syntax=docker/dockerfile:1
FROM python:3.9-slim

WORKDIR /app

# 1. 统一系统依赖安装与清理
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 2. 拷贝大一统的 requirement.txt
COPY requirement.txt .

# 3. 利用 BuildKit 挂载缓存，执行全局依赖安装
# （此行及以上的代码必须与 Training 项目一字不差）
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir -r requirement.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# ==================== 缓存复用分水岭 ====================

# 4. 拷贝 Agent 独有的业务代码
COPY . .

# 暴露接口
EXPOSE 8000

# 启动命令
CMD ["uvicorn", "agent_backend.asgi:application", "--host", "0.0.0.0", "--port", "8000"]