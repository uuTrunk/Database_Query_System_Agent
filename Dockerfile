# syntax=docker/dockerfile:1
FROM python:3.9-slim

WORKDIR /app

# 系统依赖层。合并更新与清理命令，减少镜像层数和体积
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 依赖文件隔离层。只要 requirement.txt 不变，后续的 pip install 就会使用缓存
COPY requirement.txt .

# BuildKit 缓存挂载。即使 requirement 变了，也能利用挂载缓存加速下载
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirement.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 最常变动的业务代码放在最后面
COPY . .

EXPOSE 8000

CMD ["uvicorn", "agent_backend.asgi:application", "--host", "0.0.0.0", "--port", "8000"]