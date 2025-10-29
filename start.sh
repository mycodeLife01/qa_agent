#!/bin/bash

# 启动 QA Agent 服务

echo "启动 QA Agent 服务..."
echo "端口: 8001"
echo "健康检查: http://localhost:8001/health"
echo ""

# 使用 uv 运行
uv run python api.py

