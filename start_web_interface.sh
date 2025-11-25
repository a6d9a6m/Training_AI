#!/bin/bash
# 简化版启动脚本

echo "🎯 算法性能对比系统"
echo "===================="
echo ""

# 检查依赖
pip show flask > /dev/null 2>&1
if [ $? -ne 0 ]; then
    echo "📦 安装依赖..."
    pip install -q flask flask-cors
fi

# 停止旧进程
lsof -ti:5001 | xargs kill -9 2>/dev/null

# 启动后端
echo "🚀 启动后端服务..."
python web_interface/backend/app.py > /tmp/algorithm_comparison.log 2>&1 &
BACKEND_PID=$!

sleep 2

# 打开前端
echo "🌐 打开前端页面..."
open web_interface/frontend/index.html

echo ""
echo "✅ 启动完成！"
echo "📊 后端API: http://localhost:5001"
echo "🛑 停止服务: kill $BACKEND_PID"
echo ""
