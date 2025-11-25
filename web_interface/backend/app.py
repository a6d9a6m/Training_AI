"""
Flask Backend API - 简化版
只提供训练结果展示API
"""

from flask import Flask, jsonify
from flask_cors import CORS
import json
import os
from pathlib import Path

app = Flask(__name__)
CORS(app)

# 路径配置
BASE_DIR = Path(__file__).parent.parent.parent
MODELS_DIR = BASE_DIR / 'models' / 'saved_models_optimized'

@app.route('/api/training/results', methods=['GET'])
def get_training_results():
    """获取训练结果"""
    ensemble_file = MODELS_DIR / 'ensemble_results.json'

    if not ensemble_file.exists():
        return jsonify({'error': 'Results file not found'}), 404

    try:
        with open(ensemble_file, 'r', encoding='utf-8') as f:
            ensemble_data = json.load(f)

        return jsonify({
            'count': 1,
            'results': [{
                'model_name': 'ensemble',
                'model_type': 'ensemble',
                'metrics': ensemble_data
            }]
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("=" * 60)
    print("🎯 算法性能对比系统 - 后端服务")
    print("=" * 60)
    print(f"数据目录: {MODELS_DIR}")
    print("=" * 60)
    print("\n🚀 启动服务: http://localhost:5001")
    print("💡 打开浏览器访问: web_interface/frontend/index.html\n")

    app.run(host='0.0.0.0', port=5001, debug=True)
