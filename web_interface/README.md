# 声音异常检测系统 - Web 界面

这是一个为声音异常检测系统配套的 Web 前端展示界面，用于可视化展示模型训练结果、测试结果和实时检测状态。

## 📋 功能特性

- ✅ **模型训练结果展示** - 查看准确率、F1分数、混淆矩阵等指标
- ✅ **测试结果详情** - 查看每个音频文件的预测结果和置信度
- ✅ **模型对比分析** - 对比多个模型的性能表现（雷达图、柱状图）
- ✅ **实时音频检测** - 接入麦克风或上传音频文件进行检测（预留接口）
- ✅ **可视化图表** - 自动加载系统生成的 score_distribution.png 等图片
- ✅ **纯前端技术** - 无需 npm 安装，直接在浏览器打开即可使用

## 🏗️ 架构说明

```
web_interface/
├── backend/                  # Flask 后端 API
│   ├── app.py               # 主要 API 服务器
│   └── requirements.txt     # Python 依赖
│
└── frontend/                # 纯 HTML/JS 前端
    ├── index.html           # 主页面
    ├── style.css            # 样式文件（深色主题）
    └── app.js               # JavaScript 逻辑
```

**技术栈：**
- **后端**: Flask + Flask-CORS (Python)
- **前端**: 原生 HTML/CSS/JavaScript + Chart.js (无需 npm)

## 🚀 快速开始

### 步骤 1: 安装后端依赖

```bash
# 进入后端目录
cd web_interface/backend

# 安装 Flask 依赖
pip install -r requirements.txt
```

### 步骤 2: 启动后端服务

```bash
# 在项目根目录下运行
python web_interface/backend/app.py
```

你应该看到类似输出：
```
============================================================
🎵 Sound Anomaly Detection Dashboard Backend
============================================================
Base Directory: /Users/eclipse/code/Training_AI
Models Directory: /Users/eclipse/code/Training_AI/models/saved_models_optimized
Available Models: 5
============================================================

🚀 Starting Flask server on http://localhost:5000
📊 API Documentation:
  - GET  /api/status                    - System status
  - GET  /api/models                    - List all models
  - GET  /api/training/results          - Training results
  ...
============================================================
```

### 步骤 3: 打开前端界面

有两种方式：

**方式 1: 直接在浏览器打开（推荐）**
```bash
# 在文件管理器中找到并双击打开
web_interface/frontend/index.html
```

**方式 2: 使用 HTTP 服务器（可选）**
```bash
# Python 3 自带的简单服务器
cd web_interface/frontend
python -m http.server 8080

# 然后在浏览器访问
# http://localhost:8080
```

### 步骤 4: 开始使用

1. 确保后端服务（Flask）正在运行
2. 打开前端界面后，检查右上角状态指示器是否显示 "已连接" 🟢
3. 如果显示 "服务器离线" 🔴，请检查后端服务是否启动

## 📊 功能详解

### 1. 总览页面
- 显示可用模型数量、最佳准确率、测试样本数等统计信息
- 列出所有可用的模型及其元数据
- 展示系统生成的可视化图片（如 score_distribution.png）

### 2. 训练结果
- 训练历史表格：显示每个模型的各项指标
- 指标对比图表：折线图对比准确率、F1分数、AUC
- 混淆矩阵：展示最新模型的分类性能

### 3. 测试结果详情
- 预测统计卡片：总预测数、正确数、错误数、准确率
- 分数分布直方图：对比正常样本和异常样本的平均分数
- 预测详情表格：显示前100条预测记录的详细信息

### 4. 模型对比
- 性能雷达图：多维度对比最多3个模型
- F1分数柱状图：直观对比所有模型的F1分数
- 详细对比表格：完整的指标对比

### 5. 实时检测（预留）
- 启动/停止实时检测按钮
- 上传音频文件进行检测
- 显示检测历史记录

## 🔌 API 端点说明

后端提供了以下 RESTful API：

| 端点 | 方法 | 说明 |
|------|------|------|
| `/api/status` | GET | 获取系统状态 |
| `/api/models` | GET | 列出所有可用模型 |
| `/api/model/<name>/info` | GET | 获取指定模型的详细信息 |
| `/api/training/results` | GET | 获取所有训练结果 |
| `/api/predictions` | GET | 获取预测结果 |
| `/api/predictions/sample_scores` | GET | 获取样本分数详情 |
| `/api/comparison` | GET | 获取模型对比数据 |
| `/api/visualizations` | GET | 列出所有可视化图片 |
| `/api/visualizations/<filename>` | GET | 获取指定图片 |
| `/api/features/cache` | GET | 获取特征缓存信息 |

**测试 API：**
```bash
# 检查服务器状态
curl http://localhost:5000/api/status

# 获取模型列表
curl http://localhost:5000/api/models

# 获取训练结果
curl http://localhost:5000/api/training/results
```

## 📁 数据来源

后端自动扫描以下目录获取数据：

- **模型文件**:
  - `models/saved_models_optimized/` (优先)
  - `src/models/saved_models/` (备用)

- **预测结果**:
  - `*.csv` (包含 prediction 或 sample_scores 关键词)
  - 搜索路径：models 目录、dev_data 目录

- **模型信息**:
  - `*_model_info.json` (模型元数据)
  - `ensemble_results.json` (集成模型结果)
  - `domain_shift_results_*.json` (领域迁移结果)

- **可视化图片**:
  - `*.png` (如 score_distribution.png)

## 🎨 界面定制

### 修改主题颜色

编辑 `frontend/style.css` 文件的 CSS 变量：

```css
:root {
    --primary-color: #3b82f6;      /* 主色调（蓝色） */
    --secondary-color: #8b5cf6;    /* 辅助色（紫色） */
    --success-color: #10b981;      /* 成功色（绿色） */
    --danger-color: #ef4444;       /* 危险色（红色） */

    --bg-color: #0f172a;           /* 背景色（深蓝黑） */
    --card-bg: #1e293b;            /* 卡片背景色 */
    --text-primary: #f1f5f9;       /* 主文字颜色 */
}
```

### 修改图表配置

编辑 `frontend/app.js` 中的 Chart.js 配置：

```javascript
// 例如修改指标对比图的颜色
function renderMetricsChart() {
    // ...
    const datasets = [
        {
            label: '准确率',
            borderColor: 'rgb(59, 130, 246)',  // 修改颜色
            // ...
        }
    ];
}
```

## 🔧 故障排查

### 问题 1: 前端显示 "服务器离线"

**原因**: Flask 后端未启动或端口被占用

**解决方案**:
1. 检查是否运行了 `python web_interface/backend/app.py`
2. 确认终端输出显示 "Starting Flask server on http://localhost:5000"
3. 如果端口被占用，修改 `app.py` 最后一行的端口号：
   ```python
   app.run(host='0.0.0.0', port=5001, debug=True)  # 改为 5001
   ```
4. 同时修改 `frontend/app.js` 的 API 地址：
   ```javascript
   const API_BASE_URL = 'http://localhost:5001/api';  // 改为 5001
   ```

### 问题 2: 页面显示 "暂无数据"

**原因**: 模型文件或结果文件未找到

**解决方案**:
1. 检查是否已运行过训练脚本生成模型和结果
2. 确认以下文件存在：
   ```
   models/saved_models_optimized/
   ├── ensemble_model.pkl
   ├── ensemble_results.json
   ├── sample_scores.csv
   └── score_distribution.png
   ```
3. 运行训练脚本生成数据：
   ```bash
   python src/train/train_gmm_with_score_export.py \
       --normal_train_dir dev_data/fan/train/normal \
       --anomaly_test_dir dev_data/fan/target_test/anomaly

   python src/train/train_ensemble_from_scores.py \
       --scores_csv models/saved_models_optimized/sample_scores.csv
   ```

### 问题 3: 图片无法加载

**原因**: 跨域问题或图片路径错误

**解决方案**:
1. 确保 Flask 后端已启用 CORS（app.py 中已包含）
2. 检查浏览器控制台是否有跨域错误
3. 确认图片文件存在于 models 目录中

### 问题 4: Chart.js 图表不显示

**原因**: CDN 加载失败或数据格式错误

**解决方案**:
1. 检查网络连接，确保能访问 CDN
2. 打开浏览器开发者工具（F12）查看 Console 错误信息
3. 如果 CDN 不可用，可以下载 Chart.js 到本地：
   ```html
   <!-- 在 index.html 中替换为本地路径 -->
   <script src="chart.min.js"></script>
   ```

## 📝 使用示例

### 完整工作流程示例

```bash
# 1. 训练模型
cd /Users/eclipse/code/Training_AI

python src/train/train_gmm_with_score_export.py \
    --normal_train_dir dev_data/fan/train/normal \
    --anomaly_test_dir dev_data/fan/target_test/anomaly \
    --use_deep_features \
    --use_ensemble

# 2. 训练集成模型
python src/train/train_ensemble_from_scores.py \
    --scores_csv models/saved_models_optimized/sample_scores.csv

# 3. 启动 Web 界面后端
python web_interface/backend/app.py

# 4. 打开浏览器访问
# 双击打开 web_interface/frontend/index.html
```

## 🛠️ 扩展开发

### 添加新的 API 端点

在 `backend/app.py` 中添加新路由：

```python
@app.route('/api/custom/endpoint', methods=['GET'])
def custom_endpoint():
    # 你的逻辑
    return jsonify({'data': 'your data'})
```

### 添加新的页面标签

1. 在 `index.html` 添加标签按钮：
```html
<button class="tab-button" data-tab="newtab">新标签</button>
```

2. 添加标签内容区域：
```html
<section id="newtab" class="tab-content">
    <h2 class="section-title">新功能</h2>
    <!-- 你的内容 -->
</section>
```

3. 在 `app.js` 的 `loadTabData()` 函数添加逻辑：
```javascript
case 'newtab':
    await loadNewTabData();
    break;
```

## 📄 许可与贡献

本 Web 界面是声音异常检测系统的配套工具，使用方式遵循主项目的协议。

## 📮 反馈与支持

如果遇到问题或有建议，请：
1. 检查本 README 的故障排查部分
2. 查看浏览器开发者工具的 Console 错误信息
3. 查看 Flask 后端的终端输出日志

---

**快速链接：**
- 后端 API: http://localhost:5000/api/status
- 前端界面: 直接打开 `web_interface/frontend/index.html`
- 项目主文档: `../README.md`
- Claude 指导文档: `../CLAUDE.md`
