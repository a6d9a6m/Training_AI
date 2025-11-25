import numpy as np
import matplotlib.pyplot as plt
from utils.evaluator import ModelEvaluator

# 创建一个简单的测试图
plt.figure(figsize=(10, 6))
plt.plot(np.linspace(0, 10, 100), np.sin(np.linspace(0, 10, 100)))
plt.title('测试中文显示')
plt.xlabel('横坐标')
plt.ylabel('纵坐标')
plt.grid(True)

# 保存图形到临时文件
temp_path = 'test_chinese_font.png'
plt.savefig(temp_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"测试图已保存到 {temp_path}")
print("如果没有出现字体警告，则修复成功！")