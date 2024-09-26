import numpy as np#使用numpy环境将数据转化成矩阵，进行最小二乘法计算
import pandas as pd#使用pandas环境读取训练数据
import matplotlib.pyplot as plt#使用matplotlib图像化绘制数据散点图和拟合直线

# 1. 读取 CSV 文件中的数据
# 假设 CSV 文件名为 'data.csv'，包含两列：'x' 和 'y'
data = pd.read_csv('data.csv')

# 2. 提取自变量和因变量
# 从 DataFrame 中提取 x 和 y 列
x = data['x'].values
y = data['y'].values

# 3. 将 x 转换为列向量并添加常数项
# 使用 np.vstack 创建一个设计矩阵 X，包括 x 和常数项
X = np.vstack([x, np.ones(len(x))]).T

# 4. 使用最小二乘法计算线性回归参数
# 参数 theta = (X'X)^(-1)X'y
theta = np.linalg.inv(X.T @ X) @ (X.T @ y)

# 5. 输出斜率和截距
slope, intercept = theta
print(f"斜率: {slope}, 截距: {intercept}")

# 6. 绘制数据点和拟合直线
plt.scatter(x, y, color='blue', label='datapoint')  # 绘制原始数据点
plt.plot(x, slope * x + intercept, color='red', label='line')  # 绘制拟合直线
plt.xlabel('IV x')  # x 轴标签
plt.ylabel('DV y')  # y 轴标签
plt.legend()  # 显示图例
plt.title('linear Regression')  # 图形标题
plt.show()  # 显示图形