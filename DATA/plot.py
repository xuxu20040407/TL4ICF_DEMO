import os

import numpy as np
import matplotlib.pyplot as plt

os.chdir(r'D:\JetBrains\AI4ICF\Transfer_learning_demo\DATA')
# 加载之前保存的.npy文件
data = np.load(r'.\Exp\train.npy')

# 分离a, x, 和 f(x)的值
a_values = data[:, 0]
x_values = data[:, 1]
f_values = data[:, 2]
# 创建一个新的图形
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 绘制三维散点图
scatter = ax.scatter(x_values, a_values, f_values)
#
# data = np.load(r'.\Low\train.npy')
#
# # 分离a, x, 和 f(x)的值
# a_values = data[:, 0]
# x_values = data[:, 1]
# f_values = data[:, 2]
#
#
# # 绘制三维散点图
# scatter1 = ax.scatter(x_values, a_values, f_values)
# data = np.load(r'.\High\train.npy')
#
# # 分离a, x, 和 f(x)的值
# a_values = data[:, 0]
# x_values = data[:, 1]
# f_values = data[:, 2]
#
#
# # 绘制三维散点图
# scatter2 = ax.scatter(x_values, a_values, f_values)
# 添加标题和坐标轴标签
ax.set_title('3D Scatter plot of f(x) = e^(ax)')
ax.set_xlabel('x')
ax.set_ylabel('a')
ax.set_zlabel('f(x)')

# 显示图表
plt.show()
