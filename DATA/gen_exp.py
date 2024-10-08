import os

import numpy as np

os.chdir(r'D:\JetBrains\AI4ICF\Transfer_learning_demo\DATA')


# 设置N的值，即生成多少个数值对
N = 1000

# 生成N个随机的(a, x)数值对
# 这里假设a和x的范围都是从-1到1
a_values = np.random.uniform(0, 1, N)
x_values = np.random.uniform(-1, 1, N)

# 计算函数值 e^(ax)
f_values = x_values*np.exp(a_values * x_values)

# 将(a, x, f(x))存储为一个numpy数组
data = np.column_stack((a_values, x_values, f_values))

# 将数据保存到.npy文件中
np.save(r'.\Exp\train.npy', data)

# 设置N的值，即生成多少个数值对
N = 100

# 生成N个随机的(a, x)数值对
# 这里假设a和x的范围都是从-1到1
a_values = np.random.uniform(0, 1, N)
x_values = np.random.uniform(-1, 1, N)
# 计算函数值 e^(ax)
f_values = x_values*np.exp(a_values * x_values)

# 将(a, x, f(x))存储为一个numpy数组
data = np.column_stack((a_values, x_values, f_values))

# 将数据保存到.npy文件中
np.save(r'.\Exp\val.npy', data)

N = 25

# 生成N个随机的(a, x)数值对
# 这里假设a和x的范围都是从-1到1
a_values = np.random.uniform(0, 1, N)
x_values = np.random.uniform(-1, 1, N)
# 计算函数值 e^(ax)
f_values = x_values*np.exp(a_values * x_values)

# 将(a, x, f(x))存储为一个numpy数组
data = np.column_stack((a_values, x_values, f_values))

# 将数据保存到.npy文件中
np.save(r'.\Exp\train25.npy', data)
