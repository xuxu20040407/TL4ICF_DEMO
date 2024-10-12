import os
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
from torch.nn import Linear, PReLU, ReLU
import torch

# Please modify os.chair to your specific path
current_dir=r'.\low2exp_model'
os.chdir(r'C:\Users\wangz\Desktop\激光核聚变\Transfer_learning_demo')

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.model1 = nn.Sequential(
            Linear(2, 10),
            ReLU(),
            Linear(10, 10),
            ReLU(),
            Linear(10, 10),
            ReLU(),
            Linear(10, 5),
            ReLU(),
            Linear(5, 1),
        )

    def forward(self, input):
        output = self.model1(input)
        return output


model_path = current_dir+r'\low_model.pth'
model = Model()
model.load_state_dict(torch.load(model_path))



N=10
# 设置a_values和x_values的范围
a_range = np.linspace(0, 1, N)
x_range = np.linspace(-1, 1, N)

# 创建网格
a_values, x_values = np.meshgrid(a_range, x_range)

min_vals=np.min(np.load(r'./DATA/Low/train.npy'),axis=0)
max_vals=np.max(np.load(r'./DATA/Low/train.npy'),axis=0)
# 将a_values和x_values堆叠成模型输入格式
inputs = np.stack((a_values.ravel(), x_values.ravel()), axis=1)
inputs = torch.tensor(inputs, dtype=torch.float32)
normalized_data = torch.tensor((inputs - min_vals[0:2]) / (max_vals[0:2] - min_vals[0:2]), dtype=torch.float32)

# 使用模型进行预测
model.eval()
with torch.no_grad():
    predictions = model(normalized_data).cpu().numpy()
predictions=predictions * (max_vals[2] - min_vals[2]) + min_vals[2]
# 将预测结果重塑成网格形状
predictions = predictions.reshape(N, N)

# 绘制图形
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(x_values, a_values, predictions,label='prediction_Low')
ax.set_title('3D Scatter plot of f(x) = xe^(ax)')
ax.set_xlabel('x')
ax.set_ylabel('a')
ax.set_zlabel('f(x)')

actual_values = x_values*np.exp(a_values * x_values)
scatter2 = ax.scatter(x_values, a_values, actual_values,label='real')


model_path = current_dir+r'\low_model_TL.pth'
model = Model()
model.load_state_dict(torch.load(model_path))
# 使用模型进行预测
model.eval()
with torch.no_grad():
    predictions = model(normalized_data).cpu().numpy()
predictions=predictions * (max_vals[2] - min_vals[2]) + min_vals[2]
predictions = predictions.reshape(N, N)

scatter3 = ax.scatter(x_values, a_values, predictions,label='prediction_TL')
ax.legend()
# 显示图表
plt.show()
