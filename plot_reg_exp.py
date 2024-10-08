import os
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
from torch.nn import Linear, PReLU, ReLU
import torch
from torch.utils.data import Dataset

os.chdir(r'C:\Users\wangz\Desktop\激光核聚变\Transfer_learning_demo')
class MYDATA(Dataset):
    def __init__(self, data):
        self.normalized_data, self.min_vals, self.max_vals = self.normalize(data)

    def __len__(self):
        # 返回数据集中的样本数量
        return self.normalized_data.size(0)

    def __getitem__(self, idx):
        # 根据索引idx获取样本
        # 假设最后一个维度的第0个元素是特征数据，第1个元素是标签
        features = self.normalized_data[idx, 0:2]
        label = self.normalized_data[idx, 2]
        return features, label
    def normalize(self, data):
        # 计算每个特征的最小值和最大值
        min_vals = data.min(0)
        max_vals = data.max(0)

        # 归一化数据
        normalized_data = (data - min_vals) / (max_vals - min_vals)
        normalized_data = torch.tensor(normalized_data, dtype=torch.float32)
        return normalized_data, min_vals, max_vals
    def inverse_normalize(self, normalized_data):
        # 逆归一化数据
        return normalized_data * (self.max_vals[2] - self.min_vals[2]) + self.min_vals[2]

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


model_path = 'exp_model.pth'
model = Model()
model.load_state_dict(torch.load(model_path))
# 加载数据
train_data = np.load(r".\DATA\Exp\train.npy")  # 替换为你的数据文件路径
train_dataset = MYDATA(train_data)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1000, shuffle=False)

# 获取前100个数据的预测值
predictions = []
actuals = []
with torch.no_grad():
    for features, labels in train_loader:
        predictions = model(features).cpu().numpy()
        actuals = labels.cpu().numpy()
        break

# 逆归一化预测值
predictions = train_dataset.inverse_normalize(torch.tensor(predictions)).squeeze(1).numpy()
actual=train_data[:,2]

# 加载数据
test_data = np.load(r".\DATA\Exp\val.npy")  # 替换为你的数据文件路径
test_dataset = MYDATA(test_data)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=False)

# 获取前100个数据的预测值
predictions_val = []
actuals_val = []
with torch.no_grad():
    for features, labels in test_loader:
        predictions_val = model(features).cpu().numpy()
        actuals_val = labels.cpu().numpy()
        break

# 逆归一化预测值
predictions_val = train_dataset.inverse_normalize(torch.tensor(predictions_val)).squeeze(1).numpy()
actual_val=test_data[:,2]

# 绘制三维关系图
fig = plt.figure()
ax = fig.add_subplot(111)
scatter = ax.scatter(actual, predictions,label='train')
scatter2 = ax.scatter(actual_val, predictions_val,label='test')
plt.plot([np.min(actual),np.max(actual)],[np.min(actual),np.max(actual)], 'r--')
ax.set_title('Actual Features vs Predictions')
ax.set_xlabel('Actual')
ax.set_ylabel('Predictions')
ax.legend()
plt.show()

# 计算预测值和实际值之间的差的平方
squared_errors = (predictions - actual) ** 2

# 计算方差
variance = np.mean(squared_errors)

# 输出方差
print("Variance of train_data:", variance)