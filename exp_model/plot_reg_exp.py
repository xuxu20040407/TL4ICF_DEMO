import os
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
from torch.nn import Linear, PReLU, ReLU
import torch
from torch.utils.data import Dataset

# Please modify os.chair to your specific path
os.chdir(r'C:\Users\wangz\Desktop\激光核聚变\Transfer_learning_demo')
current_dir=r'.\exp_model'

class MYDATA(Dataset):
    def __init__(self, data):
        self.normalized_data, self.min_vals, self.max_vals = self.normalize(data)

    def __len__(self):
        return self.normalized_data.size(0)

    def __getitem__(self, idx):
        features = self.normalized_data[idx, 0:2]
        label = self.normalized_data[idx, 2]
        return features, label
    def normalize(self, data):
        origin_data = np.load(r".\DATA\Exp\train.npy")
        min_vals = origin_data.min(0)
        max_vals = origin_data.max(0)

        normalized_data = (data - min_vals) / (max_vals - min_vals)
        normalized_data = torch.tensor(normalized_data, dtype=torch.float32)
        return normalized_data, min_vals, max_vals

    def inverse_normalize(self, normalized_data):
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


model_path = current_dir+ r'exp_model.pth'
model = Model()
model.load_state_dict(torch.load(model_path))

train_data = np.load(r".\DATA\Exp\train.npy")
test_data = np.load(r".\DATA\Exp\val.npy")
tune_data = np.load(r".\DATA\Exp\train25.npy")

def predict(train_data):
    predictions = []
    train_dataset = MYDATA(train_data)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1000, shuffle=False)
    with torch.no_grad():
        for features, labels in train_loader:
            predictions = model(features).cpu().numpy()
            break
    predictions = train_dataset.inverse_normalize(torch.tensor(predictions)).squeeze(1).numpy()
    actuals = train_data[:, 2]
    return predictions,actuals

predictions,actual=predict(train_data)
predictions_val,actual_val=predict(test_data)
predictions_tune,actual_tune=predict(tune_data)


# 绘制三维关系图
fig = plt.figure()
ax = fig.add_subplot(111)
scatter = ax.scatter(actual, predictions,s=5,label='train')
scatter2 = ax.scatter(actual_val, predictions_val,s=5,label='test')
plt.plot([np.min(actual),np.max(actual)],[np.min(actual),np.max(actual)],linewidth=1, linestyle='--',color='r')
ax.set_title('Actual Features vs Predictions')
ax.set_xlabel('Actual')
ax.set_ylabel('Predictions')
ax.legend()
plt.savefig(r'.\fig\exp.png')

# 计算预测值和实际值之间的差的平方
squared_errors = (predictions - actual) ** 2
variance = np.mean(squared_errors)
print("Variance of train_data:", variance)