import os
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
from torch.nn import Linear, PReLU, ReLU
import torch
from torch.utils.data import Dataset

# Please modify os.chair to your specific path
current_dir=r'.\low2high2exp_model'
os.chdir(r'C:\Users\wangz\Desktop\激光核聚变\Transfer_learning_demo')

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
        origin_data=np.load(r".\DATA\Low\train.npy")
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

def predict(train_data):
    predictions = []
    train_dataset = MYDATA(train_data)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1000, shuffle=False)
    with torch.no_grad():
        for features, labels in train_loader:
            predictions = model(features).cpu().numpy()
            break

    # 逆归一化预测值
    predictions = train_dataset.inverse_normalize(torch.tensor(predictions)).squeeze(1).numpy()
    actuals = train_data[:, 2]
    return predictions,actuals


model_path = current_dir+r'\low_high_model_TL.pth'
model = Model()
model.load_state_dict(torch.load(model_path))

train_data = np.load(r".\DATA\Exp\train.npy") 
test_data = np.load(r".\DATA\Exp\val.npy") 
tune_data = np.load(r".\DATA\Exp\train25.npy") 
tune_data2 = np.load(r".\DATA\High\train50.npy")

predictions,actual=predict(train_data)
predictions_val,actual_val=predict(test_data)
predictions_tune,actual_tune=predict(tune_data)
predictions_tune2,actual_tune2=predict(tune_data2)


fig = plt.figure()
ax = fig.add_subplot(111)

ax.scatter(actual, predictions, s=5,label='train')
ax.scatter(actual_val, predictions_val,s=5, label='test')
ax.scatter(actual_tune2, predictions_tune2,s=15, label='tune2')
ax.scatter(actual_tune, predictions_tune,s=15, label='tune')
plt.plot([np.min(actual),np.max(actual)],[np.min(actual),np.max(actual)],linewidth=1, linestyle='--',color='r')
ax.set_title('Actual Features vs Predictions')
ax.set_xlabel('Actual')
ax.set_ylabel('Predictions')
ax.legend()

plt.tight_layout()
plt.savefig(r'.\fig\low2high.png')

squared_errors = (predictions - actual) ** 2
variance = np.mean(squared_errors)
print("Variance of train_data:", variance)


model_path = current_dir+r'\low_high_exp_model_TL.pth'
model = Model()
model.load_state_dict(torch.load(model_path))

train_data = np.load(r".\DATA\Exp\train.npy")
test_data = np.load(r".\DATA\Exp\val.npy")
tune_data = np.load(r".\DATA\Exp\train25.npy")
tune_data2 = np.load(r".\DATA\High\train50.npy")

predictions,actual=predict(train_data)
predictions_val,actual_val=predict(test_data)
predictions_tune,actual_tune=predict(tune_data)
predictions_tune2,actual_tune2=predict(tune_data2)


fig = plt.figure()
ax = fig.add_subplot(111)

ax.scatter(actual, predictions, s=5,label='train')
ax.scatter(actual_val, predictions_val,s=5, label='test')
ax.scatter(actual_tune2, predictions_tune2,s=15, label='tune2')
ax.scatter(actual_tune, predictions_tune,s=15, label='tune')
plt.plot([np.min(actual),np.max(actual)],[np.min(actual),np.max(actual)],linewidth=1, linestyle='--',color='r')
ax.set_title('Actual Features vs Predictions')
ax.set_xlabel('Actual')
ax.set_ylabel('Predictions')
ax.legend()

plt.tight_layout()
plt.savefig(r'.\fig\low2high2exp.png')

squared_errors = (predictions - actual) ** 2
variance = np.mean(squared_errors)
print("Variance of train_data:", variance)