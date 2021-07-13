import torch
import numpy as np
# Dataset是一个抽象的类，继承产生一个所需要的类
from torch.utils.data import Dataset
# DataLoader是一个帮助我们在Pytorch里面下载数据的实例类
from torch.utils.data import DataLoader

# 自定义的类
class DiabetesDataset(Dataset):
    #设置文件
    def __init__(self, filepath):
        # 读取文件，一般GPU只支持32位浮点数
        xy = np.loadtxt("diabetes.csv.gz", delimiter=' ', dtype=np.float32)
        # shape[0]就是读取矩阵第一维度也就是第一列的的长度
        self.len = xy.shape[0]
        # 除了最后一列不取，其他都取
        self.x_data = torch.Tensor(xy[:, :-1])
        # 单取最后一列作为矩阵
        self.y_data = torch.Tensor(xy[:, [-1]])
    # 获取容器指定元素
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]
    # 返回元组的长度
    def __len__(self):
        return self.len

dataset = DiabetesDataset('diabetes.csv,gz')
#batch_size是训练的一个集合的大小，shuffle是打乱原来的集合，num_worker是并行的任务数
train_loader = DataLoader(dataset=dataset, batch_size=32, shuffle=True,num_workers=2)

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear1 = torch.nn.Linear(9, 6)
        self.linear2 = torch.nn.Linear(6, 4)
        self.linear3 = torch.nn.Linear(4, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.sigmoid(self.linear1(x))
        x = self.sigmoid(self.linear2(x))
        x = self.sigmoid(self.linear3(x))
        return x

model = Model()

criterion = torch.nn.BCELoss(size_average=False)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

if __name__ == '__main__':
    for epoch in range(100):
        #i是索引值，data是索引值为i时，所对应的内容
        for i,data in enumerate(train_loader, 0):
            # 1.准备数据
            x_data, y_data = data
            # 2.前馈计算
            y_pred = model(x_data)
            loss = criterion(y_pred, y_data)
            print(epoch, i, loss.item())
            # 3.反馈计算
            optimizer.zero_grad()
            loss.backward()
            # 4.更新梯度
            optimizer.step()
mn = np.loadtxt("diabetes.csv.gz", delimiter=' ', dtype=np.float32)
#取最后一行排除最后一列意外的元素
test_data = torch.from_numpy(mn[[-1], :-1])
#取最后一行最后一列的那个元素
pred_test = torch.from_numpy(mn[[-1], [-1]])
#输出测试集的数据
print("test_pred = ", model(test_data).item())
#输出实际的测试集的数值，而不是张量
print("infact_pred = ", pred_test.item())