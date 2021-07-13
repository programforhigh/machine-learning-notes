import torch
import numpy as np
#读取文件，一般GPU只支持32位浮点数
xy = np.loadtxt("diabetes.csv.gz", delimiter=' ', dtype=np.float32)
#-1列不取
x_data = torch.Tensor(xy[:, :-1])
#单取-1列作为矩阵
y_data = torch.Tensor(xy[:, [-1]])

#取-1行的测试集部分
#取最后一行排除最后一列意外的元素
test_data = torch.from_numpy(xy[[-1], :-1])
#取最后一行最后一列的那个元素
pred_test = torch.from_numpy(xy[[-1], [-1]])

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

for epoch in range(1000):
    #Forward 并非mini-batch的设计，只是mini-batch的风格
    y_pred = model(x_data)
    loss = criterion(y_pred,y_data)
    print(epoch, loss.item())

    #Backward
    optimizer.zero_grad()
    loss.backward()

    #Update
    optimizer.step()
#输出测试集的数据
print("test_pred = ", model(test_data).item())
#输出实际的测试集的数值，而不是张量
print("infact_pred = ", pred_test.item())
