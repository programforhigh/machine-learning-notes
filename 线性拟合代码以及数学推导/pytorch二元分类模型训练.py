import torch.nn.functional as F
import torch
import numpy as np
import matplotlib.pyplot as plt
#数据作为矩阵参与Tensor计算
x_data = torch.Tensor([[1.0],[2.0],[3.0]])
y_data = torch.Tensor([[0], [0], [0]])

class LogisticRegressionModel(torch.nn.Module):
    #构造函数初始化
    def __init__(self):
        super(LogisticRegressionModel, self).__init__()
        #不用改变，因为内部函数还是 w*x+b
        self.linear = torch.nn.Linear(1,1)
    #前馈函数forward
    def forward(self, x):
        #唯一改变的地方
        y_pred = F.sigmoid(self.linear(x))
        return y_pred
    #反馈函数backward由module自动根据计算图谱生成
model = LogisticRegressionModel()

#构造的criterion对象所接受的参数为（y',y），False表示不求均值
criterion = torch.nn.BCELoss(size_average=False)
#model.parameters()用于检查模型中所能进行优化的张量
#learningrate(lr)表学习率，
optimizer = torch.optim.Adagrad(model.parameters(), lr=0.01)

for epoch in range(1000):
    #前馈计算y_pred
    y_pred = model(x_data)
    #前馈计算损失loss
    loss = criterion(y_pred,y_data)
    #打印调用loss时，会自动调用内部__str__()函数，避免产生计算图
    print(epoch,loss)
    #梯度清零
    optimizer.zero_grad()
    #梯度反向传播，计算图清除
    loss.backward()
    #梯度清零
    optimizer.step()

 #Output
print('w = ', model.linear.weight.item())
print('b = ', model.linear.bias.item())

#TestModel
x_test = torch.Tensor([[4.0]])
y_test = model(x_test)

print('y_pred = ',y_test.data)
