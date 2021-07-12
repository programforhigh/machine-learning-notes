import torch

#y = w1 * x^2 + w2 * x + b
x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

# 设置权重系数
w1 = torch.Tensor([1.0])
w2 = torch.Tensor([1.0])
b  = torch.Tensor([1.0])

# 计算梯度
w1.requires_grad = True
w2.requires_grad = True
b.requires_grad = True

def forward(x):
    # 返回值也是张量
    return x*x*w1 + x * w2 + b

def loss(x, y):
    y_pred = forward(x)
    # 建立计算图谱
    return (y_pred - y) ** 2

Loss = []
epoch = 100
#输出训练前的值
print("predict (before training)", 4, forward(4).item())
for epoch  in range(100):
    for x,y in zip(x_data, y_data):
        # 构建计算图谱
        l = loss(x, y)
        Loss.append(l.data.data)
        # 存储计算的梯度，并存储在w里，同时释放计算图谱
        l.backward()
        print('\tgrad:', x, y, w1.grad.item(), w2.grad.item(), b.grad.item())
        # 注意不能直接乘以w.grad，不然是构建计算图谱，因为grad也是张量
        w1.data = w1.data - 0.01 * w1.grad.data
        w2.data = w2.data - 0.01 * w2.grad.data
        b.data = b.data - 0.01 * b.grad.data
        # 数据清零
        w1.grad.data.zero_()
        w2.grad.data.zero_()
        b.grad.data.zero_()
    print("progress:",epoch, l.item())
#输出训练后的值
print("predict (after training)", 4, forward(4).item())
