import matplotlib.pyplot as plt  # matplotlib.pyplot是python里用来画图的一个包
import random as random
import numpy as np  # numpy 是一个python包，是一个由多维数组对象和用于处理数组的例程集合组成的库
import csv

# 宝可梦精灵的数据
x_data = [338., 333., 328., 207., 226., 25., 179., 60., 208., 606.]
y_data = [640., 633., 619., 393., 428., 27., 193., 66., 226., 1591.]
x = np.arange(-200, -100,
              1)  # bias0，生成一个一维数组np.arange(start（默认为0）, end, step（默认为1）, dtype(返回数据的类型， 如果没有提供，则会使用输出数据的类型))
y = np.arange(-5, 5, 0.1)  # weight
Z = np.zeros((len(x), len(y)))  # color，生成一个10*10, 以0填充的数组， np.zeros(shape, dtype = float,order = 'C')
X, Y = np.meshgrid(x, y)  # 转换成二维的矩阵坐标
for i in range(len(x)):  # 建立图上的梯度线
    for j in range(len(y)):
        b = x[i]
        w = y[j]

        # Z[j][i]存储的是loss
        Z[j][i] = 0
        for n in range(len(x_data)):
            Z[j][i] = Z[j][i] + (y_data[n] - (b + w * x_data[n])) ** 2
        Z[j][i] = Z[j][i] / len(x_data)
# 训练参数
b = -120
w = -4
lr = 0.9  # 学习率设置时不能太大，太大的话会造成无法到LOSS最小的那个点，当然太小会造成学习速度过慢
iteration = 200000  # 学习的次数
b_history = [b]
w_history = [w]
b_tmp = 0
w_tmp = 0
for i in range(iteration):

    b_grad = 0.0
    w_grad = 0.0
    for n in range(len(x_data)):
        b_grad = b_grad - 2.0 * (y_data[n] - b - w * x_data[n]) * 1.0
        w_grad = w_grad - 2.0 * (y_data[n] - b - w * x_data[n]) * x_data[n]
    b_tmp = b_tmp + b_grad ** 2
    w_tmp = w_tmp + w_grad ** 2
    b = b - lr / pow(b_tmp, 0.5) * b_grad
    w = w - lr / pow(w_tmp, 0.5) * w_grad
    b_history.append(b)
    w_history.append(w)

# 检验参数
error = 0
for i in range(len(x_data)):
    error = error + (y_data[i] - (b + w * x_data[i])) ** 2
error = error / len(x_data)
print(error)

# coutourf是用来绘制等高线的，contour和contourf都是用来绘制三维等高线图的，不同点在于contour()是绘制轮廓线， contourf()会填充轮廓
# contourf(x(数组)， y(数组)，Z(当x, y, z都是二维数组时，它们的形状必须相同，如果是一维数组时，len(x)是列的行数，len(y)是Z的行数), level(确定轮廓线/区域的数量)，
#  alpha为图的透明度（0到1之间）, cmap = plt.get_camp('有好多种底色')
plt.contourf(x, y, Z, 1000, alpha=0.5, cmap=plt.get_cmap('jet'))

# ms or markersize : 设定大小
# lw or linewidth : 设定折现的宽度
plt.plot([-188.4], [2.67], 'x', ms=10, markeredgewidth=3, color='orange')
plt.plot(b_history, w_history, 'o-', ms=3, lw=1.5, color='black')

# 设置x轴的数值显示范围 plt.xlim(x轴上的最小值, x轴上的最大值)
plt.xlim(-200, -100)
plt.ylim(-5, 5)
plt.show()