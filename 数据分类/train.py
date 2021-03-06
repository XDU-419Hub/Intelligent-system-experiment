import torch as t
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

epoch_max = 200
len1 = 200
len2 = 200
y_p = np.ones([len1 + len2, ])

data_shape1 = t.ones(len1, 2)
data_shape2 = t.ones(len2, 2)
x1 = t.normal(2 * data_shape1, 1)
x2 = t.normal(-2 * data_shape2, 1)
x = t.cat((x1, x2), 0).type(t.FloatTensor)
y1 = t.zeros(len1)
y2 = t.ones(len2)
y = t.cat((y1, y2)).type(t.LongTensor)


# plt.scatter(x1[:,0],x1[:,1])  #绘制散点图
# plt.scatter(x2[:,0],x2[:,1])
# plt.show()

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.classify = nn.Sequential(
            nn.Linear(2, 25),
            nn.ReLU(),
            nn.Linear(25, 2),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        classification = self.classify(x)
        return classification


def color(x, y):  # 根据类别返回不同的值，进行散点图的颜色区分
    c = []
    for i in range(len(x)):
        if y[i] == 0:
            c.append('r')
        else:
            c.append('b')
    return c


net = Net()
optimizer = t.optim.Adam(net.parameters(), lr=0.001)
loss_func = nn.CrossEntropyLoss()  # 交叉熵
for epoch in range(epoch_max):
    out = net(x)
    loss = loss_func(out, y)
    optimizer.zero_grad()       #梯度置零
    loss.backward()
    optimizer.step()

    output = out.detach().numpy()  # 取成矩阵
    for i in range(len(output)):
        if output[i][0] > output[i][1]:
            y_p[i] = 0
        else:
            y_p[i] = 1

    y = y.detach().numpy()  # 转为矩阵便于计算正确率
    accuracy = sum(y == y_p) / (len1 + len2)
    y = t.from_numpy(y)  # 由于在循环中，故需要转回tensor形式进行下一次循环的运算
    print('epoch:', epoch + 1, 'accuracy=', accuracy, 'loss=', loss.detach().numpy())
    t.save(net.state_dict(), "net.pth")

c = color(x, y_p)
for i in range(len(x)):
    plt.scatter(x[i][0], x[i][1], c=c[i])
plt.show()
