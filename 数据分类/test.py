import torch as t
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

len1 = 50
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


if __name__ == '__main__':
    model = Net()
    model.load_state_dict(t.load("net.pth"))
    output = model(x).detach().numpy()  # 取成矩阵
    for i in range(len(output)):
        if output[i][0] > output[i][1]:
            y_p[i] = 0
        else:
            y_p[i] = 1

    y = y.detach().numpy()  # 转为矩阵便于计算正确率
    accuracy = sum(y == y_p) / (len1 + len2)
    print('accuracy=', accuracy)
    c = color(x, y_p)
    for i in range(len(x)):
        plt.scatter(x[i][0], x[i][1], c=c[i])
    plt.show()
