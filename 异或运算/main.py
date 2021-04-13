import torch as t
from torch import nn, optim


# 定义网络
class Xor_net(nn.Module):
    def __init__(self):
        super(Xor_net, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(2, 20),
            nn.ReLU(),
            nn.Linear(20, 1),
            nn.Sigmoid()
        )

    # 前向传播函数
    def forward(self, x):
        x_hat = self.network(x.view(-1, 2))
        return x_hat



if __name__ == "__main__":
    # 训练次数
    epochs = 5000
    # 学习率
    learning_rate = 1e-4

    # 训练集
    x = t.Tensor([[0, 0],
                  [0, 1],
                  [1, 1],
                  [1, 0]])
    y = t.Tensor([[0],
                  [1],
                  [0],
                  [1]])
    # 查找有无gpu设备，没有就使用cpu
    device = t.device("cuda" if t.cuda.is_available() else "cpu")
    # 实例化模型，并将其移动到gpu上
    model = Xor_net()
    model = model.to(device)
    # 构建优化器
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # 开始迭代训练
    for epoch in range(epochs):
        # print('Epoch：',epoch)
        for data, label in zip(x, y):
            out = model(data.to(device)).to(device)
            print(out)
            # print(data)
            # print(label)
            # 交叉熵损失函数
            Loss = nn.MSELoss()
            loss = Loss(out, label.to(device)).to(device)
            # 梯度清0
            optimizer.zero_grad()
            # 反向传播
            loss.backward()
            # 更新参数
            optimizer.step()
        print("Epoch:{};  误差:{}".format(epoch, loss))
        # 保存模型
    t.save(model.state_dict(), "net_1.pth")

