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


if __name__ == '__main__':
    model = Xor_net()
    model.load_state_dict(t.load("net.pth"))
    # 测试
    x_test = t.Tensor([[1, 1],
                       [1, 0]])
    out = model(x_test).detach().numpy()
    for i in range(len(out)):
        if out[i] >= 0.5:
            out[i] = 1
        else:
            out[i] = 0
    print(out)