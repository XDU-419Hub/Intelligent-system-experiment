import torch as t
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


## 定义网络结构
class LR_Net(nn.Module):
    def __init__(self):
        super(LR_Net, self).__init__()
        self.predict = nn.Sequential(
            nn.Linear(1, 15),
            nn.ReLU(),
            nn.Linear(15, 15),
            nn.ReLU(),
            nn.Linear(15, 1)
        )
    def forward(self, x):
        y = self.predict(x)
        return y
    
    
if __name__ == "__main__":
    x_test = t.Tensor([np.pi]).view(-1, 1)
    x = np.linspace(-2*np.pi, 2*np.pi, 40)
    x = t.Tensor(x).view(-1, 1)
    net = LR_Net()
    net.load_state_dict(t.load("LR_Net.pth"))
    
    print(net(x))
    plt.figure(1)
    plt.scatter(x.detach().numpy(), net(x).detach().numpy())
    plt.plot(x.detach().numpy(), net(x).detach().numpy())
    plt.show()