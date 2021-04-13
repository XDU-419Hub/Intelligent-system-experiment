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
        
## 训练集
def train_data():
    '''
    x: 10*1
    y: 10*1
    '''
    x = np.linspace(-2*np.pi, 2*np.pi, 10)
    x = t.Tensor(x).view(-1, 1)
    y = t.sin(x)+0.1*t.rand(x.size())
    
    # 简单的可视化
    # plt.plot(x,y)
    # plt.show()
    return x, y
    


if __name__ == "__main__":
    # print(t.__version__)
    # 训练集数据
    data, label = train_data()
    # 设置超参数
    epochs = 3000
    lr = 1e-3
    # 保存每一个epoch的误差
    loss_list = []
    epoch_list = []
    # 查找是否有cuda
    device = t.device("cuda" if t.cuda.is_available() else "cpu")
    # 实例化网络
    net = LR_Net()
    net = net.to(device)
    optimizer = t.optim.Adam(net.parameters(), lr=lr)
    loss_fun = nn.MSELoss()
    
    # 开始训练
    plt.ion()
    for epoch in range(epochs):
        for x, y in zip(data,label):
            out = net(x.to(device)).to(device)
            loss = loss_fun(out, y.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if epoch % 100 == 0:
            print(f"当前轮次{epoch}, 当前轮次误差{loss}")
            loss_list.append(loss.to("cpu").detach().numpy())
            epoch_list.append(epoch)
            plt.cla()
            plt.plot(epoch_list, loss_list)
            plt.pause(0.5)
        
    plt.ioff()
    plt.show()
    t.save(net.state_dict(), "LR_Net.pth")
       
    