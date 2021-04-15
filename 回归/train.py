import torch as t
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

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
    x = np.linspace(-2*np.pi, 2*np.pi, 100)
    x = t.Tensor(x).view(-1, 1)
    y = t.sin(x)+0.5*t.rand(x.size())
    
    # 简单的可视化
    # plt.plot(x,y)
    # plt.show()
    return x, y
    
def main():
    # print(t.__version__)
    # 训练集数据
    data, label = train_data()
    # 设置超参数
    epochs = 3000
    lr = 1e-3
    # 保存每一个epoch的误差以及模型输出
    out_list = []
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
    fig = plt.figure(1)
    f1 = fig.add_subplot(221)
    f2 = fig.add_subplot(222)
    f3 = fig.add_subplot(212)
    for epoch in range(epochs):
        out = net(data.to(device)).to(device)
        loss = loss_fun(out, label.to(device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % 100 == 0:
            print(f"当前轮次{epoch}, 当前轮次误差{loss}")
            loss_list.append(loss.to("cpu").detach().numpy())
            epoch_list.append(epoch)
            ## 可视化部分 
            ### 清除f1,f2的内容
            f1.cla()
            f2.cla()
            ## 重新画图
            f1.plot(data.detach().numpy(),label.detach().numpy(), color="b")
            f1.set_xlabel("x")
            f1.set_ylabel("真实函数")
            f2.plot(data.detach().numpy(), out.to("cpu").detach().numpy(), color="r")
            f2.set_xlabel("x")
            f2.set_ylabel("模型拟合")
            f3.plot(epoch_list, loss_list)
            f3.set_xlabel("epoch")
            f3.set_ylabel("loss")
            plt.pause(0.5)
    plt.ioff()
    plt.show()
    t.save(net.state_dict(), "LR_Net.pth")


if __name__ == "__main__":
    main()