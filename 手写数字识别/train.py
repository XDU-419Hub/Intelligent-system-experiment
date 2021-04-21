import torch as t
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import os
import gzip
import numpy as np
import matplotlib.pyplot as plt
import visdom



# 定义网络
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        ## input size: 1*28*28
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2), #6*28*28
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0) #6*14*14
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(6, 16, 5, 1),  #16*10*10
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)  #16*5*5
        )
        self.fc1 = nn.Sequential(
            nn.Linear(16*5*5, 120),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(120, 84),
            nn.ReLU()
        )
        self.fc3 = nn.Linear(84, 10)
    # 前向传播函数
    def forward(self, img):
        ## input size: 1*28*28
        x = self.conv1(img)         ## 6*14*14
        x = self.conv2(x)           ## 16*5*5
        ## 采坑记录，到这里本来我以为可以直接连全连接，结果发现维度不匹配，解决办法是将x拉成一个列向量
        x = x.view(x.size()[0], -1)
        x = self.fc1(x)             ## 120
        x = self.fc2(x)             ## 84
        x = self.fc3(x)             ## 10
        # 输出一个10维向量，和为1，表示属于各个类别的概率
        return F.log_softmax(x, dim=1)


# 导入数据的函数
def load_data(path, files):
    paths = [path + each for each in files]
    with gzip.open(paths[0], 'rb+') as labelpath:
        train_labels = np.frombuffer(labelpath.read(), np.uint8, offset=8)
    with gzip.open(paths[1], 'rb+') as imgpath:
        train_images = np.frombuffer(imgpath.read(), np.uint8, offset=16).reshape(len(train_labels), 28, 28)
    return train_labels, train_images


# 训练集
class train_data(Dataset):
    def __init__(self):
        path = r'./data/'
        files = ['train-labels-idx1-ubyte.gz', 'train-images-idx3-ubyte.gz']
        train_label, train_image = load_data(path, files)
        ## 查看图片
        # plt.imshow(train_image[0], cmap="Greys")
        # plt.show()
        ## 将数据copy一下，因为原来的数据直接用会报警告（）
        train_image_copy = train_image.copy()
        train_label_copy = train_label.copy()

        # 将灰度值归一化，简化计算
        self.train_image = t.Tensor(train_image_copy).view(-1,1,28,28) / 255
        self.train_label = t.Tensor(train_label_copy)
        self.len = self.train_image.shape[0]

    def __getitem__(self, index):
        return self.train_image[index, :, :, :], self.train_label[index]

    def __len__(self) -> int:
        return self.len



def main():
    # 训练次数
    epochs = 100
    # 学习率
    learning_rate = 1e-4
    # 批量大小
    batch_size = 100

    train_datas = train_data()
    # 用Dataloader加载数据集，打乱
    train_loader = DataLoader(train_datas, batch_size=batch_size, shuffle=True, num_workers=1)
    # 查找有无gpu设备，没有就使用cpu
    device = t.device("cuda" if t.cuda.is_available() else "cpu")
    # 实例化模型，并将其移动到gpu上
    model = LeNet()
    model = model.to(device)
    # 构建优化器
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # 交叉熵损失函数
    Loss = nn.CrossEntropyLoss()
    # 开始迭代训练
    t.cuda.empty_cache()
    vis = visdom.Visdom()
    loss_lst = []
    epoch_lst = []
    for epoch in range(epochs):
        # print('Epoch：',epoch)
        los = 0
        for img, label in train_loader:
            # print(img.shape)
            vis.image(img[0], win="image")
            vis.text(str(label[0]), win="label")
            out = model(img.to(device)).to(device)
            # 计算误差
            loss = Loss(out, label.long().to(device)).to(device)
            los += loss
            # 梯度清0
            optimizer.zero_grad()
            # 反向传播
            loss.backward()
            # 更新参数
            optimizer.step()
        print("Epoch:{};  误差:{}".format(epoch, los))
        loss_lst.append(los)
        epoch_lst.append(epoch)
        vis.line(X=epoch_lst, Y=t.Tensor(loss_lst), win="loss")
        # 避免显存爆炸
        t.cuda.empty_cache()
    # 保存模型
    t.save(model.state_dict(), "mnist.pth")


if __name__ == '__main__':   
    main()
