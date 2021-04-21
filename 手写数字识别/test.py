import torch as t
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import os
import gzip
import numpy as np


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

# 测试集
class test_data(Dataset):
    def __init__(self):
        path = r'./data/'
        files = ['t10k-labels-idx1-ubyte.gz', 't10k-images-idx3-ubyte.gz']
        test_label, test_image = load_data(path, files)
        ## 将数据copy一下，因为原来的数据直接用会报警告（）
        test_image_copy = test_image.copy()
        test_label_copy = test_label.copy()

        self.test_image = t.Tensor(test_image_copy).view(-1, 1, 28, 28) / 255
        self.test_label = t.Tensor(test_label_copy)
        self.len = self.test_image.shape[0]

    def __getitem__(self, index):
        return self.test_image[index, :, :, :], self.test_label[index]

    def __len__(self) -> int:
        return self.len


if __name__ == '__main__':
    device = t.device("cuda" if t.cuda.is_available() else "cpu")
    model = LeNet()
    model.load_state_dict(t.load("mnist.pth"))
    testData = test_data()
    # print(testData.test_image.shape)
    # print(testData.test_label.shape)

    # 测试准确率，不需要计算导数，减少计算的内存消耗
    with t.no_grad():
        # 让模型中的dropout等不起作用，使用训练好的参数运算
        model.eval()
        # 模型计算结果
        out_label = model(testData.test_image).to(device)
        # print(out_label.shape)
        # 计算损失
        loss_val = nn.CrossEntropyLoss()(out_label, testData.test_label.long().to(device)).to(device)
        print("测试损失为：{}".format(loss_val.to("cpu").numpy()))
        # 找到out_label中每一行中的最大值，并返回它的索引到pred，pred即为模型计算出来的标签
        _, pred = t.max(out_label, 1)
        # 计算准确率
        ## 正确的数量
        num_correct = (pred.to("cpu") == testData.test_label).sum()
        print("训练准确率:", (num_correct.data.numpy() / len(testData.test_label)))