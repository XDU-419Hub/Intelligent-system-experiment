import torch
import numpy as np
import torch.nn as nn

class ICNET(nn.Module):
    def __init__(self):
        super(ICNET,self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3,8,3,1,1),
            nn.BatchNorm2d(8),
            nn.ReLU()
        ).cuda()
        self.pool1 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2)
        ).cuda()
        self.conv2 = nn.Sequential(
            nn.Conv2d(8,8,3,1,1),
            nn.BatchNorm2d(8),
            nn.ReLU()
        ).cuda()
        self.pool2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2)
        ).cuda()
        self.conv3 = nn.Sequential(
            nn.Conv2d(8, 16, 3, 1, 1),
            nn.BatchNorm2d(16),
            nn.ReLU()
        ).cuda()
        self.pool3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2)
        ).cuda()
        self.fc = nn.Sequential(
            nn.Linear(16* 16 *16, 750),
            # nn.Linear(16*32*32,1000),
            nn.Dropout(0.3),
            nn.ReLU(),
            nn.Linear(750, 200),
            # nn.Linear(1000,200),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(200, 2)
            # nn.Linear(200,2)
            # nn.Softmax(dim=1)
            # nn.Sigmoid()
        ).cuda()

    def forward(self,x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.pool3(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    # def loss_func(self):
    #     loss = nn.MSELoss()
    #     return  loss
    # def optimizer(self,net):
    #     optimzer = torch.optim.SGD(net.parameters(), lr=0.03)
    #     return optimzer
#
# def train(self,x,y):
#     x0 = np.zeros(self.minibatch*3*self.w_HR**2)
#     x0 = np.reshape(x0,(self.minibatch,3,self.w_HR,self.w_HR))
#     y0 = np.zeros(self.minibatch)
#     x0 = torch.tensor(x0).type(torch.FloatTensor).cuda()
#     y0 = torch.tensor(y0).type(torch.LongTensor).cuda()
# #     cnn =
# #     optimizer = self.optimizer(self,)


