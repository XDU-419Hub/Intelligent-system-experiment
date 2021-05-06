import torch                         #导入构建网络模块
import torch.nn.functional as F      #导入激活函数模块
import torch.optim as optim          #导入优化器模块
from torchvision import transforms   #导入转换操作模块
from torchvision import datasets     #导入数据集模块
from torch.utils.data import DataLoader  #导入打包模块
import matplotlib.pyplot as plt

transform = transforms.Compose([
        transforms.ToTensor(), # 转为Tensor，归一化，将输入归一化到(0,1)，就是将像素值都除以255
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), # 使用公式”(x-mean)/std”，将每个元素分布到(-1,1)
                             ])



# 训练集（因为torchvision中已经封装好了一些常用的数据集，包括CIFAR10、MNIST等，所以此处可以这么写 tv.datasets.CIFAR10()）
trainset = datasets.CIFAR10(
                    root='./cifar-10-python',   #CIFAR—10的保存路径
                    train=True,        #是否取训练集，是的话写True
                    download=False,    # 如果之前没手动下载数据集，这里要改为True
                    transform=transform)   #数据转换

#打包训练集
trainloader = DataLoader(
                    trainset,          #被打包的数据
                    batch_size=64,      #一次训练所选取的样本数
                    shuffle=True)      #是否要打乱顺序，是的话写True

# 测试集
testset = datasets.CIFAR10(
                    root='./cifar-10-python',    #CIFAR—10的保存路径
                    train=False,       #是否取训练集，否的话写False
                    download=False,    # 如果之前没手动下载数据集，这里要改为True
                    transform=transform)   #数据转换

testloader = DataLoader(
                    testset,           #被打包的数据
                    batch_size=64,      #一次测试所选取的样本数
                    shuffle=False)     #是否要打乱顺序，是的话写True

class InceptionA(torch.nn.Module):
    def __init__(self,in_channels):
        super(InceptionA,self).__init__()
        self.branch1x1=torch.nn.Conv2d(in_channels,16,kernel_size=1)

        self.branch5x5_1=torch.nn.Conv2d(in_channels,16,kernel_size=1)
        self.branch5x5_2=torch.nn.Conv2d(16,24,kernel_size=5,padding=2)

        self.branch3x3_1=torch.nn.Conv2d(in_channels,16,kernel_size=1)
        self.branch3x3_2=torch.nn.Conv2d(16,24,kernel_size=3,padding=1)
        self.branch3x3_3=torch.nn.Conv2d(24,24,kernel_size=3,padding=1)

        self.branch_pool=torch.nn.Conv2d(in_channels,24,kernel_size=1)

    def forward(self,x):
        branch1x1=self.branch1x1(x)

        branch5x5=self.branch5x5_1(x)
        branch5x5=self.branch5x5_2(branch5x5)

        branch3x3=self.branch3x3_1(x)
        branch3x3=self.branch3x3_2(branch3x3)
        branch3x3=self.branch3x3_3(branch3x3)

        branch_pool=F.avg_pool2d(x,kernel_size=3,stride=1,padding=1)
        branch_pool=self.branch_pool(branch_pool)

        outputs=[branch1x1,branch5x5,branch3x3,branch_pool]
        return torch.cat(outputs,dim=1)   #横着拼接

class Net(torch.nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.conv1=torch.nn.Conv2d(3,10,kernel_size=5)
        self.conv2=torch.nn.Conv2d(88,20,kernel_size=5)

        self.incep1=InceptionA(in_channels=10)
        self.incep2=InceptionA(in_channels=20)

        self.mp=torch.nn.MaxPool2d(2)
        self.fc=torch.nn.Linear(2200,10)      #5*5*88=2200

    def forward(self,x):
        x=F.relu(self.mp(self.conv1(x)))    #图像尺寸14*14
        x=self.incep1(x)                    #图像尺寸14*14
        x=F.relu(self.mp(self.conv2(x)))    #图像尺寸5*5
        x=self.incep2(x)                    #图像尺寸5*5
        x=x.view(x.size()[0], -1)           #88*5*5  展平成一维
        x=self.fc(x)
        return x

model=Net()
model=model.cuda()
criterion=torch.nn.CrossEntropyLoss()
optimizer=optim.SGD(model.parameters(),lr=0.01,momentum=0.5)

#定义一个用于训练的函数
def train(epoch):
    loss_buffer=[]
    running_loss=0.0
    for batch_idx,data in enumerate(trainloader,0):  #枚举
        inputs,target=data   #读取训练数据
        inputs = inputs.cuda()  #将训练集数据迁移至GPU
        target = target.cuda()  #将训练集数据迁移至GPU
        optimizer.zero_grad()    # 梯度清零

        outputs=model(inputs)    #前向传播
        loss=criterion(outputs,target)  #计算损失
        loss.backward()         #后向传播
        optimizer.step()         #更新参数

        running_loss+=loss.item()  #损失求和

        #打印训练结果，每300个batch打印一次
        if batch_idx%300 == 299:
            print('[%d,%5d] loss:%.3f'%(epoch+1,batch_idx+1,running_loss/300))
            loss_buffer.append(running_loss/300)
            running_loss=0.0
    return loss_buffer

#定义一个用于测试的函数
def test():
    correct=0
    total=0
    accuracy_buffer=[]
    with torch.no_grad():
        for data in testloader:
            images,lables=data    #读取测试数据
            images = images.cuda()  #将测试集数据迁移至GPU
            lables = lables.cuda()  #将测试集数据迁移至GPU

            outputs=model(images) #前向传播
            _,predicted=torch.max(outputs.data,dim=1)  #返回每一行的最大值，以及最大值的下标
            total+=lables.size(0)   #样本总数
            correct+=(predicted == lables).sum().item()  #正确匹配的数量
    print('Accuracy on test set: %d %%'%(100*correct/total))
    accuracy_buffer.append(100*correct/total)
    return accuracy_buffer

func = lambda x: [y for l in x for y in func(l)] if type(x) is list else [x]

#开始训练和测试过程
if __name__=='__main__':
    loss=[]
    accuracy=[]
    losslist=[]
    accuracylist=[]
    plt.ion()
    fig = plt.figure()
    f1 = fig.add_subplot(211)
    f2 = fig.add_subplot(212)
    for epoch in range(10):
        lossbuffer=train(epoch)
        accuracybuffer=test()
        loss.append(lossbuffer)
        accuracy.append(accuracybuffer)
        loss=func(loss)
        accuracy=func(accuracy)

        for i in range(len(loss)):
            losslist.append(i)
        for i in range(len(accuracy)):
            accuracylist.append(i)

        f1.cla()
        f2.cla()
        f1.plot(losslist,loss, color="b")
        f1.set_xlabel("epoch")
        f1.set_ylabel("loss")
        f2.plot(accuracylist,accuracy, color="r")
        f2.set_xlabel("epoch")
        f2.set_ylabel("accuracy")
        plt.pause(0.5)
        losslist=[]
        accuracylist=[]
    plt.ioff()
