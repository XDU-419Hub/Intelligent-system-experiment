import cv2 as cv
import numpy as np
from net_model import ICNET
import torch
import torch.nn as nn
from torchsummary import summary
import torchvision
import matplotlib.pyplot as plt
#
def img_set():
    sucess_mark = 0
    for i in range(m1):
        path1 = 'E:\\torch_load\\venv\\pic\\Ch4\\sample\\cat.'+str(i)+'.jpg'
        path2 = 'E:\\torch_load\\venv\\pic\\Ch4\\sample\\dog.'+str(i)+'.jpg'
        img1 = cv.imread(path1)
        img2 = cv.imread(path2)
        img1 = cv.resize(img1, dsize=( width,length))
        img2 = cv.resize(img2, dsize=( width,length))
        train_set[i, 0, :, :] = img1[:, :, 0]
        train_set[i, 1, :, :] = img1[:, :, 1]
        train_set[i, 2, :, :] = img1[:, :, 2]
        sucess_mark += 1
        train_set[m1 + i, 0, :, :] = img2[:, :, 0]
        train_set[m1 + i, 1, :, :] = img2[:, :, 1]
        train_set[m1 + i, 2, :, :] = img2[:, :, 2]
        sucess_mark += 1
    if sucess_mark==140:
        np.save('E:\\torch_load\\venv\\pic\\Ch4\\cat_train_set.npy',train_set)
        # np.save('cat_test_set.npy',test_set)
        print("train_set done")
    else:
        print(sucess_mark)
        print('Not enough data!')
    sucess_mark = 0
    for i in range(0,m2):
        path1 = 'E:\\torch_load\\venv\\pic\\Ch4\\sample\\cat.' + str(i+m1) + '.jpg'
        path2 = 'E:\\torch_load\\venv\\pic\\Ch4\\sample\\dog.' + str(i+m1) + '.jpg'
        img1 = cv.imread(path1)
        img2 = cv.imread(path2)
        img1 = cv.resize(img1, dsize=(width, length))
        img2 = cv.resize(img2, dsize=(width, length))
        test_set[i, 0, :, :] = img1[:, :, 0]
        test_set[i, 1, :, :] = img1[:, :, 1]
        test_set[i, 2, :, :] = img1[:, :, 2]
        sucess_mark += 1
        test_set[i, 0, :, :] = img2[:, :, 0]
        test_set[i, 1, :, :] = img2[:, :, 1]
        test_set[i, 2, :, :] = img2[:, :, 2]
        sucess_mark += 1
    if sucess_mark == 60:
        # np.save('cat_train_set.npy', train_set)
        np.save('E:\\torch_load\\venv\\pic\\Ch4\\cat_test_set.npy', test_set)
        print("test_set done")
    else:
        print(sucess_mark)
        print('Not enough data!')
    return train_set,test_set
# def model_info(model):
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     backbone = model.to(device)
#     summary(backbone, (3, 128, 128))
# , (3, 128, 128)
if __name__ == '__main__':
    m1 = 70  # 训练集样本数
    m2 = 30  # 测试样本数
    length = 128  # 图像的长和宽
    width = 128
    samplenum = 140
    minibatch = 4
    w_HR = 128
    epochs = 140

    # train_set = np.zeros([m1 * 2, length, width, 3])
    # train_set = np.reshape(train_set, (m1 * 2, 3, length, width))
    # test_set = np.zeros([m2 * 2, length, width, 3])
    # test_set = np.reshape(test_set, (m2 * 2, 3, length, width))
    # img_set()
    #导入数据
    net = ICNET()
    if torch.cuda.is_available():
        net = net.cuda()
    # model_info(net)
    # print(net)
    x0 = np.zeros(minibatch * 3 * w_HR ** 2)
    x0 = np.reshape(x0,(minibatch,3,w_HR,w_HR))
    y0 = np.zeros(minibatch)
    x0 = torch.tensor(x0).type(torch.FloatTensor).cuda()
    y0 = torch.tensor(y0).type(torch.LongTensor).cuda()
    x = np.load("E:\\torch_load\\venv\\pic\\Ch4\\cat_train_set.npy")
    x = torch.tensor(x)
    x = x.float().cuda()
    zero = torch.zeros(m1)
    one = torch.ones(m1)
    # cat = np.hstack((one,zero))
    # dog = np.hstack((zero,one))
    # y = np.vstack((one,zero))
    y = torch.cat((zero,one)).type(torch.LongTensor)
    # y = torch.tensor(y).cuda()
    y = y.float().cuda()

    loss_func = nn.CrossEntropyLoss()#交叉熵
    optimizer = torch.optim.SGD(net.parameters(), lr=0.005)#
    loss_value = []
    for epoch in range(epochs):
        for iteration in range(int(samplenum/minibatch)):
            k = 0
            for i in range(iteration*minibatch,iteration*minibatch+minibatch):
                x0[k,0,:,:] = x[i,0,:,:]
                x0[k,1,:,:] = x[i,1,:,:]
                x0[k,2,:,:] = x[i,2,:,:]
                y0[k] = y[i]
                k = k+1
            # x0 = torch.tensor(x0)
            # y0 = torch.tensor(y0)
            out = net.forward(x0)
            loss = loss_func(out,y0)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        loss_value.append(loss)
        print("epoch:{0},loss:{1}".format(epoch,loss))
    plt.plot(loss_value)
    plt.title("loss graph")
    plt.show()
    torch.save(net, "./model4.pkl")