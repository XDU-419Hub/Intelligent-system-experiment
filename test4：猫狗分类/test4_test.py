import torch
import numpy as np
net=torch.load("./model4_1.pkl")
# print(net)
x = np.load("E:\\torch_load\\venv\\pic\\Ch4\\cat_train_set.npy")/255
x = torch.tensor(x)
x = x.float().cuda()
m2 = 70  # 测试样本数
zero = torch.zeros(m2)
one = torch.ones(m2)
y0 = torch.cat((zero,one))

y = net(x)
a1 = torch.max(y,1)[1].cpu().data.numpy()
a2 = y0.data.numpy()
print(f"accuracy:{sum(a1==a2)/(2*m2)}")
