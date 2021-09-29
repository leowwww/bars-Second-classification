import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.autograd import Variable, variable
from torch.nn.modules.activation import ReLU
from torch.nn.modules.pooling import MaxPool2d
from torch.utils.data import Dataset , DataLoader,TensorDataset, dataloader
from data_processing import *
import time 

##数据处理
x_base , x_volume , y = data_normalization()
#x_base = np.array(x_base)
y = np.array(y)
x_base = change_data(x_base)
x_base = np.array(x_base)
#print(type(x_base))
x_train = x_base[:4720]
y_train = y[:4720]
x_test = x_base[4720:]
y_test = y[4720:]

x_train = variable(x_train)
y_train = variable(y_train)
print(type(x_train))
deal_dataset = TensorDataset(x_train , y_train)
train_loader = DataLoader(dataset=deal_dataset , batch_size=1,shuffle=True , num_workers=0)

class NET(nn.Module):
    def __init__(self):
        super(NET , self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1,1,2,1,2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(1,1,5,1,2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.out = nn.Linear(267 ,2)
    def forward(self, x):
        x = self.conv1(x)
        #x = self.conv2(x)
        #print('hhhhhhhhhhhhhhh',x.shape)#[1,1,89,3]
        x = x.view(x.size(0), -1) #flat (batch_size, 32*7*7)
        output = self.out(x)
        return output
cnn = NET()
optimizer = torch.optim.Adam(cnn.parameters(),lr = 0.01)
loss_func = nn.CrossEntropyLoss()

for epoch in range(2):
    correct = 0
    for i,data in enumerate(train_loader):
        inputs , lable = data
        inputs = Variable(inputs)
        lable = Variable(lable).long()
        inputs = Variable(torch.unsqueeze(inputs, dim=0).float(), requires_grad=False)
        output = cnn(inputs)
        loss = loss_func(output , lable)
        #print(loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        predicted = torch.max(output.data,1)[1]
        correct += (predicted == lable).sum()
            #print(correct)
        if i % 100 == 0:
            print('Epoch :{}[{}/{}({:.0f}%)]\t Loss:{:.6f}\t Accuracy:{:.3f}'.format(epoch,i * len(inputs),len(train_loader.dataset),100.*i / len(train_loader),loss.data.item(),float(correct*100)/float(1)*(i+1)))
x_test = torch.from_numpy(x_test)
x_test = Variable(torch.unsqueeze(x_test, dim=0).float(), requires_grad=False)
test_out = cnn(Variable(x_test))
pred_y = torch.max(test_out,1)[1].data.numpy().squeeze()
print(pred_y,'预测成员')
print(y_test)


