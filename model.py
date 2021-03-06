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
from train_test_data_struct import train_test, voluem_train_test

##数据处理
'''x_base , x_volume , y = data_normalization()
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
train_loader = DataLoader(dataset=deal_dataset , batch_size=1,shuffle=True , num_workers=0)'''
#train_loader , x_test , y_test = train_test(0)

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
        self.out = nn.Linear(44 ,2)
    def forward(self, x):
        x = Variable(torch.unsqueeze(x, dim=0).float(), requires_grad=False)###############成交量作为属性多加了一行
        x = self.conv1(x)
        x = self.conv2(x)
        #print('hhhhhhhhhhhhhhh',x.shape)#[1,1,89,3]
        x = x.view(x.size(0), -1) #flat (batch_size, 32*7*7)
        output = self.out(x)
        return output
    def par(self):
        params_cnn1 = list(self.conv1.named_parameters())
        '''print(params_cnn1.__len__())
        print(params_cnn1[0])
        print(params_cnn1[1])'''
        params_cnn2 = list(self.conv2.named_parameters())
        '''print(params_cnn2.__len__())
        print(params_cnn2[0])
        print(params_cnn2[1])'''
        params_linear = list(self.out.named_parameters())
        '''print(params_linear[0])
        print(params_linear[1])'''
        return params_cnn1 , params_cnn2 , params_linear

def run(train_loader,x_test , y_test , index):
    cnn = NET()
    optimizer = torch.optim.Adam(cnn.parameters(),lr = 0.01)
    loss_func = nn.CrossEntropyLoss()
    ######训练开始
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
                print('Epoch :{}[{}/{}({:.0f}%)]\t Loss:{:.6f}\t Accuracy:{:.3f}'.format(epoch,i * len(inputs),len(train_loader.dataset),100.*i / len(train_loader),loss.data.item(),float(correct)/(i+1)))
    ########################训练结束
    #######################保存模型
    torch.save(cnn,'modle\\'+str(index) + '.pt')
    ##############3#########测试开始
    y_pred = []
    cnn.par()
    for i in range(len(x_test)):
        test_x = x_test[i]
        test_x = torch.from_numpy(test_x)
        test_x = Variable(torch.unsqueeze(test_x, dim=0).float(), requires_grad=False)
        test_x = Variable(torch.unsqueeze(test_x, dim=0).float(), requires_grad=False)
        test_out = cnn(Variable(test_x))
        pred_y = torch.max(test_out,1)[1].data.numpy().squeeze()
        y_pred.append(pred_y)
        '''print(pred_y,'预测成员')
        print(y_test[i])'''
    result = 0
    for i in range(len(y_test)):
        if y_test[i] == y_pred[i]:
            result += 1
    print('->>>>>>>>>>>>>>>>>>>>>>>>>>>准确率：{}'.format(result/len(y_test)))
    a , b , c = cnn.par()
    return a,b ,c , result/len(y_test)
    


if __name__ =='__main__':
    cnn = NET() 
    a , b , c = cnn.par()
    print(c)
    print(a)
    time.sleep(10)
    
    #a = [('0.weight', Parametercontaining: tensor([[[[-0.1991, -0.3832],[ 0.2564, -0.0863]]]], requires_grad=True)), ('0.bias', Parameter containing:tensor([-0.0599], requires_grad=True))]
    ################开高低收作为属性训练
    train_loader , x_test , y_test = train_test(0)
    print('开始')
    a , b ,c =  run(train_loader , x_test , y_test,0) 
    print(a.__len__())
    ###################成交量作为属性的训练
    train_loader , x_test , y_test = voluem_train_test(0)
    print('开始')
    a , b ,c ,d=  run(train_loader , x_test , y_test,0) 
    print(a.__len__())