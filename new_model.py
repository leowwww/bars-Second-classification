import torch
from torch._C import *
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.autograd import Variable, variable
from torch.nn.modules import padding
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
        super(NET, self).__init__()
        #########################第一层卷积和参数
        kernel_con1 = [
            [-0.0003,-0.2824],
            [-0.4692,-0.0292]]
        kernel_con1 = torch.FloatTensor(kernel_con1).unsqueeze(0).unsqueeze(0)
        self.weight_con1 = nn.Parameter(data=kernel_con1, requires_grad=False)
        self.bias_con1 = nn.Parameter(data = torch.FloatTensor([-0.5247]),requires_grad=False )
        ########################第二层卷积层参数
        kernel_con2 = [[-0.1589, -0.1334, -0.165, -0.1542, 0.0135],
                        [-0.0624, 0.1028, 0.0047, -0.129, -0.0851],
                        [-0.1823, -0.0507, -0.01, -0.2193, -0.1763],
                        [-0.1552, -0.1868, -0.1929, -0.0695, 0.0538],
                        [-0.1777, -0.1924, -0.1317, -0.0695, -0.1663]]
        kernel_con2 = torch.FloatTensor(kernel_con2).unsqueeze(0).unsqueeze(0)   
        self.weight_con2 = nn.Parameter(data=kernel_con2, requires_grad=False)  
        self.bias_con2 = nn.Parameter(data = torch.FloatTensor([0]),requires_grad=False )
        ########################全连接层参数
        kernel_lin = [[0.0083, -0.1212, -0.15, -0.0576, -0.1, -0.1163, 0.0651, 0.0112, 0.0661, 0.0216, -0.0349, -0.069, -0.1448, 0.0444, -0.0539, -0.0251, -0.0999, -0.144, -0.0499, -0.085, 0.1056, -0.1238, -0.1068, -0.0927, 0.0229, -0.0259, -0.1485, -0.0083, 0.0477, -0.0969, -0.0846, -0.0598, -0.0082, 0.1252, -0.0825, -0.0097, -0.0075, -0.1482, 0.0787, -0.016, 0.0293, -0.0282, 0.0021,-0.1005],
                      [ -0.0162, -0.0499, -0.1062, 0.0586, 0.0572, 0.0031, 0.012, 0.0606, 0.0047, -0.1035, 0.0433, -0.1051, 0.017, -0.1431, -0.073, -0.0267, -0.1493, -0.1305, -0.1046, 0.1276, 0.039, -0.0864, -0.0882, -0.1325, -0.0871, -0.1381, -0.0318, -0.1371, 0.0882, -0.0572, -0.117, -0.1246, 0.0955, -0.0718, -0.0901, -0.1301, 0.0331, -0.0968, 0.1065, -0.1325, -0.0167, -0.0824, -0.1378, 0.0088]]          
        self.weight_lin =  nn.Parameter(data= torch.FloatTensor(kernel_lin), requires_grad=False)
        self.bias_lin = nn.Parameter(data = torch.FloatTensor([0.424 , -0.5613]),requires_grad=False )
        
        #######################构造网络层
        '''self.conv1 = F.conv2d(x, self.weight_con1,self.bias_con1, stride = 1,padding=2)
        self.conv2 = F.conv2d(x,self.weight_con2 , self.bias_con2 , stride=1 , padding=2)
        self.out = F.linear(x,self.weight_lin , self.bias_lin)'''
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2)

    def forward(self, x):
        #print(x.shape)
        x = F.conv2d(x, self.weight_con1,self.bias_con1, stride = 1,padding=2)
        x = self.relu(x)
        x = self.maxpool(x)
        #print(x.shape)
        x = F.conv2d(x,self.weight_con2 , self.bias_con2 , stride=1 , padding=2)
        x = self.relu(x)
        x = self.maxpool(x)
        x = x.view(x.size(0), -1)
        #print(x.shape)
        out = F.linear(x,self.weight_lin , self.bias_lin)
        return out

def run(x_test , y_test):
    cnn = NET()
    ##############3#########测试开始
    y_pred = []
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
    return result/len(y_test)
    


if __name__ =='__main__':
    train_loader , x_test , y_test = train_test(0)
    result = run(x_test,y_test)
    acc = []
    for i in range(100):
        train_loader , x_test , y_test = train_test(i)
        result = run(x_test , y_test)
        acc.append(result)
    ACC = pd.DataFrame(acc)
    ACC.to_excel('new_model\\accuracy.xlsx')