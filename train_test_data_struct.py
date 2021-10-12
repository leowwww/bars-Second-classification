import numpy as np 
import pandas as pd
from torch.functional import split
from data_processing import *
from torch.autograd import Variable, variable
from torch.utils.data import Dataset , DataLoader,TensorDataset, dataloader
###############生成训练集和测试集


def train_test(plit):
    x_base , x_volume , y = data_normalization()
    '''print(x_volume[0])
    x_volume = change_data(x_volume)
    x_volume = np.array(x_volume)
    print(x_volume.shape)'''
    #x_base = np.array(x_base)
    y = np.array(y)
    #print(len(y))
    x_base = change_data(x_base)
    x_base = np.array(x_base)
    #print(type(x_base))
    '''x_train = x_base[:4720]
    y_train = y[:4720]
    x_test = x_base[4720:]
    y_test = y[4720:]'''
    x_test = x_base[plit:plit+1180]#每一个交易日都要作为测试集，同样每一个交易日也要作为训练集
    y_test= y[plit:plit+1180]
    #print(y_test)
    a = x_base[:plit]
    b = x_base[plit+1180:]
    x_train = np.vstack((a,b))
    #print(x_train.shape)
    #print(type(y))
    a = y[:plit]
    b = y[plit+1180:]
    y_train = np.hstack((a,b))
    #rint(y_train)

    x_train = variable(x_train)
    y_train = variable(y_train)
    #print(type(x_train))
    deal_dataset = TensorDataset(x_train , y_train)
    train_loader = DataLoader(dataset=deal_dataset , batch_size=1,shuffle=True , num_workers=0)
    return train_loader , x_test , y_test
def voluem_train_test(plit):#将成交量作为属性进行训练
    x_base , x_volume , y = data_normalization()
    print(x_volume[0])
    x_volume = change_data(x_volume)
    x_volume = np.array(x_volume)
    #print(x_volume.shape)
    x_test = x_volume[plit:plit+1180]#每一个交易日都要作为测试集，同样每一个交易日也要作为训练集
    y_test= y[plit:plit+1180]
    #print(y_test)
    a = x_volume[:plit]
    b = x_volume[plit+1180:]
    x_train = np.vstack((a,b))
    #print(x_train.shape)
    #print(type(y))
    a = y[:plit]
    b = y[plit+1180:]
    y_train = np.hstack((a,b))
    #rint(y_train)

    x_train = variable(x_train)
    y_train = variable(y_train)
    #print(type(x_train))
    deal_dataset = TensorDataset(x_train , y_train)
    train_loader = DataLoader(dataset=deal_dataset , batch_size=1,shuffle=True , num_workers=0)
    return train_loader , x_test , y_test




if __name__ == '__main__':
    a , b ,c= train_test(0)
    a , b ,c = voluem_train_test(0)
    print(a)