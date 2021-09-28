from numpy import result_type
import pandas as pd
import scipy.io
import os
import time
import datetime
import numpy as np

def date(dates):#定义转化日期戳的函数,dates为日期戳
  delta=datetime.timedelta(days=dates)
  today=datetime.datetime.strptime('1899-12-30 00:00:00','%Y-%m-%d %H:%M:%S')+delta#将1899-12-30转化为可以计算的时间格式并加上要转化的日期戳
  return datetime.datetime.strftime(today,'%Y-%m-%d %H:%M:%S')#制定输出日期的格式

def data_normalization():
    #data = scipy.io.loadmat('result\\1.mat')['yz_data_cut']
    df = pd.read_excel('result\\OP_IF_result.xlsx',usecols=[6])
    y = []#标签
    x_base =[]#数据
    x_volume = []
    for i in range(len(df)):
        if df.loc[i].values[0] < 1:
            y.append(0)
        else:
            y.append(1)
    
    file_head = os.listdir('result')
    for i in range(len(file_head) - 1):
        file_path = 'D:\\代码\\策略\\result'+'\\'+file_head[i]
        data = scipy.io.loadmat(file_path)['yz_data_cut']
        data_base = data[:,1:-1]
        x_base.append(data_base)
        data_volume = data[:,-1]
        x_volume.append(data_volume)
    return(x_base , x_volume , y)
def change_data(data):
    #对维度不一样的数据进行填充
    b = []
    for j in range (198 - 176):
        b.append(176+j)

    for i in range (len(data)):
        if len(data[i]) == 198:
            data[i] = np.delete(data[i],b,axis=0)
    return data


if __name__ == '__main__':
    x_base , x_volume , y = data_normalization()
    #x_base = np.array(x_base)
    y = np.array(y)
    print(type(x_base))
    print(x_base[0].shape)
    x_test = x_base[:10]
    y_test = y[:10]
    new_data = change_data(x_base)
    new_data = np.array(new_data)
    print(type(new_data))
    