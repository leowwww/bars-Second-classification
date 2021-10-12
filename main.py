import numpy as np
import pandas as pd
from model import run
from train_test_data_struct import train_test, voluem_train_test
import time

if __name__ == '__main__':

    bias_con1 = []
    weight_con1 = []
    bias_con2 = []
    weight_con2 = []
    bias_lin = []
    weight_lin = []
    weight_con1_ex = []
    weight_con2_ex = []
    bias_lin_ex = []
    weight_lin_ex = []
    ACC = []
    start = time.time()
    for  k in range(100):
        weight_con1_ex = []
        weight_con2_ex = []
        bias_lin_ex = []
        weight_lin_ex = []
        train_loader , x_test , y_test = voluem_train_test(k)
        a , b , c, acc = run(train_loader=train_loader , x_test= x_test , y_test= y_test , index=0)
        ACC.append(acc)
        '''print(type(a))
        print(type(a[0]))
        print(a[0])
        print(a[1])
        print(a[1][1][0].item())###bias
        print(a[0][1][0][0][0][0].item())##weight
        print(a[0][1][0][0][0][1].item())
        print(a[0][1][0][0][1][0].item())
        print(a[0][1][0][0][1][1].item())
        print('b[0]',b[0])
        print('b[1]',b[1])
        print('c[0]',c[0])
        print('c[1]',c[1])
        print(b[0][1][0][0][4][4].item())
        print(c[0][1][0][43].item())'''

    #####################第一个卷积层的参数
        for i in range(len(a[0][1][0][0])):
            for j in range(len(a[0][1][0][0][0])):
                num = round(a[0][1][0][0][i][j].item(),4)
                weight_con1_ex.append(num)
        print(weight_con1_ex)
        weight_con1.append(weight_con1_ex)
        bias_con1.append(round(a[1][1][0].item(),4))
    #####################第二层卷积层的参数
        for i in range(len(b[0][1][0][0])):
            for j in range(len(b[0][1][0][0][0])):
                num = round(b[0][1][0][0][i][j].item(),4)
                weight_con2_ex.append(num)
        weight_con2.append(weight_con2_ex)
        bias_con2.append(round(b[1][1][0].item(),4))
    #####################全连接层的参数
        for i in range (len(c[0][1])):
            for j in range(len(c[0][1][0])):
                num = round(c[0][1][i][j].item(),4)
                weight_lin_ex.append(num)
        weight_lin.append(weight_lin_ex)
        for i in range(len(c[1][1])):
            #num = ('%.4f' % c[1][1][i].item())
            num = round(c[1][1][i].item(),4)
            bias_lin_ex.append(num)
        bias_lin.append(bias_lin_ex)
        
#############写入excel中
    
    weight_con1 = pd.DataFrame(weight_con1)
    weight_con1.to_excel('excel_volume\\weight_con1.xlsx')

    bias_con1 = pd.DataFrame(bias_con1)
    bias_con1.to_excel('excel_volume\\bias_con1.xlsx')

    weight_con2 = pd.DataFrame(weight_con2)
    weight_con2.to_excel('excel_volume\\weight_con2.xlsx')

    bias_con2 = pd.DataFrame(bias_con2)
    bias_con2.to_excel('excel_volume\\bias_con2.xlsx')

    weight_lin = pd.DataFrame(weight_lin)
    weight_lin.to_excel('excel_volume\\weight_lin.xlsx')

    bias_lin =pd.DataFrame(bias_lin)
    bias_lin.to_excel('excel_volume\\bias_lin.xlsx')

    ACC = pd.DataFrame(ACC)
    ACC.to_excel('excel_volume\\accuracy.xlsx')



    end = time.time()
    print('耗时：',end-start)