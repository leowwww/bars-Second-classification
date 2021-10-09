import numpy as np
import pandas as pd
from model import run
from train_test_data_struct import train_test
import time

if __name__ == '__main__':
    bias = []
    weight = []
    start = time.time()
    train_loader , x_test , y_test = train_test(0)
    a , b , c = run(train_loader=train_loader , x_test= x_test , y_test= y_test , index=0)
    '''print(type(a))
    print(type(a[0]))
    print(a[0])
    print(a[1])
    print(a[1][1][0].item())###bias
    print(a[0][1][0][0][0][0].item())##weight
    print(a[0][1][0][0][0][1].item())
    print(a[0][1][0][0][1][0].item())
    print(a[0][1][0][0][1][1].item())'''
    
    print(b[0])
    print(b[1])
    end = time.time()
    print('耗时：',end-start)