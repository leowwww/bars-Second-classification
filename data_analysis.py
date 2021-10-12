import pandas as pd 
import numpy as np
import json
import time
def data_analysis (path , cols , save_path):
    df = pd.read_excel(path , usecols = cols)
    df = df.sort_values( by = cols[0]-1,axis=0)
    bias_con1_resullt = {}
    count = 0
    cache = 0
    for i in range(len(df)-1):
        if df.values[i][0] == df.values[i+1][0] and count == 0:
            cache = df.values[i][0]
            count += 1
            continue
        if count != 0 and df.values[i][0] == cache:
            count += 1
            continue
        if count != 0 and df.values[i][0] != cache:
            bias_con1_resullt[cache] = count
            count = 0
            cache = 0
    ############将结果写入文本文件
    bias_con1_resullt = sorted(bias_con1_resullt.items(),key=lambda x:x[1],reverse=True)
    file = open(save_path,'w')
    for i in range (len(bias_con1_resullt)):
        file.write(f"{bias_con1_resullt[i][0]} : {bias_con1_resullt[i][1]}\n".format(bias_con1_resullt[i][0],bias_con1_resullt[i][1]))
    '''for k,v in bias_con1_resullt.items():
        file.write(f'{k} : {v}\n'.format(k,v))'''
    file.close()
    return bias_con1_resullt[0][0]
if __name__ == '__main__':
    weight_con1_parameter = []
    weight_con2_parameter = []
    weight_lin_parameter = []
    a = data_analysis('excel\\bias_lin.xlsx',[2],'parameter\\bias_lin_2.txt')
    time.sleep(10)
    for i in range(4):
        save_path = f'parameter\\weight_con1_{i+1}.txt'.format(i+1)
        #print(save_path)
        a = data_analysis('excel\\weight_con1.xlsx',[i+1],save_path)
        weight_con1_parameter.append(a)
    print(weight_con1_parameter)
    for i in range(25):
        save_path = f'parameter\\weight_con2_{i+1}.txt'.format(i+1)
        #print(save_path)
        a = data_analysis('excel\\weight_con2.xlsx',[i+1],save_path)
        weight_con2_parameter.append(a)
    print(weight_con2_parameter)
    for i in range(88):
        save_path = f'parameter\\weight_lin_{i+1}.txt'.format(i+1)
        #print(save_path)
        a = data_analysis('excel\\weight_lin.xlsx',[i+1],save_path)
        weight_lin_parameter.append(a)
    print(weight_lin_parameter)

