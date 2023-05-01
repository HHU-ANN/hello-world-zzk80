# 最终在main函数中传入一个维度为6的numpy数组，输出预测值

import os

try:
    import numpy as np
except ImportError as e:
    os.system("sudo pip3 install numpy")
    import numpy as np

def ridge(data):
    x,y = read_data()
    lambd = 1e-10
    weight = np.dot(np.linala.inv(np.(np.dot(x.T,x)+np.dot(lambd,np.eye(6)))),np.dot(x.T,y))
    return weight @ data
    
def lasso(data):
    label = 1e-10
    x,y = read_data()
    weight = np.zeros([6,6])
    y1 = np.dot(weight,x)
    rate = 1e-5
    for i in range(int(1e10)):
        y = np.dot(weight, x)
        loss = (np.sum(y1 - y) ** 2) / 6
        if loss < label:
            break
        weight = weight - np.dot((y1 - y),x.T) * rate
    return weight @ data

def read_data(path='./data/exp02/'):
    x = np.load(path + 'X_train.npy')
    y = np.load(path + 'y_train.npy')
    return x, y
