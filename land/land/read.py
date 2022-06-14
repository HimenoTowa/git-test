import numpy as np
import matplotlib.pyplot as plt
import math

y = []
with open('land/27.csv','r') as f:
    x = f.readline()
    x.strip()
    x = x.split(',')
    x = [float(i) for i in x]
    for line in f.readlines()[0:]:
        line.strip()
        t = line.split(',')
        t = [float(i) for i in t]
        y.append(t)
print(x)
print(y)

plt.figure(figsize=(20, 8))

data_t = np.array(x)
data_y = np.array(y)



# data = np.fromfile('test.dat',dtype='float32')
# print(data)

pred_x = []
pred_y = []
with open('land/test.dat','r') as f:
    _ = f.readline()
    for line in f.readlines():
        line.strip()
        t = line.split()
        t = [float(i) for i in t]
        pred_x.append(t[0])
        pred_y.append(t[1:])
pred_xx = np.array(pred_x)
pred_yy = np.array(pred_y).T
# print('pred_x',np.array(pred_x))
# # print('pred_y',pred_y)
# print('pred_y_T',np.array(pred_y).T)





for i in range(20):
    plt.subplot(2, 10, i+1)
    plt.title('C'+str(i+1))
    plt.plot(pred_xx,pred_yy[i], color='b')
    plt.scatter(data_t, data_y[i], s=2, color='r')

plt.tight_layout()
plt.show()
