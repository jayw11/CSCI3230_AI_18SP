from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
print(__doc__)
import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.svm import SVR
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import csv
X_train= []
y_train= []
X_test= []
y_test= []
xdata=[]
trainlen=0
testlen=0
i=0
f = open('Datatrain.csv')
for row in csv.reader(f):
    y_train.append(float(row[4]))
    X_train.append(float(row[0]))
    X_train.append(float(row[1]))
    X_train.append(float(row[2]))
    X_train.append(float(row[3]))
    trainlen=trainlen+1
f.close()

f = open('Datatest.csv')
for row in csv.reader(f):
    i=i+1
    X_test.append(float(row[0]))
    X_test.append(float(row[1]))
    X_test.append(float(row[2]))
    X_test.append(float(row[3]))
    y_test.append(float(row[4]))
    testlen=testlen+1
    xdata.append(i)
f.close()


X_train=np.reshape(X_train,(trainlen,4))
y_train=np.reshape(y_train,(len(y_train),1))
X_test=np.reshape(X_test,(testlen,4))
y_test=np.reshape(y_test,(len(y_test),1))
svr_rbf = SVR(kernel='rbf', C=1e7, gamma=0.0000001)
svr_lin = SVR(kernel='linear', C=0.001)


y_rbf = svr_rbf.fit(X_train, y_train).predict(X_test)

ypretest = svr_rbf.fit(X_train, y_train).predict(X_test)
y_lin = svr_lin.fit(X_train, y_train).predict(X_test)


r21=r2_score(y_test , y_rbf, multioutput='variance_weighted')
print("r2 of rbf=",r21)
r22=r2_score(y_test , y_lin, multioutput='variance_weighted')
print("r2 of linear",r22)

k=0
SE1=0
SE2=0

while k < len(y_test):
    SE1=SE1 + pow((y_test[k]-y_rbf[k]), 2)
    SE2=SE2 + pow((y_test[k]-y_lin[k]), 2)
    k=k+1
MSE1=SE1/len(y_test)
MSE2=SE2/len(y_test)
print("MSE of rbf",MSE1,"MSE of Linear",MSE2)


xdat = np.reshape(xdata, (len(xdata), 1))
ydat = np.reshape(y_test, (len(y_test), 1))

pv1 = np.reshape(y_rbf, (len(xdata), 1))
pv2 = np.reshape(y_lin, (len(xdata), 1))

plt.plot(xdat, pv1, color='yellow',linewidth=1)
plt.plot(xdat, pv2, color='green',linewidth=1)

plt.scatter(xdat, ydat, color='black',linewidth=1)


plt.show()