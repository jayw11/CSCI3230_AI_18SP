from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
print(__doc__)
import numpy as np
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

svr_poly = SVR(kernel='poly', C=0.0001, degree=2)

y_poly = svr_poly.fit(X_train, y_train).predict(X_test)



r23=r2_score(y_test , y_poly, multioutput='variance_weighted')
print("r2=",r23)
k=0

SE3=0
while k < len(y_test):
    SE3=SE3 + pow((y_test[k]-y_poly[k]), 2)
    k=k+1

MSE3=SE3/len(y_test)

print("MSE of poly",MSE3)


xdat = np.reshape(xdata, (len(xdata), 1))
ydat = np.reshape(y_test, (len(y_test), 1))

pv3 = np.reshape(y_poly, (len(xdata), 1))


plt.plot(xdat, pv3, color='green',linewidth=1)
plt.scatter(xdat, ydat, color='black',linewidth=1)
plt.scatter(xdat, y_poly, color='red',linewidth=1)

plt.show()