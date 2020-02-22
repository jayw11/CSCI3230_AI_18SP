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
xtrain = []
ytrain = []
xtest = []
ytest = []
xdata=[]
trainsum=0
f = open('01node1train.csv')
trainlen=0
testlen=0
i=0
f = open('Datatrain.csv')
for row in csv.reader(f):
    xtrain.append(float(row[0]))
    xtrain.append(float(row[1]))
    xtrain.append(float(row[2]))
    xtrain.append(float(row[3]))
    ytrain.append(float(row[4]))
    trainlen=trainlen+1
f.close()

f = open('Datatest.csv')
for row in csv.reader(f):
    i=i+1
    xtest.append(float(row[0]))
    xtest.append(float(row[1]))
    xtest.append(float(row[2]))
    xtest.append(float(row[3]))
    ytest.append(float(row[4]))
    testlen=testlen+1
    xdata.append(i)
f.close()


xtrain = np.reshape(xtrain, (len(xtrain), 1))
ytrain = np.reshape(ytrain, (len(ytrain), 1))
xtest = np.reshape(xtest, (len(xtest), 1))
ytest = np.reshape(ytest, (len(ytest), 1))

xtrain=np.reshape(xtrain,(trainlen,4))
ytrain=np.reshape(ytrain,(len(ytrain),1))
xtest=np.reshape(xtest,(testlen,4))
ytest=np.reshape(ytest,(len(ytest),1))

def add_layer(inputs, in_size, out_size, activation_function=None):
    # add one more layer and return the output of this layer
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.5)
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs


# define placeholder for inputs to network
xs = tf.placeholder(tf.float32, [None, 4])
ys = tf.placeholder(tf.float32, [None, 1])
# add hidden layer
l1 = add_layer(xs, 4, 2, activation_function=tf.nn.relu)
# add output layer
prediction = add_layer(l1, 2, 1, activation_function=tf.nn.relu6)

# the error between prediciton and real data
loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.4).minimize(loss)

# important step
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

xdat = np.reshape(xdata, (len(xdata), 1))
ydat = np.reshape(ytest, (len(ytest), 1))

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(xdat, ydat)
plt.show(block=False)

for i in range(500):
    # training
    sess.run(train_step, feed_dict={xs: xtrain, ys: ytrain})
    if i % 50 == 0:
        # to see the step improvement
        print("loss:",sess.run(loss, feed_dict={xs: xtrain, ys: ytrain}))
        #if i > 0:
        #  try:
        #      ax.lines.remove(lines[0])
        #  except Exception:
        #      pass
        prediction_value = sess.run(prediction, feed_dict={xs: xtest})
          # plot the prediction
        #  lines = ax.plot(xtrain, prediction_value, 'r-', lw=5)
        #  plt.pause(2)

plt.plot(xdat,prediction_value, color='yellow',linewidth=1)
plt.scatter(xdat, prediction_value, color='red',linewidth=1)

#print("predict",sess.run(prediction,feed_dict={xs: xtrain}))

prediction_value = np.reshape(prediction_value, (len(prediction_value), 1))
r1=r2_score(ytest,prediction_value,multioutput='variance_weighted')
print("r2=",r1)

#print("trainloss=",r2/len(ytrain),"trainaverage=",average)
plt.show()