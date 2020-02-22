from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import math
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
import tensorflow as tf
import csv

intydata = []
ydata = []
xdata=[]
intpv = []
ytrain= []
xtrain=[]
#f = open('Datatrain.csv')
#for row in csv.reader(f):
#    ydata.append(row[2])
#    xdata.append(row[0])
#f.close()
f = open('Datatest.csv')
for row in csv.reader(f):
    ydata.append(float(row[4]))
   # xtrain.append(row[0])
f.close()

tf.logging.set_verbosity(tf.logging.INFO)

COLUMNS = ["T", "H", "P", "PM", "GT"]
FEATURES = ["T", "H", "P", "PM"]
LABEL = "GT"

def input_fn(data_set):
  feature_cols = {k: tf.constant(data_set[k].values) for k in FEATURES}
  labels = tf.constant(data_set[LABEL].values)
  return feature_cols, labels

def main(unused_argv):

  training_set = pd.read_csv("Datatrain.csv", skipinitialspace=True,
                             skiprows=0, names=COLUMNS)
  test_set = pd.read_csv("Datatest.csv", skipinitialspace=True,
                         skiprows=0, names=COLUMNS)
  prediction_set = pd.read_csv("Datatest.csv", skipinitialspace=True,
                               skiprows=0, names=COLUMNS)
  # Feature cols
  feature_cols = [tf.contrib.layers.real_valued_column(k)
                  for k in FEATURES]

  # Build 2 layer fully connected DNN with 10, 10 units respectively.
  regressor = tf.contrib.learn.DNNRegressor(
      feature_columns=feature_cols, hidden_units=[10, 10])

  # Fit
  regressor.fit(input_fn=lambda: input_fn(training_set), steps=10000)

  # Score accuracy
  ev = regressor.evaluate(input_fn=lambda: input_fn(test_set), steps=2)
  loss_score = ev["loss"]
  print("Loss: {0:f}".format(loss_score))

  # Print out predictions
  prediction_value = regressor.predict(input_fn=lambda: input_fn(prediction_set))
 # print ("Predictions: {}".format(str(prediction_value)))
  i=0
  difference=0
  sum=0

  while i < len(prediction_value):
      intydata = float(ydata[i])
      intpv = float(prediction_value[i])
      xdata.append(i)
      difference = difference + pow((intydata-intpv), 2)
      sum = sum + intydata
      i = i + 1
  mse=difference/len(prediction_value)
  average = sum / len(prediction_value)
  j=0
  avs=0
 # print("sum=",sum,"avg=",average)
  while j < len(prediction_value):
      intydata2 = int(ydata[j])
      avs = avs + pow((intydata2-average), 2)
      j=j+1

  print("avs=", avs/len(prediction_value))
  print("mse=",mse)

  print("error=",1-(difference/avs))

  xdat = np.reshape(xdata, (len(xdata), 1))
  ydat = np.reshape(ydata, (len(ydata), 1))
  #xt = np.reshape(xtrain , (len(xtrain ), 1))
  #yt  = np.reshape(ytrain , (len(ytrain ), 1))
  pv = np.reshape(prediction_value, (len(xdata), 1))
  r1 = r2_score(ydat, regressor.predict(input_fn=lambda: input_fn(prediction_set)), multioutput='variance_weighted')
  print("r1=", r1)
  #lt.plot(xt, yt, color='blue', linewidth=1)
 # plt.scatter(xt, yt, color='green', linewidth=1)
  plt.plot(xdat, pv, color='yellow',linewidth=1)
  plt.scatter(xdat, ydat, color='black',linewidth=1)
  plt.show()

if __name__ == "__main__":
   tf.app.run()