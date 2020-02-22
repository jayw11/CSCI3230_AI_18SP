print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.utils import check_random_state

import csv
diabetes_X_train= []
diabetes_y_train= []
diabetes_X_test= []
diabetes_y_test= []
f = open('Datatrain.csv')
for row in csv.reader(f):
    diabetes_X_train.append(float(row[3]))
    diabetes_y_train.append(float(row[4]))
f.close()

f = open('Datatest.csv')
for row in csv.reader(f):
    diabetes_X_test.append(float(row[3]))
    diabetes_y_test.append(float(row[4]))
f.close()


ir = IsotonicRegression()
y_ = ir.fit_transform(diabetes_X_train, diabetes_y_train)
#lr = LinearRegression()
#lr.fit(diabetes_X_train, diabetes_y_train)  # x needs to be 2d for LinearRegression

segments = [[[i, diabetes_y_train[i]], [i, y_[i]]] for i in range(len(diabetes_X_train))]
lc = LineCollection(segments, zorder=0)
lc.set_array(np.ones(len(diabetes_y_train)))
lc.set_linewidths(0.5 * np.ones(len(diabetes_X_train)))

fig = plt.figure()
#plt.plot(diabetes_X_train, diabetes_y_train, 'r.', markersize=12,color='green')
plt.plot(diabetes_X_test, diabetes_y_test, 'r.', markersize=12,color='black')
#plt.plot(diabetes_X_train, y_, 'g.-', markersize=12,color='yellow')
plt.plot(diabetes_X_test, ir.predict(diabetes_X_test), 'b-',color='red')
#plt.gca().add_collection(lc)
print("a=",diabetes_X_test)
print("a=",ir.predict(diabetes_X_test))
r1=r2_score(diabetes_y_train , ir.predict(diabetes_X_train), multioutput='variance_weighted')
print("r1=",r1)
#r2=r2_score(diabetes_y_test , ir.predict(diabetes_X_test), multioutput='variance_weighted')
#print("r2=",r2)

#plt.legend(('Data', 'Isotonic Fit', 'Linear Fit'), loc='lower right')
plt.title('Isotonic regression')
plt.show()

