# Hamid Reza Heidari
# https://github.com/HamidRezaHeidari

# Single layer perceptron from scratch

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.inspection import DecisionBoundaryDisplay


iris = load_iris()
x = iris.data[:, 0]
y = iris.data[:, 1]

# split Iris into two class
iris_class = np.zeros(len(iris.target))
for i in range(0,len(iris.target)):
    if iris.target[i] == 0:
        iris_class[i] = 1
    else:
        iris_class[i] = -1



# combine x , y array
d = np.stack((x, y), axis=1)


# divide dataset to  test and train sets
data_train, data_test, train_class, test_class = train_test_split(d, iris_class, test_size=0.2, random_state=59)

x_train = np.zeros(len(data_train))
y_train = np.zeros(len(data_train))

x_test = np.zeros(len(data_test))
y_test = np.zeros(len(data_test))

for i in range(0,len(data_train)):
    x_train[i] = data_train[i][0]
    y_train[i] = data_train[i][1]

for i in range(0,len(data_test)):
    x_test[i] = data_test[i][0]
    y_test[i] = data_test[i][1]



# initial value
w = np.array([0.1, 0.2, 0])
c = 0.1
bias = -1
m = 0
E = 0
k = 0
Ev = np.array([])

# implement Activation Function
def threshold(x):
    return np.where(x>0, 1, -1)

# implement single scratch Perceptron
while(m!=1):
    for i in range(0,len(x_train)):
        d = train_class[i]
        yt = np.array([x_train[i], y_train[i], bias])
        f = np.dot(w,yt)
        o = threshold(f)
        w = w + 0.5*c*(d-o)*yt
        E = E + 0.5*((d-o)**2)

    k = k + 1
    Ev = np.append(Ev, np.array([E]), axis=0)
    print("error:", E)
    print('weight:', w)

    if E == 0:
        m = 1
    else:
        E = 0


print("Training cycle :", k)
print("Final Weight is:", w)
print("Learning finished")


# plot main dataset and new dataset

gs1 = gridspec.GridSpec(2, 2)
gs2 = gridspec.GridSpec(1, 2)

fig1 = plt.figure()

ax = plt.subplot(gs1[0,1])
plt.scatter(x, y, c=iris_class)
plt.title("Iris dataset splitted in two class")
plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1])

ax = plt.subplot(gs1[0, 0])
plt.scatter(x, y, c=iris.target)
plt.title("Main Iris dataset ")
plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1])

ax = plt.subplot(gs1[1, :])
plt.scatter(x_train, y_train, c='b')
plt.scatter(x_test, y_test, c='r')
plt.plot(x, (-w[0]/w[1])*x + (w[2]/w[1]), "--");
plt.title("Train-Test split")
plt.legend(["Train data", "Test data", "decision line"])

# plot Error in each cycle & Perceptron Classification result

fig2 = plt.figure()

ax = plt.subplot(gs2[0,0])
plt.step(range(0,len(Ev)), Ev , c="c")
plt.title("Error Value in earch cycle ")


ax = plt.subplot(gs2[0,1])
plt.scatter(x_test, y_test, c=test_class)
plt.plot(x, (-w[0]/w[1])*x + (w[2]/w[1]));
plt.title("Final Classification with Test dataset")

##############################################################

x_p = iris.data
y_p = iris_class

x_train, x_test, y_train, y_test = train_test_split(x_p, y_p, test_size=0.2, random_state=431)


# combine x , y array
d = np.stack((x, y), axis=1)



# Implement Perceptron with sklearn
P = Perceptron()
P.fit(x_train, y_train)
y_prediction = P.predict(x_test)

accuracy = accuracy_score(y_test, y_prediction)
print("sklearn Perceptron Accuracy :", accuracy)



reg = linear_model.LinearRegression(fit_intercept=True)
reg.fit(x_train, y_train)

classifier = LogisticRegression().fit(d, iris_class)
disp = DecisionBoundaryDisplay.from_estimator(
    classifier, d, response_method="predict",
    xlabel=iris.feature_names[0], ylabel=iris.feature_names[1],
    alpha=0.5,
)


# plot sklearn Perceptron Classification result

disp.ax_.scatter(iris.data[:, 0], iris.data[:, 1], c=iris_class, edgecolor="k")
plt.plot(x, (-w[0]/w[1])*x + (w[2]/w[1]), "--", c="k");
plt.title(" Classification with sklearn for Iris dataset and decision line from scratch Perceptron ")

plt.show()


