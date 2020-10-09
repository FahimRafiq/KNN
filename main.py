import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from Knn import Knn
import matplotlib.pyplot as plt

iris = datasets.load_digits()
# print(print(iris))
X, y = iris.data, iris.target


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1000)
# print(X_train.shape)
# print(X_train[0])

# plt.figure()
# plt.scatter(X[:,[0]], X[:,[1]], c=y)
# plt.show()
acc = 0
for k in [3,5,7]:
    clf = Knn(k)

    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)

    t_acc = np.sum(predictions == y_test) / len(y_test)
    if t_acc > acc:
        acc = t_acc
        best_k = k

print(acc)
print(best_k)

