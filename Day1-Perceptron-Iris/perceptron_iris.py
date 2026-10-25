from sklearn.datasets import load_iris
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

# 1) Import dataset: first 100 rows = Setosa (0) + Versicolor (1)
#    columns 2 and 3 = petal length, petal width
iris = load_iris()
X = iris.data[:100, 2:4]
y = iris.target[:100]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2) Train single layer perceptron
model = Perceptron()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# 3) Confusion matrix
print(confusion_matrix(y_test, y_pred))
