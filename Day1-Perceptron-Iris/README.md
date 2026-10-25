# Day 1: Single Layer Perceptron on Iris

## Question

Design and implement a Single Layer Perceptron for classification of linearly separable datasets and visualize the decision boundary.

**Dataset:** Iris
**Classes used:** Setosa, Versicolor
**Attributes used:** Petal Length, Petal Width

Train a single perceptron and plot the decision boundary.

> **Note:** This is the minimal exam version. It trains the perceptron and prints the confusion matrix. The decision-boundary plot is left out on purpose.

## How to run

```
pip install scikit-learn
python perceptron_iris.py
```

Iris is built into scikit-learn, so no dataset file is needed.

## The code in three parts

### 1) Import the dataset

```python
iris = load_iris()
X = iris.data[:100, 2:4]
y = iris.target[:100]
```

- Iris has 150 flowers: rows 0–49 are **Setosa**, 50–99 **Versicolor**, 100–149 Virginica.
- `[:100]` keeps only the first 100 rows, so only Setosa (label 0) and Versicolor (label 1).
- The 4 columns are sepal length, sepal width, **petal length**, **petal width**. `2:4` picks columns 2 and 3 (the end index 4 is not included).

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

- 80% of the flowers train the model, 20% test it.
- `random_state=42` makes the split the same every run.

### 2) Train the perceptron

```python
model = Perceptron()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

**How a perceptron works:**

1. Multiply each input by a weight and add a bias: `z = w1·x1 + w2·x2 + b`
2. Step function: output **1** if `z ≥ 0`, else **0**.
3. If the prediction is wrong, adjust the weights: `w = w + learning_rate × (actual − predicted) × x`
4. Repeat over the data until there are no mistakes (or the max number of passes is reached).

`fit` does all of this; `predict` applies step 1 and 2 to new flowers.

### 3) Print the confusion matrix

```python
print(confusion_matrix(y_test, y_pred))
```

A 2×2 table: **rows = actual class, columns = predicted class.**

|  | Predicted Setosa | Predicted Versicolor |
|---|---|---|
| **Actual Setosa** | correct | mistakes |
| **Actual Versicolor** | mistakes | correct |

## Expected output

```
[[12  0]
 [ 0  8]]
```

All 20 test flowers are correct (100%). This is expected: Setosa and Versicolor are **linearly separable** on petal size, so one straight line divides them perfectly.

## Quick viva points

- **What is a perceptron?** The simplest neural network: one neuron with weights, a bias and a step activation.
- **What is linearly separable?** Two classes that can be split by one straight line (or a flat plane in higher dimensions).
- **What is the decision boundary?** The line where `w1·x1 + w2·x2 + b = 0`. Points on one side are class 0, on the other side class 1.
- **Limitation:** a single perceptron can only draw a straight line, so it **cannot solve XOR**. That's why Day 2 uses a multi-layer perceptron.
- **Perceptron convergence theorem:** if the data is linearly separable, the perceptron is guaranteed to find a separating line.
