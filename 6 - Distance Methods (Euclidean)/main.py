from math import sqrt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

def euclidean_distance(a, b):
    return sqrt(sum((el - e2) ** 2 for el, e2 in zip(a, b)))

def manhattan_distance(a, b):
    return sum(abs(el - e2) ** 2 for el, e2 in zip(a, b))

def minkowski_distance(a, b, p):
    return sum(abs(el - e2) ** p for el, e2 in zip(a, b)) ** (1 / p)

actual = [1, 0, 0, 1, 0, 0, 1, 0, 0, 1]
predicted = [1, 0, 0, 1, 0, 0, 1, 0, 0, 1]

dist_euclidean = euclidean_distance(actual, predicted)
dist_manhattan = manhattan_distance(actual, predicted)
dist_minkowski_1 = minkowski_distance(actual, predicted, 1)
dist_minkowski_2 = minkowski_distance(actual, predicted, 2)

print(f"Euclidean distance: {dist_euclidean}")
print(f"Manhattan distance: {dist_manhattan}")
print(f"Minkowski distance (p=1): {dist_minkowski_1}")
print(f"Minkowski distance (p=2): {dist_minkowski_2}")

matrix = confusion_matrix(actual, predicted, labels=[1, 0])
print("Confusion matrix:\n", matrix)

tp, fn, fp, tn = confusion_matrix(actual, predicted, labels=[1, 0]).reshape(-1)
print("Outcome values:\n", tp, fn, fp, tn)

report = classification_report(actual, predicted, labels=[1, 0])
print("Classification report:\n", report)
