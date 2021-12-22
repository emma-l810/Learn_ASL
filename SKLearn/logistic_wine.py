# python 3.7
# Scikit-learn ver. 0.23.2
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix
# matplotlib 3.3.1
from matplotlib import pyplot
from sklearn.preprocessing import StandardScaler # scalar normalization

wine = load_wine()
wineX = wine.data   # similar to digitx (but easier)
wineY = wine.target # similar to digity (but easier)

# log wineX and wineY
scaler = StandardScaler()
wineX = scaler.fit_transform(wineX)
wineY = wineY.reshape(-1, 1)

# print/check the 
print(f"wineX:{wineX}")
# print(f"wineY:{wineY}")
print(f"featurenames:{wine.feature_names}")
print(f"targetnames:{wine.target_names}")

trainX, testX, trainY, testY = train_test_split(
    wineX, wineY, test_size = 0.3, shuffle = True
    )

# scale train and test sets
scaler.fit(trainX)
trainX = scaler.transform(trainX)
scaler.fit(testX)
testX = scaler.transform(testX)

classifier = LogisticRegression(max_iter = 10000) #lower to get faster time, raise to get higher accuracy
classifier.fit(trainX, trainY)
preds = classifier.predict(testX)

correct = 0
incorrect = 0
for pred, gt in zip(preds, testY):
    if pred == gt: correct += 1
    else: incorrect += 1
print(f"Correct: {correct}, Incorrect: {incorrect}, % Correct: {correct/(correct + incorrect): 5.2}")

plot_confusion_matrix(classifier, testX, testY)
pyplot.show()
