# python 3.7
# Scikit-learn ver. 0.23.2
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import fetch_species_distributions
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix
# matplotlib 3.3.1
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

species_distr = fetch_species_distributions()
species_x = species_distr.data
species_y = species_distr.target

scaler = StandardScaler()
species_x = scaler.fit_transform(species_x)
species_y = species_y.reshape(-1, 1)

print(f"species_x: {species_x}")
print(f"species_y: {species_y}")

trainX, testX, trainY, testY = train_test_split(
    species_x, species_y, test_size = 0.3, shuffle = True
    )

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
plt.pyplot.show()

# doesn't work for some reason - error: certificant verify failed: unavle to get local issuer certificate

# x = images
# y = labels