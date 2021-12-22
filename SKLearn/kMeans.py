# python 3.7
# Scikit-learn ver. 0.23.2
import sklearn.cluster
import sklearn.datasets
import sklearn.preprocessing
# numpy 1.19.1
from numpy import mean

# scikit-learn's built-in version of the MNIST handwritten digits
# colors are 4-bit grayscale (values 0-16), 8x8 pixels
digits = sklearn.datasets.load_digits()
# Get the images out of the data blob
digitsX = digits.images
# Reshape each from 8x8 to single-column, length 64
digitsX = digitsX.reshape((len(digitsX), 64))
# Get the target values, meaning the digit that the picture represents
digitsY = digits.target

# Using k-means clustering as the classifier
estimator = sklearn.cluster.KMeans(
    init="k-means++",
    n_clusters = 10,
    n_init = 10,
)

# The sample code in scikit-learn's documentation scales the data this way
# This linearly scales the data by subtracting the mean and dividing by the
#   standard deviation, which gives a (not necessariluy normal) distribution
#   with mean 0, s.d. 1.
scaledData = sklearn.preprocessing.scale(digitsX)
# Choose one of the two pairs of lines below to use scaled/unscaled data.

# estimator.fit(scaledData)
# print("Using scaled data for estimator")
estimator.fit(digitsX)
print("Using unscaled data for estimator")

# Make predictions using the estimator
predY = estimator.predict(digitsX)

# Analyzing two different pixels:
#  -third pixel over across the top (row 0, column 2, or entry 2 in single-col)
#  -third pixel down the left side (row 2, column 0, or entry 16 in single-col)

# First get the unscaled values (in range 0-16)
vals2 = []
vals16 = []
for digit in digitsX:
    vals2.append(digit[2])
    vals16.append(digit[16])
# Then get the scaled values (in a distribution with mean 0, s.d. 1)
scaledVals2 = []
scaledVals16 = []
for digit in scaledData:
    scaledVals2.append(digit[2])
    scaledVals16.append(digit[16])

# Output what we find about the unscaled/scaled values
print(f"Third pixel across the top, unscaled: Mean: {mean(vals2):.4}, Min: {min(vals2)}, Max: {max(vals2)}")
print(f"Third pixel down the left, unscaled: Mean: {mean(vals16):.4}, Min: {min(vals16)}, Max: {max(vals16)}")
print(f"Third pixel across the top, scaled: Min: {min(scaledVals2):.4}, Max: {max(scaledVals2):.4}")
print(f"Third pixel down the left, scaled: Min: {min(scaledVals16):.4}, Max: {max(scaledVals16):.4}")

# PROBLEM: Because k-means is a clustering algorithm, not a classifier,
#   the cluster labels are not lined up with the data labels -- e.g.
#   the third cluster is not likely to be the third label (2).  So, we
#   have to decide which cluster corresponds to which label.
# SOLUTION: Go through and count, for each cluster, which digit is most
#   likely to fall in that cluster.  Then, associate that cluster with
#   that digit.
# ASSUMPTION: This will actually result in a single distinct digit assigned
#   to each cluster; i.e. that the same digit isn't most common in two
#   clusters and/or two digits aren't both tied for most common and/or
#   that each digit is actually most common in some cluster.

# After the loop runs, the i-th entry in this is the digit for the i-th cluster.
finalPreds = [0 for i in range(10)]
# For each label, count how many digits with this label fall in this cluster.
for label in range(10):
    counts = [0 for i in range(10)]
    for pred, gt in zip(predY, digitsY):
        if pred == label:
            counts[gt] += 1
    # Assign this label to the cluster that has the most of this label.
    finalPreds[label] = counts.index(max(counts))
# Go through all the predictions and change cluster numbers to digit labels.
for i in range(len(predY)):
    predY[i] = finalPreds[predY[i]]

# Now, see how accurate we were.
correct = 0
incorrect = 0
for pred, gt in zip(predY, digitsY):
    if pred == gt: correct += 1
    if pred != gt: incorrect += 1
print(f"Num correct: {correct}, Num incorrect: {incorrect}")
print(f"Proportion: {correct/len(predY):5.2} correct")
