# imports for image contour detection
import cv2

# imports for sci-kit learn 
from sklearn.linear_model import HuberRegressor, LogisticRegression
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix
from sklearn.preprocessing import StandardScaler # scalar normalization

# imports for reading data and graphing
from matplotlib import pyplot
import os, os.path
import pandas as pd
import numpy as np

def load_test_data(path):
    '''
    param: path to asl alphabet test folder
    return: list of files in the folder
    '''
    img_list = []
    for file in os.listdir(path):
        if file.endswith(".jpg"):
            img_list.append(file)

    img_list = sorted(img_list)
    return img_list


def load_train_data(path):
    '''
    param: path to asl alphabet test folder
    return: list of files in the folder
    '''
    img_list = []
    folder_list = sorted(os.listdir(path))
    for item in folder_list:
        print("item:", item)
        item_list = []

        if item != '.DS_Store' and item != 'modify_filename.py':
            for file in os.listdir(path + item):
                if file.endswith(".jpg"):
                    item_list.append(file)

        img_list.append(sorted(item_list))

    return img_list

def get_contours(path):
    '''
    param: path for image
    return: canny image of the image
    '''
    # print(path)
    image = cv2.imread(path)
    #cv2.imshow("path:" + path, image)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    canny = cv2.Canny(blurred, 30, 150)
    cv2.imshow("Canny Edge Detection", canny)
    
    canny.resize(64, 64)
    return canny


def get_logistic_regression(trainX, trainY, testX, testY):
    # scaler = StandardScaler()

    # scaler.fit(trainX)
    # trainX = scaler.transform(trainX)
    # scaler.fit(testX)
    # testX = scaler.transform(testX)
    print("trainX.shape: ", trainX.shape)
    trainX = np.reshape(trainX, (-1, 4096))
    print("trainX.shape: ", trainX.shape)
    testX = np.reshape(testX, (-1, 4096))

    classifier = LogisticRegression(max_iter = 50000) #lower to get faster time, raise to get higher accuracy
    print("Length of trainX[0]: ", len(trainX[0]))
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

def main():
    # save files of test/train photos to list
    asl_alphabet_test = []
    asl_alphabet_train  = []
    alphabet = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

    # variables for path to test/train folders
    asl_alphabet_test_path = '/Users/emmal/Documents/Scikit_Learn/LearnASL/archive/asl_alphabet_test/'
    asl_alphabet_train_path = '/Users/emmal/Documents/Scikit_Learn/LearnASL/archive/asl_alphabet_train/'

    # load sorted training data using load_files()
    asl_alphabet_test = load_test_data(asl_alphabet_test_path)
    asl_alphabet_train = load_train_data(asl_alphabet_train_path) # list of lists

    # get rid of two extra folders at the beginning and end
    asl_alphabet_train = asl_alphabet_train[1:-1]

    # get canny images for each letter in testing set (single list with 26 images)
    asl_alphabet_test_img_list = []
    for img in asl_alphabet_test:
        canny_img = get_contours(asl_alphabet_test_path + img)
        asl_alphabet_test_img_list.append(canny_img)

    print(asl_alphabet_test_img_list)

    # get canny images for each letter in training set (single list with 78000 images)
    asl_alphabet_train_img_list = []
    counter = 0
    for folder in asl_alphabet_train:
        print(len(folder))
        for img in folder:
            # print(asl_alphabet_train_path + alphabet[counter] + '/' + str(img))
            canny_img = get_contours(asl_alphabet_train_path + alphabet[counter] + '/' + str(img))
            asl_alphabet_train_img_list.append(canny_img)

        counter += 1
    
    print(len(asl_alphabet_test_img_list))
    print(len(alphabet))
    print(len(asl_alphabet_train_img_list))

    # make them np arrays to flatten
    asl_alphabet_test_img_list = np.array(asl_alphabet_test_img_list)
    alphabet = np.array(alphabet)
    asl_alphabet_train_img_list = np.array(asl_alphabet_train_img_list)

    # get rid of the 255 values
    # asl_alphabet_test_img_list = asl_alphabet_test_img_list/255
    # print(asl_alphabet_test_img_list)
    # asl_alphabet_test_img_list = asl_alphabet_test_img_list/255

    # create testY (labels) for testing set and train data
    training_labels = ['A'] * 3000 + ['B'] * 3000 + ['C'] * 3000 + ['D'] * 3000 + ['E'] * 3000 + ['F'] * 3000 + ['G'] * 3000 + ['H'] * 3000 + ['I'] * 3000 + ['J'] * 3000 + ['K'] * 3000 + ['L'] * 3000 + ['M'] * 3000 + ['N'] * 3000 + ['O'] * 3000 + ['P'] * 3000 + ['Q'] * 3000 + ['R'] * 3000 + ['S'] * 3000 + ['T'] * 3000 + ['U'] * 3000 + ['V'] * 3000 + ['W'] * 3000 + ['X'] * 3000 + ['Y'] * 3000 + ['Z'] * 3000
    training_labels = np.array(training_labels)
    print(len(training_labels))
    get_logistic_regression(asl_alphabet_test_img_list, alphabet, asl_alphabet_train_img_list, training_labels)


if __name__ == "__main__":
    main()