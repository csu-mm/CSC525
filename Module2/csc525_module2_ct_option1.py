'''
MS - Artificial Intelligence and Machine Learning
Course: CSC525 - Principles of Machine Learning
Module 2: Critical Thinking Assignment
Professor: Dr. Dong Nguyen
Created by Mukul Mondal
May 21, 2026

Problem statement: 
Option #1: KNN Classifier with Iris Data
KNN cluster classification works by finding the distances between a query and all examples in its data. 
The specified number of examples (K) closest to the query are selected. The classifier then votes for the most frequent label found.

There are several advantages of KNN classification, one of them being simple implementation. 
The search space is robust as classes do not need to be linearly separable. 
It can also be updated online easily as new instances with known classes are presented.

A KNN model can be implemented using the following steps:

Load the data;
Initialize the value of k;
For getting the predicted class, iterate from 1 to the total number of training data points;
Calculate the distance between the test data and each row of training data;
Sort the calculated distances in ascending order based on distance values;
Get top k rows from the sorted array;
Get the most frequent class of these rows; and
Return the predicted class.
For your assignment, you will build a KNN classifier in Python.

Download the class data Links to an external site.in CSV format.

Your classifier should be able to, using this data, predict the type of iris based on the sepal 
length and width (the parts of the calyx) and the petal length and width, in centimeters.
'''

import os
from os import system, name

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


# Helper function.
# Clears the terminal
def clearScreen():
    if name == 'nt':  # For windows
        _ = system('cls')
    else:             # For mac and linux(here, os.name is 'posix')
        _ = system('clear')
    return

# reads input data file (iris.csv)
# create dataframe with this data
# creates NumPy array from features data
# creates NumPy array from label data
#    then returns these two NumPy array data
def dataPreprocess(dataFile: str):
    if dataFile is None or len(dataFile.strip()) < 1:
        return    

    # Load your CSV file
    df = pd.read_csv(dataFile)

    # Split into features (X) and labels (y)
    X = df.iloc[:, :4].values      # first 4 columns: sepal/petal measurements
    y = df.iloc[:, 4].values       # last column: species name or class name or label
    return X, y


# split train and test data at 80% and 20%
# creates classifier Model with k=3
# train the created classifier Model
#     then returns the classifier Model
def trainAndModelCreate(X, y):
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=42 )

    # Create default model and Train KNN model
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train, y_train)
    return knn

# does prediction on input data class name, based on the input model
#     then returns the predicted class name or label
def predict(knnModel, testSample):
    if knnModel is None or testSample is None or len(testSample) < 1:
        return
    return knnModel.predict(testSample)[0]

# takes user input for the "to be tested data"
# formats the user input in list.
#     then returns the user input list.
def ReadTestData() -> list:
    print('--- Please enter below the input data for testing ---')
    sepal_length: float = readFloatData("Enter Sepal Length: ") #5.1
    sepal_width: float = readFloatData("Enter Sepal Width: ") # 3.5
    petal_length: float = readFloatData("Enter Petal Length: ") #1.4
    petal_width: float = readFloatData("Enter Petal Width: ") # 0.4
    return [sepal_length, sepal_width, petal_length, petal_width]


# takes user input for the "to be tested data"
# this function helps the above function to take user input
def readFloatData(usrPrompt: str) -> float:
    if usrPrompt is None or len(usrPrompt.strip()) < 1:
        return
    usrInput: float = 0.0    
    redoInput: bool = True
    while redoInput:
        try:
            # Read user inputs and validate
            usrInput = float(input(f'{usrPrompt}'))
            redoInput = False  # valid input
        except ValueError:
            print('Please try again with numeric input.')
            redoInput = True
    return usrInput


# Application execution main entry point.
# it calls all above functions to execute the needed job for this project.
if __name__ == "__main__":
    clearScreen()
    print("Course: CSC525 - Principles of Machine Learning")
    print("Module 2: Critical Thinking Assignment")
    print("  Option #1: KNN Classifier with Iris Data\n")

    trainingDataFile: str = "./datafiles/Module2/iris.csv"
    x, y = dataPreprocess(trainingDataFile)
    knnModel = trainAndModelCreate(x, y)
    testData = ReadTestData()
    predictedClass = predict(knnModel, [testData])
    print(f"Predicted Class: {predictedClass}")
    

# --- needed pip installs ---
# (csc525) C:\Projs\Python\csc525>python.exe -m pip install --upgrade pip
# (csc525) C:\Projs\Python\csc525>pip install numpy pandas
# (csc525) C:\Projs\Python\csc525>pip install scikit-learn
#