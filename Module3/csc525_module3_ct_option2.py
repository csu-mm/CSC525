'''
MS - Artificial Intelligence and Machine Learning
Course: CSC525 - Principles of Machine Learning
Module 3: Critical Thinking Assignment
Professor: Dr. Dong Nguyen
Created by Mukul Mondal
May 26, 2026

Problem statement: 
Option #2: Simple Polynomial Regression in Python
Polynomial regression is a form of nonlinear regression that describes nonlinear relationships
in a dataset.

There are several advantages to linear regression, mainly high accuracy.

For your assignment, you will build a polynomial regression model in Python.

Please download the Years of experience and Salary datase. in CSV format:
( https://www.kaggle.com/datasets/rohankayan/years-of-experience-and-salary-dataset )

Using this data, our model should be able to predict the value of an employee candidate
given their years of experience.

Consider using Google to conduct your own research for this assignment. 
Feel free to also use the following documentation and resources:

Robust nonlinear regression in scipy Links to an external site.
Machine Learning - Polynomial RegressionLinks to an external site.
Submission should include an executable Python file demonstrating the prediction 
of employee salary based on years of experience.
'''

import os
from os import system, name
import math

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import sklearn
from sklearn.linear_model import LinearRegression
from scipy.interpolate import make_interp_spline


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
# remove duplicate 'YearsExperience' and keep max value item for 'Salary'
# Sort dataframe in ascending order based on 'YearsExperience'
#    then returns these two colums as input and output data
def dataLoadAndPrepare(dataFile: str):
    if dataFile is None or len(dataFile.strip()) < 1:
        return    

    # Load your CSV file
    try:        
        df = pd.read_csv(dataFile)
    except Exception as e:
        print("data file load Error", e)
        exit(1)

    # Sample data (Salary_Data.csv) downloaded from the link
    # YearsExperience,  Salary
    # 1.1,              39343.00
    # 1.3,              46205.00
    #
    
    # remove any duplicate 'YearsExperience' and keep the 'higher Salary' item
    df = df.sort_values('Salary', ascending=False).drop_duplicates('YearsExperience')
    
    # inPlace sort dataframe with 'YearsExperience' ascending
    df.sort_values('YearsExperience', ascending=True, inplace=True)

    # Split into features (X) and labels (y)
    X = df.iloc[:, :1].values      # first column: YearsExperience
    y = df.iloc[:, 1].values       # column having index = 1: Salary
    return X, y


# This fuinction creates the LinearRegression model and Poly regression model
# Train these models
# creates graphs showing these models performance
#     then returns the Poly classifier Model for future use by the caller.
def model_create_train_predict_plot(X, y, polyDegree):
    if X is None or X.size == 0:
        return
    if y is None or y.size == 0:
        return
    if polyDegree < 1:
        return
    
    # Simple Linear Regression Model for comparison
    lin_reg = LinearRegression()
    lin_reg.fit(X,y) 

    # Visualization # ok
    # plt.scatter(X, y, color='red')
    # plt.plot (X, lin_reg.predict(X), color='blue')
    # plt.title("Linear Regression")
    # plt.xlabel('Level')
    # plt.ylabel('Salary')
    # plt.show()

    # Polynomial Regression
    poly_fts = sklearn.preprocessing.PolynomialFeatures(degree=polyDegree)
    X_poly = poly_fts.fit_transform(X)
    poly_reg = LinearRegression()
    poly_reg.fit(X_poly, y)

    # Calculate RMSE
    lin_preds = lin_reg.predict(X)
    poly_preds = poly_reg.predict(X_poly)
    
    rmse = {"linear": np.sqrt(sklearn.metrics.mean_squared_error(y,lin_preds)),
            "polynomial": np.sqrt(sklearn.metrics.mean_squared_error(y, poly_preds))}

    # Smooth Polynomial Regression Graph    
    spline = make_interp_spline(np.array(np.reshape(X,len(X))), poly_preds)    
    x_spline = np.linspace(np.array(X).min(), np.array(X).max(), 500)
    y_spline = spline(x_spline)

     # Plot Graphs
    f, axes = plt.subplots(1, 2, sharex=False, sharey=False, figsize=(12,7))

    axes[0].scatter(X, y, color='red', label='Data')
    axes[0].plot(X, lin_preds, color='blue', label='Predictions')
    axes[0].legend()
    axes[0].set_title("Linear Regression\nRMSE: {}".format(rmse['linear']))
    axes[0].set_ylabel("Salary")
    axes[0].set_xlabel("Level")

    axes[1].scatter(X, y, color='red', label='Data')
    axes[1].plot(x_spline, y_spline, color='blue', label='Predictions')
    axes[1].legend()
    axes[1].set_title("Polynomial Regression (Poly Degree: {})\nRMSE: {}".format(polyDegree, rmse['polynomial']))
    axes[1].set_ylabel("Salary")
    axes[1].set_xlabel("Level")

    plt.show()

    return poly_reg


# This function predicts the salary based on the YearsExperience.
# Caller also has to suppy the Poly degree with which the model was created.
def predictPoly(polyRegModel, input_yrsExp, polyDegree):
    if polyRegModel is None:
        return
    if input_yrsExp <= 0:
        return

    poly_fts = sklearn.preprocessing.PolynomialFeatures(degree=polyDegree)
    X_poly = poly_fts.fit_transform(np.array([[input_yrsExp]]))
    return polyRegModel.predict(X_poly)


# Helper function.
# takes user input for the "to be tested data"
# this function helps the above function to take user input
def readNumericInput(usrPrompt: str) -> float:
    if usrPrompt is None or len(usrPrompt.strip()) < 1:
        return
    usrInput: int = 0
    redoInput: bool = True
    while redoInput:
        try:
            # Read user inputs and validate
            usrInput = float(input(f'{usrPrompt}'))
            if usrInput >= 1:
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
    print("Module 3: Critical Thinking Assignment")
    print("   Option #2: Simple Polynomial Regression in Python\n")

    # locally downloaded and saved .csv data file
    # Please update the path or filename for your environment.
    trainingDataFile: str = "./datafiles/Module3/Salary_Data.csv"
    if os.path.exists(trainingDataFile) == False:
        print("Data file not found.")
        print("Please check and update Data file path.")
        exit(1)

    # read user input for Poly degree
    polyDegree: int = math.floor(readNumericInput("Please enter polynomial degree (>1) for Polynomial Regression: "))
   
    x, y = dataLoadAndPrepare(trainingDataFile) # load and prepare data for further processing

    # create regression models and show their performance
    polyRegModel = model_create_train_predict_plot(x, y, polyDegree)
    
    # take user input for prediction
    print("\nPlease note the curvatures at the end-points if you want to make predictions beyond the input range.")
    print(f"Input data 'x' range: {x.min()} - {x.max()}")
    yrsExp: float = readNumericInput("\nPrediction:\n  Please enter Years of Experience: ")

    # do prediction and show the result(predicted salary)
    pred = predictPoly(polyRegModel, yrsExp, polyDegree)
    print(f"  Predicted Salary: {pred[0]} for YrsOfExperience: {yrsExp}")



# --- needed pip installs ---
# (csc525) C:\Projs\Python\csc525>python.exe -m pip install --upgrade pip
# (csc525) C:\Projs\Python\csc525>pip install numpy pandas
# (csc525) C:\Projs\Python\csc525>pip install scikit-learn
# (csc525) C:\Projs\Python\csc525>pip install matplotlib
#