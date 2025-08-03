import pandas as pd                   # For data loading and handling (tables, csv files)
import numpy as np                     # For numerical operations (arrays, math functions)
import matplotlib.pyplot as plt        # For plotting graphs and visualizations
from sklearn.model_selection import train_test_split   # For splitting data into training and testing
from sklearn.linear_model import LinearRegression      # For creating a simple linear regression model
from sklearn import metrics            # For checking model accuracy (errors, score, etc.)
print('starting pgm')
# Read CSV and remove unwanted columns
data = pd.read_csv('F:/Python/Hrs_vs_Scores.csv')

# Drop the 'Unnamed: 2' column
data = data.drop(columns=['Unnamed: 2'])

print(data.head())

