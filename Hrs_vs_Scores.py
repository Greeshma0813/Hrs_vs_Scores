//import pandas as pd                   # For data loading and handling (tables, csv files)
import numpy as np                     # For numerical operations (arrays, math functions)
import matplotlib.pyplot as plt        # For plotting graphs and visualizations
from sklearn.model_selection import train_test_split   # For splitting data into training and testing
from sklearn.linear_model import LinearRegression      # For creating a simple linear regression model
from sklearn import metrics            # For checking model accuracy (errors, score, etc.)
print('starting program')
file_path='F:/Python_Practice/Hrs_vs_Scores/Hrs_vs_Scores.csv'
# Read CSV and remove unwanted columns
data = pd.read_csv(file_path)

# Drop the 'Unnamed: 2' column
data = data.drop(columns=['Unnamed: 2'])

print(data.head())

import pandas as pd

# Read CSV and remove unwanted columns
data = pd.read_csv(file_path)

# Drop the 'Unnamed: 2' column
data = data.drop(columns=['Unnamed: 2'])

print(data.head(10))

import pandas as pd


# Read CSV and remove unwanted columns
data = pd.read_csv(file_path)

# Drop the 'Unnamed: 2' column
data = data.drop(columns=['Unnamed: 2'])

print(data)

# Read CSV and remove unwanted columns
data = pd.read_csv(file_path).dropna()

# Drop the 'Unnamed: 2' column
data = data.drop(columns=['Unnamed: 2'])

print(data)

# Read CSV and remove unwanted columns
data = pd.read_csv(file_path)

data = data.drop(columns=['Unnamed: 2'])
data=data.dropna()

print(data)

# Scatter plot to see relation between Hours and Scores
plt.scatter(data['Hours'], data['Scores'])
plt.title('Hours vs Percentage')
plt.xlabel('Hours Studied')
plt.ylabel('Percentage Score')
plt.show()

import matplotlib.pyplot as plt

# Customizing scatter plot
plt.scatter(data['Hours'], data['Scores'], color='green', marker='*', s=80)  # s controls the size of markers

# Adding title and labels
plt.title('Hours vs Scores (Customized)', fontsize=16, color='darkblue')  # Title customization
plt.xlabel('Hours Studied', fontsize=14, color='red')
plt.ylabel('Scores', fontsize=14, color='purple')

# Adding gridlines for better readability
plt.grid(True)

# Show plot
plt.show()

import matplotlib.pyplot as plt

# Example data (you can use your actual 'Scores' column from your dataset)
scores = data['Scores']

# Create a histogram
plt.hist(scores, bins=10, color='skyblue', edgecolor='black')

# Add title and labels
plt.title('Distribution of Scores', fontsize=16)
plt.xlabel('Scores', fontsize=14)
plt.ylabel('Frequency', fontsize=14)

# Show the plot
plt.show()

x=data[['Hours']]
y=data['Scores']
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2)

print("x_train:\n",x_train)
print("x_test:\n",x_test)
print("y_train:n",y_train)
print("y_tst:n",y_test)

from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Define X and y
X = data[['Hours']]  # make sure this matches your actual column names
y = data['Scores']

# Handle missing values using imputer
imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(X)  # Now X has no NaN

# Split the data
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(x_train, y_train)

print("Intercept (b):",model.intercept_)
print("Slope (m):",model.coef_[0])

print("Intercept (b):", model.intercept_)
print("Slope (m):", model.coef_[0])

y_pred = model.predict(x_test)

y_pred = model.predict(x_test)
print(y_pred)

import pandas as pd

results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(results)

comparison = pd.DataFrame({
    'Hours Studied': x_test.flatten(),  # if x_test is a NumPy array
    'Actual Score': y_test.values,
    'Predicted Score': y_pred
})
print(comparison)

plt.scatter(x_test, y_test, color='blue', label='Actual')
plt.plot(x_test, y_pred, color='red', label='Predicted Line')
plt.xlabel('Hours Studied')
plt.ylabel('Score')
plt.title('Actual vs Predicted Scores')
plt.legend()
plt.show()


