import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
# Load data from CSV file
data = pd.read_csv('data.csv')

# Extract X and Y columns
X = data['X'].values.reshape(-1, 1)
Y = data['Y'].values

# Create and fit the Linear Regression model (same as above)
model = LinearRegression()
model.fit(X, Y)

# Get the slope (coefficient) and intercept of the fitted line
slope = model.coef_[0]
intercept = model.intercept_

print(f"The equation of the fitted line: Y = {slope} * X + {intercept}")