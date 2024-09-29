from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error

# Load Iris dataset
iris = load_iris()
X = iris.data  # Features
y = iris.target  # Target (species) - used for regression demonstration

# Let's use petal length and sepal length as features to predict petal width
# Choose features and target
X_regression = X[:, [0, 2]]  # Petal length and sepal length
y_regression = X[:, 3]  # Petal width as target

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_regression, y_regression, test_size=0.2, random_state=42)

# Multi-variable Linear Regression
multi_regression = LinearRegression()
multi_regression.fit(X_train, y_train)

# Predict on test set
y_pred_multi = multi_regression.predict(X_test)

# Calculate the Mean Squared Error (MSE) for multi-variable linear regression
mse_multi = mean_squared_error(y_test, y_pred_multi)
print(f"Mean Squared Error (Multi-variable Linear Regression): {mse_multi}")

# Polynomial Regression (degree = 2)
poly = PolynomialFeatures(degree=2)
X_poly_train = poly.fit_transform(X_train)
X_poly_test = poly.transform(X_test)

poly_regression = LinearRegression()
poly_regression.fit(X_poly_train, y_train)

# Predict on test set
y_pred_poly = poly_regression.predict(X_poly_test)

# Calculate the Mean Squared Error (MSE) for polynomial regression
mse_poly = mean_squared_error(y_test, y_pred_poly)
print(f"Mean Squared Error (Polynomial Regression - degree 2): {mse_poly}")
