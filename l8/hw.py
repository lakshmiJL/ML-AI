import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load the Titanic Dataset
url = "l8/titanic.csv"
titanic = pd.read_csv(url)

# Check the column names
print(titanic.columns)

# Preprocess the data
columns_to_drop = ['Ticket', 'Cabin', 'PassengerId']
columns_to_drop = [col for col in columns_to_drop if col in titanic.columns]
titanic.drop(columns_to_drop, axis=1, inplace=True)

# Convert categorical variables to numerical
titanic['Sex'] = titanic['Sex'].map({'male': 0, 'female': 1})
titanic['Embarked'] = titanic['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})

# Fill missing values with the mean
titanic.fillna(titanic.mean(), inplace=True)

# Perform PCA on the features
X = titanic.drop('Survived', axis=1)
y = titanic['Survived']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)

# Apply Logistic Regression to classify the passengers
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
