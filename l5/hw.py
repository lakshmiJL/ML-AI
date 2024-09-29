import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
  
# fetch dataset 
data = pd.read_csv("l5/bank.csv")
  


from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()

data["age"] = label_encoder.fit_transform(data["age"])
data["education"] = label_encoder.fit_transform(data["education"])
data["job"] = label_encoder.fit_transform(data["job"])
data["loan"] = label_encoder.fit_transform(data["loan"])
data["housing"] = label_encoder.fit_transform(data["housing"])
# data (as pandas dataframes) 
X = data[["age", "education"]]
Y = data["housing"]
  
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, random_state = 2)

from sklearn.tree import DecisionTreeClassifier  
classifier= DecisionTreeClassifier(criterion='entropy', random_state=0)  
classifier.fit(X_train, Y_train)  

y_pred = classifier.predict(X_test)

from sklearn.tree import plot_tree

# Train the decision tree classifier
classifier = DecisionTreeClassifier(criterion='entropy', random_state=0)
classifier.fit(X_train, Y_train)



from sklearn.metrics import classification_report, confusion_matrix
matrix = confusion_matrix(Y_test, y_pred)

print(classification_report(Y_test, y_pred))
sns.heatmap(matrix, annot = True, fmt = "d")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

print(classification_report(Y_test, y_pred))