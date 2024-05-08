import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
iris = pd.read_csv("https://raw.githubusercontent.com/amankharwal/Website-data/master/IRIS.csv")
print(iris.head())
print("-------------------------------------------------------------------------------------------------------")
print(iris.describe())
print("-------------------------------------------------------------------------------------------------------")
print("Target Labels", iris["species"].unique())
print("-------------------------------------------------------------------------------------------------------")
x = iris.drop("species", axis=1)
y = iris["species"]
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(x_train, y_train)
x_new = np.array([[7, 3.9, 4.1, 1.2]])
prediction = knn.predict(x_new)
print(prediction)
print("-------------------------------------------------------------------------------------------------------")