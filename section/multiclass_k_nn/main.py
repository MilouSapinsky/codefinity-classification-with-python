from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings('ignore')

df = pd.read_csv('https://codefinity-content-media.s3.eu-west-1.amazonaws.com/b71ff7ac-3932-41d2-a4d8-060e24b00129/starwars_multiple.csv')

X_train = df.drop('StarWars6', axis=1)
y_train = df['StarWars6']
X_test = np.array([[5, 5], [4.5, 4]])

# Write your code below
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

knn = KNeighborsClassifier(n_neighbors=13).fit(X_train, y_train)
y_pred = knn.predict(X_test)

# Testing the result
print(y_pred)