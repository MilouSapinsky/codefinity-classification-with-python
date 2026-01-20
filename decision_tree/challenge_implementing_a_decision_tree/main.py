import pandas as pd 
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
# Read the data and assign the variables
df = pd.read_csv('https://codefinity-content-media.s3.eu-west-1.amazonaws.com/b71ff7ac-3932-41d2-a4d8-060e24b00129/titanic.csv') 
X = df.drop(columns=['Survived'])
y = df['Survived']

# Write your code below
decision_tree = DecisionTreeClassifier()
param_grid = {'max_depth': [1,2,3,4,5,6,7,], 'min_samples_leaf': [1,2,4,6]}
grid_cv = GridSearchCV(estimator=decision_tree, param_grid=param_grid, cv=10).fit(X, y)

best_score = grid_cv.best_score_
best_model = grid_cv.best_estimator_

# Testing the result
print(best_score)
print(best_model)