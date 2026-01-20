import pandas as pd 
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
# Read the data and assign the variables
df = pd.read_csv('https://codefinity-content-media.s3.eu-west-1.amazonaws.com/b71ff7ac-3932-41d2-a4d8-060e24b00129/titanic.csv')
X = df.drop('Survived', axis=1)
y = df['Survived']

# Write your code below
random_forest = RandomForestClassifier(n_estimators=100, max_depth=None, max_features='sqrt', min_samples_leaf=1, random_state=42).fit(X, y)
cv_scores = cross_val_score(random_forest, X, y, cv=10)

print(cv_scores.mean())

for feature in zip(X.columns, random_forest.feature_importances_):
  print(feature)