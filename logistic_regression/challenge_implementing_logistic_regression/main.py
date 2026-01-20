import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

df = pd.read_csv('https://codefinity-content-media.s3.eu-west-1.amazonaws.com/b71ff7ac-3932-41d2-a4d8-060e24b00129/marketing_bank.csv')
X = df.drop('deposit', axis=1).values
y = df['deposit'].values

# Write your code below
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

lr = LogisticRegression().fit(X_train, y_train)
test_accuracy = lr.score(X_test, y_test)

# Testing the result
print(f'Test set accuracy: {test_accuracy:.3f}')