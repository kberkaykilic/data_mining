import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss
from sklearn.preprocessing import LabelEncoder

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

print("Status columns values:")
print(train['Status'].value_counts())

train = train[train['Status'].isin(['C', 'CL', 'D'])]
y = train['Status'].map({'C': 0, 'CL': 1, 'D': 2})
print("NaN numbers in y:", y.isnull().sum())

categorical_cols = train.select_dtypes(include=['object']).columns
encoder = LabelEncoder()
for col in categorical_cols:
    train[col] = encoder.fit_transform(train[col])
    if col in test.columns:
        test[col] = encoder.transform(test[col])

X = train.drop(columns=['id', 'Status'])
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=41)

hyperparameters = [
    {'n_estimators': 100, 'max_depth': 10, 'min_samples_split': 2, 'min_samples_leaf': 1},
    {'n_estimators': 200, 'max_depth': 20, 'min_samples_split': 5, 'min_samples_leaf': 2},
    {'n_estimators': 150, 'max_depth': None, 'min_samples_split': 10, 'min_samples_leaf': 4},
]

best_log_loss = float('inf')
best_params = None

for params in hyperparameters:
    model = RandomForestClassifier(random_state=41, **params)
    model.fit(X_train, y_train)
    y_pred = model.predict_proba(X_val)
    current_log_loss = log_loss(y_val, y_pred)
    print(f"Params: {params}, Log Loss: {current_log_loss}")
    if current_log_loss < best_log_loss:
        best_log_loss = current_log_loss
        best_params = params

print("Best Hyperparameters:", best_params)
print("Best Log Loss:", best_log_loss)

best_model = RandomForestClassifier(random_state=41, **best_params)
best_model.fit(X_train, y_train)

X_test = test.drop(columns=['id'])
test_preds = best_model.predict_proba(X_test)

submission = pd.DataFrame(test_preds, columns=['Status_C', 'Status_CL', 'Status_D'])
submission['id'] = test['id']
submission.to_csv("submission.csv", index=False)
print("submission.csv saved")
