import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from scipy.stats import uniform, randint
import xgboost as xgb

"""IF THE CODE DOES'NT RUN PLEASE TRY scikit-learn's 1.5.1 version (change in python interpreter)"""
"""PROCESS WILL FINISH IN A FIVE MINS. AROUND I HOPE SO :D PLS BE PATIENT"""
"""Kemal Berkay Kilic (1904290) and Toprak Taybara (1902756)"""

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
sample_submission = pd.read_csv('sample_submission.csv')

numerical_cols = train.select_dtypes(include=['float64']).columns
for col in numerical_cols:
    train[col] = train[col].fillna(train[col].median())
    if col in test.columns:
        test[col] = test[col].fillna(train[col].median())

categorical_cols = train.select_dtypes(include=['object']).columns
for col in categorical_cols:
    train[col] = train[col].fillna(train[col].mode()[0])
    if col in test.columns:
        test[col] = test[col].fillna(train[col].mode()[0])

label_encoders = {}
for col in ['Drug', 'Sex', 'Ascites', 'Hepatomegaly', 'Spiders', 'Edema', 'Status']:
    le = LabelEncoder()
    train[col] = le.fit_transform(train[col])
    if col in test.columns:
        test[col] = le.transform(test[col])
    label_encoders[col] = le

X = train.drop(columns=['id', 'Status'])
y = train['Status']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(test.drop(columns=['id']))

param_dist = {
    'learning_rate': uniform(0.001, 0.05),
    'max_depth': randint(3, 10),
    'n_estimators': randint(500, 1500),
    'subsample': uniform(0.7, 0.3),
    'colsample_bytree': uniform(0.7, 0.3),
    'gamma': uniform(0, 0.5),
    'reg_alpha': uniform(0, 1),
    'reg_lambda': uniform(0, 1)
}

random_search = RandomizedSearchCV(
    estimator=xgb.XGBClassifier(objective='multi:softprob', eval_metric='mlogloss', random_state=42),
    param_distributions=param_dist,
    n_iter=50,
    scoring='neg_log_loss',
    cv=5,
    verbose=1,
    random_state=42,
    n_jobs=-1
)

random_search.fit(X_scaled, y)

print(f"best parameters: {random_search.best_params_}")
print(f"best log loss: {-random_search.best_score_}")

best_model = random_search.best_estimator_
test_predictions = best_model.predict_proba(X_test_scaled)

submission = sample_submission.copy()
submission[['Status_C', 'Status_CL', 'Status_D']] = test_predictions

submission.to_csv('submission.csv', index=False)
print("submission file saved as: submission.csv")

"""IF THE CODE DOES'NT RUN PLEASE TRY scikit-learn's 1.5.1 version (change in python interpreter)"""
"""PROCESS WILL FINISH IN A FIVE MINS. AROUND I HOPE SO :D PLS BE PATIENT"""
"""Kemal Berkay Kilic (1904290) and Toprak Taybara (1902756)"""
