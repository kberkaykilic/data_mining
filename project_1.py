import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold
import xgboost as xgb

# Load data
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
sample_submission = pd.read_csv('sample_submission.csv')

# Handle missing values
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

# Encode categorical variables
label_encoders = {}
for col in ['Drug', 'Sex', 'Ascites', 'Hepatomegaly', 'Spiders', 'Edema', 'Status']:
    le = LabelEncoder()
    train[col] = le.fit_transform(train[col])
    if col in test.columns:
        test[col] = le.transform(test[col])
    label_encoders[col] = le

# Features and target
X = train.drop(columns=['id', 'Status'])
y = train['Status']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(test.drop(columns=['id']))

# Cross-validation with a slightly optimized XGBoost
kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
log_loss_values = []
final_predictions = np.zeros((X_test_scaled.shape[0], len(np.unique(y))))

for train_idx, val_idx in kf.split(X_scaled, y):
    X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    model = xgb.XGBClassifier(
        learning_rate=0.03,  # Slightly reduced learning rate
        max_depth=5,
        n_estimators=500,  # Moderate number of estimators
        subsample=0.8,
        colsample_bytree=0.8,
        objective='multi:softprob',
        eval_metric='mlogloss',
        random_state=42
    )

    model.fit(X_train, y_train)

    val_pred = model.predict_proba(X_val)
    log_loss_val = log_loss(y_val, val_pred)
    log_loss_values.append(log_loss_val)

    final_predictions += model.predict_proba(X_test_scaled) / kf.get_n_splits()

print(f"Average Log Loss (Cross-Validation): {np.mean(log_loss_values)}")

# Prepare submission
submission = sample_submission.copy()
submission[['Status_C', 'Status_CL', 'Status_D']] = final_predictions
submission.to_csv('submission_xgb_optimized_safe.csv', index=False)
print("Submission file saved as: submission_xgb_optimized_safe.csv")
