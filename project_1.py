import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss
from sklearn.preprocessing import LabelEncoder

# data setleri yüklüyoruz
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

print("Status Sütunu Değerleri:")
print(train['Status'].value_counts())

train = train[train['Status'].isin(['C', 'CL', 'D'])]
y = train['Status'].map({'C': 0, 'CL': 1, 'D': 2})

print("y'deki NaN Sayısı:", y.isnull().sum())

categorical_cols = train.select_dtypes(include=['object']).columns
encoder = LabelEncoder()
for col in categorical_cols:
    train[col] = encoder.fit_transform(train[col])
    if col in test.columns:
        test[col] = encoder.transform(test[col])

X = train.drop(columns=['id', 'Status'])
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict_proba(X_val)
print("Log Loss:", log_loss(y_val, y_pred))

X_test = test.drop(columns=['id'])
test_preds = model.predict_proba(X_test)

submission = pd.DataFrame(test_preds, columns=['Status_C', 'Status_CL', 'Status_D'])
submission['id'] = test['id']
submission.to_csv("submission.csv", index=False)
print("Gönderim dosyası oluşturuldu: submission.csv")
