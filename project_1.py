import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import log_loss

# Verilerin yüklenmesi
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
sample_submission = pd.read_csv('sample_submission.csv')

# Eksik verilerin doldurulması
# Sayısal sütunları medyan ile doldur
numerical_cols = train.select_dtypes(include=['float64']).columns
for col in numerical_cols:
    train[col] = train[col].fillna(train[col].median())
    if col in test.columns:
        test[col] = test[col].fillna(train[col].median())

# Fill categorical columns with mode
categorical_cols = train.select_dtypes(include=['object']).columns
for col in categorical_cols:
    train[col] = train[col].fillna(train[col].mode()[0])
    if col in test.columns:
        test[col] = test[col].fillna(train[col].mode()[0])


# Kategorik değişkenlerin kodlanması
label_encoders = {}
for col in ['Drug', 'Sex', 'Ascites', 'Hepatomegaly', 'Spiders', 'Edema', 'Status']:
    le = LabelEncoder()
    train[col] = le.fit_transform(train[col])
    if col in test.columns:
        test[col] = le.transform(test[col])
    label_encoders[col] = le

# Özellikler ve hedef değişken ayrımı
X = train.drop(columns=['id', 'Status'])
y = train['Status']

# Eğitim ve doğrulama setlerine ayırma
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Verilerin ölçeklendirilmesi
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(test.drop(columns=['id']))

# Optimize edilmiş Gradient Boosting modeli
optimized_gbc = GradientBoostingClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42
)

# Modelin eğitilmesi
optimized_gbc.fit(X_train, y_train)

# Doğrulama setinde tahmin
y_val_pred_proba = optimized_gbc.predict_proba(X_val)
val_log_loss = log_loss(y_val, y_val_pred_proba)
print(f"Validation Log Loss: {val_log_loss}")

# Test setinde tahmin ve submission dosyasının hazırlanması
y_test_pred_proba = optimized_gbc.predict_proba(X_test)
submission = sample_submission.copy()
submission[['Status_C', 'Status_CL', 'Status_D']] = y_test_pred_proba
submission.to_csv('submission.csv', index=False)
print("Submission dosyası kaydedildi: submission.csv")
