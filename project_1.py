import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import KNNImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import SMOTE

# Load training and test datasets
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# Display value counts for the 'Status' column in the training dataset
print("Status value counts:")
print(train_df['Status'].value_counts())


# Function to preprocess data (handles missing values, encoding, and scaling)
def preprocess_data(df, is_train=True):
    df = df.copy()

    # Convert categorical variables to numeric values
    categorical_cols = ['Drug', 'Sex', 'Ascites', 'Hepatomegaly', 'Spiders', 'Edema']

    for col in categorical_cols:
        # Fill missing categorical values with mode or a placeholder
        df[col] = df[col].fillna(df[col].mode()[0] if len(df[col].dropna()) > 0 else 'Missing')
        # Encode categorical values as numeric
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

    # List of numeric columns to process
    numeric_cols = ['N_Days', 'Age', 'Bilirubin', 'Cholesterol', 'Albumin', 'Copper',
                    'Alk_Phos', 'SGOT', 'Tryglicerides', 'Platelets', 'Prothrombin', 'Stage']

    # Use KNN Imputer to fill missing numeric values
    knn_imputer = KNNImputer(n_neighbors=5)
    df[numeric_cols] = knn_imputer.fit_transform(df[numeric_cols])

    # Standardize numeric features using StandardScaler
    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    return df


# Preprocess the training data
X_train = preprocess_data(train_df.drop(['Status', 'id'], axis=1))
y_train = train_df['Status']

# Encode target variable as numeric
le_status = LabelEncoder()
y_train = le_status.fit_transform(y_train)

# Balance the training dataset using SMOTE
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

# Define the parameter grid for hyperparameter tuning of the RandomForest model
param_grid = {
    'n_estimators': [100, 200],  # Number of trees in the forest
    'max_depth': [10, 20],  # Maximum depth of the tree
    'min_samples_split': [2, 5],  # Minimum number of samples required to split an internal node
    'min_samples_leaf': [1, 2],  # Minimum number of samples required to be at a leaf node
    'max_features': ['sqrt'],  # Number of features to consider when looking for the best split
    'bootstrap': [True]  # Whether bootstrap samples are used when building trees
}

# Initialize RandomForestClassifier
rf = RandomForestClassifier(random_state=42)

# Perform grid search for hyperparameter tuning using cross-validation
grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=3,  # Number of folds in cross-validation
    scoring='neg_log_loss',  # Scoring metric for evaluation
    n_jobs=-1  # Use all available processors
)

# Fit the grid search to the balanced training data
grid_search.fit(X_train_balanced, y_train_balanced)

# Retrieve the best model and parameters
best_model = grid_search.best_estimator_
best_params = grid_search.best_params_

# Calculate the log loss of the best model
best_log_loss = -grid_search.best_score_
print(f"Best Log Loss: {best_log_loss}")

# Preprocess the test data
X_test = preprocess_data(test_df.drop(['id'], axis=1), is_train=False)

# Generate probability predictions for the test data
pred_proba = best_model.predict_proba(X_test)

# Create a submission DataFrame with the predictions
submission = pd.DataFrame({
    'id': test_df['id'],
    'Status_C': pred_proba[:, 0],  # Probability for class 'C'
    'Status_CL': pred_proba[:, 1],  # Probability for class 'CL'
    'Status_D': pred_proba[:, 2]  # Probability for class 'D'
})

# Save the submission DataFrame to a CSV file
submission.to_csv('submission.csv', index=False)
print("Predictions saved to submission.csv")
