import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer

def load_data(file_path):
    return pd.read_csv(file_path)

def preprocess_data(df, target_column):

    # Drop target and non-predictive ID column
    X = df.drop(columns=[target_column, "name"], errors="ignore")
    y = df[target_column]

    # Save feature names
    feature_names = X.columns.tolist()

    # Identify column types
    num_cols = X.select_dtypes(include=['int64', 'float64']).columns
    cat_cols = X.select_dtypes(include=['object']).columns

    # Handle missing numerical values (ONLY if columns exist)
    if len(num_cols) > 0:
        X[num_cols] = SimpleImputer(strategy='median').fit_transform(X[num_cols])

    # Handle missing categorical values (ONLY if columns exist)
    if len(cat_cols) > 0:
        X[cat_cols] = SimpleImputer(strategy='most_frequent').fit_transform(X[cat_cols])

        # Encode categorical columns
        for col in cat_cols:
            X[col] = LabelEncoder().fit_transform(X[col])

    # Feature scaling
    X_scaled = StandardScaler().fit_transform(X)

    return X_scaled, y, feature_names
