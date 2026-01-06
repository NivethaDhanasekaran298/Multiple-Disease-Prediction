import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

from Preprocessing import load_data, preprocess_data

os.makedirs("models", exist_ok=True)

datasets = {
    "parkinsons": {
        "path": r"C:\Users\Vivekanandh D\Downloads\parkinsons - parkinsons.csv",
        "target": "status"
    },
    "kidney": {
        "path": r"C:\Users\Vivekanandh D\Downloads\kidney_disease - kidney_disease.csv",
        "target": "classification"
    },
    "liver": {
        "path": r"C:\Users\Vivekanandh D\Downloads\indian_liver_patient - indian_liver_patient.csv",
        "target": "Dataset"
    }
}

for disease, config in datasets.items():

    print(f"\nTraining model for {disease.upper()} dataset")

    df = load_data(config["path"])
    X, y, feature_names = preprocess_data(df, config["target"])

    if disease == "liver":
        y = y.replace({2: 0})

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    joblib.dump(model, f"models/{disease}_model.pkl")
    joblib.dump(feature_names, f"models/{disease}_features.pkl")

    print("Model & feature names saved successfully")

print("\n✅ All models trained and saved successfully!")
