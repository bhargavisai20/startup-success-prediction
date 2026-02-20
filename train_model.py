import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

data = pd.read_csv("startup.csv")

# Fill missing values instead of dropping everything
data = data.fillna(0)

features = [
    "funding_total_usd",
    "funding_rounds",
    "age_first_funding_year",
    "age_last_funding_year",
    "age_first_milestone_year",
    "age_last_milestone_year",
    "relationships"
]

X = data[features]
y = data["status"]

print("Class distribution:")
print(y.value_counts())  # 👈 check both 0 and 1 exist

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier()
model.fit(X_train, y_train)

joblib.dump(model, "model.pkl")

print("✅ model.pkl saved successfully")
