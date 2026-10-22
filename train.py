"""Train the symptom -> disease classifier and save it for the app."""
import json
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

df = pd.read_csv("data/diseases.csv")
# Column names in the raw file contain stray spaces ("spotting_ urination")
df.columns = [c.strip().replace(" ", "") for c in df.columns]

# The raw file has ~4,960 rows but only ~305 unique ones. Splitting before
# de-duplicating leaks identical rows into the test set and inflates accuracy.
df = df.drop_duplicates().reset_index(drop=True)

X, y = df.drop(columns="prognosis"), df["prognosis"]
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

clf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
clf.fit(X_tr, y_tr)
print(f"Held-out accuracy (de-duplicated): {accuracy_score(y_te, clf.predict(X_te)):.3f}")

# Refit on all unique rows for the deployed model
clf.fit(X, y)
joblib.dump(clf, "model/model.joblib", compress=3)
json.dump(list(X.columns), open("model/symptoms.json", "w"))
print("Saved model/model.joblib and model/symptoms.json")
