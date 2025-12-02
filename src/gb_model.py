import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("data/processed/cleaned_with_features.csv")

#drop target and url
X = df.drop(columns=["viral", " shares", "shares_log", "url"], errors="ignore")
y = df["viral"]

#split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
random_state=42, stratify=y)

#gb model
gb_classifier = GradientBoostingClassifier(random_state=42, n_estimators=200,
learning_rate=0.05,max_depth=3,)

gb_classifier.fit(X_train, y_train)
y_pred = gb_classifier.predict(X_test)
y_proba = gb_classifier.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_proba)

print("Gradient Boosting Result")
print(f"Accuracy :{acc:.3f}")
print(f"F1 score :{f1:.3f}")
print(f"AUC :{auc:.3f}\n")
print("Classification report:")
print(classification_report(y_test, y_pred, digits=3))

#feature importance
importance = gb_classifier.feature_importances_
feature_names = X.columns
idx = np.argsort(importance)[-15:] 

plt.figure(figsize=(8, 6))
plt.barh(feature_names[idx], importance[idx])
plt.title("Top 15 Important Features")
plt.xlabel("Score")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()