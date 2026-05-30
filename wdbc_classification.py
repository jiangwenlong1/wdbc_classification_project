import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)


data = load_breast_cancer()

X = data.data
y = data.target

print("=" * 50)
print("Dataset Information")
print("=" * 50)

print("Data shape:", X.shape)
print("Target shape:", y.shape)
print("Target names:", data.target_names)
print("First 5 feature names:")
print(data.feature_names[:5])

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


models = {
    "SVM": SVC(),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "KNN": KNeighborsClassifier(),
    "Random Forest": RandomForestClassifier(random_state=42)
}

results = {}

print("\n" + "=" * 50)
print("Classifier Results")
print("=" * 50)

for name, model in models.items():

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    results[name] = accuracy

    print(f"\n{name}")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")


best_classifier = max(results, key=results.get)

print("\n" + "=" * 50)
print("Best Classifier")
print("=" * 50)

print("Best Classifier:", best_classifier)
print("Best Accuracy:", round(results[best_classifier], 4))

best_model = models[best_classifier]


best_predictions = best_model.predict(X_test)

cm = confusion_matrix(y_test, best_predictions)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=data.target_names
)

fig, ax = plt.subplots(figsize=(6, 6))
disp.plot(ax=ax)

plt.title(f"Confusion Matrix - {best_classifier}")

plt.savefig(
    "wdbc_classification_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


plt.figure(figsize=(8, 6))

scatter = plt.scatter(
    X[:, 0],
    X[:, 1],
    c=y
)

plt.xlabel(data.feature_names[0])
plt.ylabel(data.feature_names[1])

plt.title("Breast Cancer Dataset Scatter Plot")

plt.colorbar(scatter)

plt.savefig(
    "wdbc_classification_scatter.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("\nScatter plot saved:")
print("wdbc_classification_scatter.png")

print("\nConfusion matrix saved:")
print("wdbc_classification_matrix.png")

print("\nProgram completed successfully.")
