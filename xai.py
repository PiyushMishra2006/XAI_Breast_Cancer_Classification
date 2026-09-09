import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    classification_report,
    roc_curve
)

data = load_breast_cancer()

X = data.data
y = data.target

feature_names = data.feature_names
target_names = data.target_names

print("=" * 60)
print("DATASET INFORMATION")
print("=" * 60)

print("Number of samples:", X.shape[0])
print("Number of features:", X.shape[1])

print("\nTarget names:")
print(target_names)

print("\nClass counts:")
print(np.bincount(y))

df = pd.DataFrame(X, columns=feature_names)
df["target"] = y

print("\n" + "=" * 60)
print("DATAFRAME INFORMATION")
print("=" * 60)

print("\nFirst 5 rows:")
print(df.head())

print("\nDataset shape:")
print(df.shape)

print("\nData types:")
print(df.dtypes)

print("\nTotal missing values:")
print(df.isnull().sum().sum())

print("\nClass distribution:")
print(df["target"].value_counts())

print("\nStatistical summary:")
print(df.describe())


print("\n" + "=" * 60)
print("EDA")
print("=" * 60)

plt.figure(figsize=(6, 4))

sns.countplot(x=y)

plt.xticks([0, 1], ["Malignant", "Benign"])
plt.xlabel("Diagnosis")
plt.ylabel("Number of Samples")
plt.title("Class Distribution")

plt.tight_layout()
plt.savefig("Images/class_distribution.png")
plt.show()
plt.close()

plt.figure(figsize=(14, 10))

correlation_matrix = df.drop(columns="target").corr()

sns.heatmap(
    correlation_matrix,
    cmap="coolwarm",
    center=0
)

plt.title("Feature Correlation Heatmap")

plt.tight_layout()
plt.savefig("Images/feature_correlation_heatmap.png")
plt.show()
plt.close()


X = df.drop(columns="target")
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\n" + "=" * 60)
print("TRAIN / TEST SPLIT")
print("=" * 60)

print("Training samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])

print("\nTraining class distribution:")
print(y_train.value_counts())

print("\nTesting class distribution:")
print(y_test.value_counts())


scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nFeature scaling completed.")


def evaluate_model(model_name, y_true, y_pred, y_probability):

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    tn, fp, fn, tp = cm.ravel()

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    specificity = tn / (tn + fp)

    roc_auc = roc_auc_score(y_true, y_probability)

    print("\n" + "=" * 60)
    print(model_name)
    print("=" * 60)

    print(f"Accuracy    : {accuracy:.4f}")
    print(f"Precision   : {precision:.4f}")
    print(f"Recall      : {recall:.4f}")
    print(f"Specificity : {specificity:.4f}")
    print(f"F1 Score    : {f1:.4f}")
    print(f"ROC-AUC     : {roc_auc:.4f}")

    print("\nConfusion Matrix:")
    print(cm)

    print("\nClassification Report:")

    print(
        classification_report(
            y_true,
            y_pred,
            target_names=["Malignant", "Benign"]
        )
    )

    return {
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "Specificity": specificity,
        "F1": f1,
        "ROC-AUC": roc_auc
    }


print("\n" + "=" * 60)
print("TRAINING LOGISTIC REGRESSION")
print("=" * 60)

lr = LogisticRegression(
    max_iter=5000,
    random_state=42
)

lr.fit(
    X_train_scaled,
    y_train
)

y_pred_lr = lr.predict(X_test_scaled)

y_probability_lr = lr.predict_proba(
    X_test_scaled
)[:, 1]

lr_results = evaluate_model(
    "Logistic Regression",
    y_test,
    y_pred_lr,
    y_probability_lr
)


print("\n" + "=" * 60)
print("TRAINING MULTILAYER PERCEPTRON")
print("=" * 60)

mlp = MLPClassifier(
    hidden_layer_sizes=(64, 32),
    max_iter=2000,
    early_stopping=True,
    random_state=42
)

mlp.fit(
    X_train_scaled,
    y_train
)

y_pred_mlp = mlp.predict(
    X_test_scaled
)

y_probability_mlp = mlp.predict_proba(
    X_test_scaled
)[:, 1]

mlp_results = evaluate_model(
    "Multilayer Perceptron",
    y_test,
    y_pred_mlp,
    y_probability_mlp
)


results = pd.DataFrame([
    lr_results,
    mlp_results
])

print("\n" + "=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

print(
    results.to_string(index=False)
)


fig, axes = plt.subplots(
    1,
    2,
    figsize=(11, 4)
)

cm_lr = confusion_matrix(
    y_test,
    y_pred_lr
)

sns.heatmap(
    cm_lr,
    annot=True,
    fmt="d",
    cmap="Blues",
    ax=axes[0]
)

axes[0].set_title("Logistic Regression")
axes[0].set_xlabel("Predicted")
axes[0].set_ylabel("Actual")


cm_mlp = confusion_matrix(
    y_test,
    y_pred_mlp
)

sns.heatmap(
    cm_mlp,
    annot=True,
    fmt="d",
    cmap="Blues",
    ax=axes[1]
)

axes[1].set_title("MLP")
axes[1].set_xlabel("Predicted")
axes[1].set_ylabel("Actual")

plt.tight_layout()
plt.savefig("Images/confusion_matrices.png")
plt.show()


fpr_lr, tpr_lr, _ = roc_curve(
    y_test,
    y_probability_lr
)

fpr_mlp, tpr_mlp, _ = roc_curve(
    y_test,
    y_probability_mlp
)

plt.figure(figsize=(7, 5))

plt.plot(
    fpr_lr,
    tpr_lr,
    label=f"Logistic Regression (AUC = {lr_results['ROC-AUC']:.3f})"
)

plt.plot(
    fpr_mlp,
    tpr_mlp,
    label=f"MLP (AUC = {mlp_results['ROC-AUC']:.3f})"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")

plt.legend()
plt.tight_layout()
plt.savefig("Images/roc_curves.png")
plt.show()



print("\nDataset:")
print("WDBC Breast Cancer Dataset")
print("Samples:", X.shape[0])
print("Features:", X.shape[1])

print("\nModels trained:")
print("1. Logistic Regression")
print("2. Multilayer Perceptron (MLP)")

print("\nResults:")
print(results.to_string(index=False))

print("\nModel Comparison Report:")
print(results.round(4).to_string(index=False))


plt.figure(figsize=(10, 6))

results_plot = results.set_index("Model")

results_plot[
    ["Accuracy", "Precision", "Recall", "Specificity", "F1", "ROC-AUC"]
].plot(
    kind="bar",
    figsize=(10, 6)
)

plt.title("Model Performance Comparison")
plt.xlabel("Model")
plt.ylabel("Score")
plt.ylim(0, 1.05)
plt.xticks(rotation=0)
plt.legend(title="Metrics")
plt.tight_layout()
plt.savefig("Images/model_performance_comparison.png")
plt.show()
plt.close()