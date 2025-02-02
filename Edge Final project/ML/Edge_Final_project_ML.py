import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import numpy as np



data = pd.read_csv("corrected_marketing_campaign.csv")
print(data.head)

print(data.describe())

print(data.info())

# Handle missing values
for col in ["Age", "EstimatedSalary", "Spending Score"]:
    data[col] = data[col].fillna(data[col].mean())

# Select features and target
x = data[["Age", "EstimatedSalary"]].values
y = data["Purchased"].values

# Split the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

# Apply MinMax scaling
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Train SVM model with linear kernel
svm_clf = SVC(kernel='linear', random_state=0)
svm_clf.fit(x_train, y_train)

# Make predictions
y_pred_svm = svm_clf.predict(x_test)

# Evaluate model performance
cm_svm = confusion_matrix(y_test, y_pred_svm)
acc_svm = accuracy_score(y_test, y_pred_svm)

print("SVM Confusion Matrix:")
print(cm_svm)
print(f"SVM Accuracy: {acc_svm:.2f}")

# Train Random Forest model
rf_clf = RandomForestClassifier(n_estimators=10, random_state=0)
rf_clf.fit(x_train, y_train)

# Make predictions
y_pred_rf = rf_clf.predict(x_test)

# Evaluate model performance
cm_rf = confusion_matrix(y_test, y_pred_rf)
acc_rf = accuracy_score(y_test, y_pred_rf)

print("Random Forest Confusion Matrix:")
print(cm_rf)
print(f"Random Forest Accuracy: {acc_rf:.2f}")

# Visualize SVM decision boundary
plt.figure(figsize=(10, 6))
plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train, cmap='winter', edgecolor='k', label='Training Data')
plt.scatter(x_test[:, 0], x_test[:, 1], c=y_pred_svm, cmap='autumn', marker='x', label='Predicted Test Data')
plt.title('SVM Decision Boundary Visualization')
plt.legend()
plt.show()
