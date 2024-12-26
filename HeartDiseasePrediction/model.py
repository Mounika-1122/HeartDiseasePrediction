import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
# , classification_report, confusion_matrix
import joblib

# Load the dataset
heart_data = pd.read_csv('heart_disease_data.csv')

# Splitting features and target
X = heart_data.drop(columns='target', axis=1)
Y = heart_data['target']

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.2, stratify=Y, random_state=2)

# Logistic Regression model with increased max_iter and alternative solver
model = LogisticRegression(max_iter=2000, solver='saga')  # Try saga for large datasets or tough convergence

model.fit(X_train, Y_train)

# Predictions
Y_pred = model.predict(X_test)

# Save the trained model and scaler
joblib.dump(model, 'heart_disease_model.pkl')
joblib.dump(scaler, 'scaler.pkl')  # Save the scaler for use during prediction

# Evaluation
accuracy = accuracy_score(Y_test, Y_pred)
print(f"Accuracy: {accuracy:.4f}")



print("Model trained, evaluated, and saved successfully!")
