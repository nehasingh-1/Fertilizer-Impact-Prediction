import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Load and Prepare the Dataset
file_path = './Crop_recommendationV2.csv'
data = pd.read_csv(file_path)

# Display dataset structure and summary
data.info()
print(data.head())

# Step 2: Feature Selection and Preprocessing
# Define features (X) and target (y)
features = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall', 'fertilizer_usage']
target = 'label'

X = data[features]
y = data[target]

# Convert categorical target to numerical
y = y.astype('category').cat.codes

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize the features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 3: Model Development
# Random Forest Classifier
model = RandomForestClassifier(random_state=42)

# Hyperparameter tuning
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Best model
best_model = grid_search.best_estimator_

# Step 4: Model Evaluation
y_pred = best_model.predict(X_test)

print("Classification Report:\n", classification_report(y_test, y_pred))
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")

# Step 5: Visualization
sns.heatmap(pd.DataFrame(confusion_matrix(y_test, y_pred)), annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# Step 6: Save the Model and Scaler
import joblib
joblib.dump(best_model, 'fertilizer_impact_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

# Notes:
# Replace label with other target variables if focusing on crop yield prediction instead of classification.
# Extend the analysis with feature importance visualization and additional metrics if needed.
