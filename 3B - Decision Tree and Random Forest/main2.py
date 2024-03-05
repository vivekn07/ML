import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# Load the dataset
url = "IBMAttritionData.csv"
data = pd.read_csv(url)

# Display the first few rows of the dataset
print("Dataset:")
print(data.head())

# Convert categorical variables to numerical using LabelEncoder
label_encoder = LabelEncoder()
for column in data.columns:
    if data[column].dtype == 'object':
        data[column] = label_encoder.fit_transform(data[column])

# Extract features and target variable
X = data.drop('Attrition', axis=1)
y = data['Attrition']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Decision Tree
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)

# Random Forest
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

# Prediction
dt_predictions = dt_model.predict(X_test)
rf_predictions = rf_model.predict(X_test)

# Test Scores
dt_accuracy = accuracy_score(y_test, dt_predictions)
rf_accuracy = accuracy_score(y_test, rf_predictions)

print("\nDecision Tree Accuracy:", dt_accuracy)
print("Random Forest Accuracy:", rf_accuracy)

# Confusion Matrix
dt_conf_matrix = confusion_matrix(y_test, dt_predictions)
rf_conf_matrix = confusion_matrix(y_test, rf_predictions)

print("\nDecision Tree Confusion Matrix:")
print(dt_conf_matrix)

print("\nRandom Forest Confusion Matrix:")
print(rf_conf_matrix)