import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.tree import DecisionTreeClassifier

# Read the CSV file
df = pd.read_csv("IBMAttritionData.csv")
df.head()

# Plot countplot for 'Attrition'
sns.countplot(x='Attrition', data=df)

# Drop the 'WorkLifeBalance' column
df.drop(["WorkLifeBalance"], axis='columns', inplace=True)

# Identify categorical columns
categorical_col = []
for column in df.columns:
    if df[column].dtype == object and len(df[column].unique()) <= 50:
        categorical_col.append(column)

# Convert 'Attrition' to categorical codes
df['Attrition'] = df['Attrition'].astype("category").cat.codes

# Label encode categorical columns
label = LabelEncoder()
for column in categorical_col:
    df[column] = label.fit_transform(df[column])

# Split the data into training and testing sets
x = df.drop('Attrition', axis=1)
y = df['Attrition']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

# Function to print model scores
def print_score(clf, x_train, y_train, x_test, y_test, train=True):
    if train:
        pred = clf.predict(x_train)
        clf_report = pd.DataFrame(classification_report(y_train, pred, output_dict=True))
        print("Train Result:\n================================== ")
        print(f"Accuracy score: {accuracy_score(y_train, pred) * 100:.2f}%")
        print(f"Classification Report:\n{clf_report}")
        print(f"Confusion matrix:\n{confusion_matrix(y_train, pred)}\n")
    elif not train:
        pred = clf.predict(x_test)
        clf_report = pd.DataFrame(classification_report(y_test, pred, output_dict=True))
        print("Test Result:\n=================================")
        print(f"Accuracy score: {accuracy_score(y_test, pred) * 100:.2f}%")
        print(f"Classification Report:\n{clf_report}")
        print(f"Confusion matrix:\n{confusion_matrix(y_test, pred)}\n")

# Create Decision Tree Classifier
tree_clf = DecisionTreeClassifier(random_state=42)
tree_clf.fit(x_train, y_train)

# Print scores for the Decision Tree Classifier
print_score(tree_clf, x_train, y_train, x_test, y_test, train=True)
print_score(tree_clf, x_train, y_train, x_test, y_test, train=False)
