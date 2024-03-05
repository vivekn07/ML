from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Sample data (replace with your dataset)
documents = [
    "This is the first document.",
    "This is the second document.",
    "This is the third document.",
    "This is the fourth document.",
    "This is the fifth document.",
    "This is the sixth document."
]
labels = [1, 0, 1, 0, 1, 0]  # 1 for positive, 0 for negative

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(documents, labels, test_size=0.2, random_state=42)

# Convert text data to numerical features using CountVectorizer
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train Naive Bayes classifier
nb_classifier = MultinomialNB()
nb_classifier.fit(X_train_vec, y_train)

# Make predictions on the test set
predictions = nb_classifier.predict(X_test_vec)

# Evaluate performance
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy: {accuracy:.2f}")

# Display classification report with zero division parameter
print("\nClassification Report:")
print(classification_report(y_test, predictions, zero_division=1))