# Import necessary libraries
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import f_classif
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('winequality-red.csv')

# Separate features and target variable
X = data.drop('quality', axis=1)
y = data['quality']

# Standardize the features
scaler = StandardScaler()
X_standardized = scaler.fit_transform(X)

# Perform Principal Component Analysis (PCA)
pca = PCA()
X_pca = pca.fit_transform(X_standardized)

# Plot the explained variance ratio
plt.figure(figsize=(8, 6))
plt.plot(range(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_.cumsum(), marker='o', linestyle='--')
plt.title('Explained Variance Ratio')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance Ratio')
plt.show()

# Select the number of components based on explained variance
n_components = 2  # You can adjust this based on the plot
pca = PCA(n_components=n_components)
X_pca = pca.fit_transform(X_standardized)

# Feature Scoring and Ranking using f_classif
f_scores, p_values = f_classif(X_standardized, y)

# Create a DataFrame for feature scores and ranks
feature_ranking_df = pd.DataFrame({'Feature': X.columns, 'F-Score': f_scores, 'P-Value': p_values})
feature_ranking_df = feature_ranking_df.sort_values(by='F-Score', ascending=False).reset_index(drop=True)
feature_ranking_df['Rank'] = feature_ranking_df.index + 1

# Display the feature scores and ranks
print("Feature Scores and Ranks:")
print(feature_ranking_df)

# You can further analyze or use the selected features (based on PCA or feature scores) for your specific task.
