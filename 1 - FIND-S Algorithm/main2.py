import pandas as pd

# Load the enjoysport.csv file
file_path = 'enjoysport.csv'
data = pd.read_csv(file_path)
concepts = data.iloc[:, :-1].values
target = data.iloc[:, -1].values

# Initialize specific_h with the first instance
specific_h = concepts[0].copy()

# Implement the Candidate Elimination Algorithm
for i, h in enumerate(concepts):
    if target[i] == 'yes':
        specific_h = [h_i if h_i == h_j else '?' for h_i, h_j in zip(specific_h, h)]

print("Maximally specific hypothesis:", specific_h)
