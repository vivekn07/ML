import numpy as np
import pandas as pd

# Read the CSV file into a Pandas DataFrame
data = pd.read_csv('enjoysport.csv')

# Extract the features (concepts) and target column
concepts = np.array(data.iloc[:, :-1])
target = np.array(data.iloc[:, -1])


def learn(concepts, target):
    specific_h = concepts[0].copy()
    print("Initialization of specific_h and general_h")
    print(specific_h)

    general_h = [["?" for _ in range(len(specific_h))] for _ in range(len(specific_h))]
    print(general_h)

    for i, h in enumerate(concepts):
        print("For Loop Starts")
        if target[i] == "yes":
            print("If instance is Positive")
            for x in range(len(specific_h)):
                if h[x] != specific_h[x]:
                    specific_h[x] = '?'
                    general_h[x][x] = '?'
        elif target[i] == "no":
            print("If instance is Negative")
            for x in range(len(specific_h)):
                if h[x] != specific_h[x]:
                    general_h[x][x] = specific_h[x]
                else:
                    general_h[x][x] = '?'

    print("Steps of Candidate Elimination Algorithm:")
    print("Specific_h:", specific_h)
    print("General_h:", general_h)
    print("\n")

    # Remove unnecessary entries from general_h
    indices = [i for i, val in enumerate(general_h) if val != ['?' for _ in range(len(specific_h))]]
    for i in indices:
        general_h.remove(['?' for _ in range(len(specific_h))])

    return specific_h, general_h


s_final, g_final = learn(concepts, target)

print("Final Specific_h:")
print(s_final)
print("Final General_h:")
print(g_final)
