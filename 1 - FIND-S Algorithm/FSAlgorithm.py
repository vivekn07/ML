import csv

# Load the enjoysport.csv file
file_path = 'enjoysport.csv'
a = []
with open(file_path, 'r') as csvfile:
    for row in csv.reader(csvfile):
        a.append(row)

# Calculate the number of attributes (excluding the target attribute)
num_attributes = len(a[0]) - 1

# Initialize the initial hypothesis
hypothesis = ['0'] * num_attributes

print("\n The initial hypothesis is:", hypothesis)
print("\n The total number of training instances is:", len(a))

# Implement the Candidate Elimination Algorithm
for i in range(0, len(a)):
    if a[i][num_attributes] == 'yes':
        for j in range(0, num_attributes):
            if hypothesis[j] == '0' or hypothesis[j] == a[i][j]:
                hypothesis[j] = a[i][j]
            else:
                hypothesis[j] = '?'

    print("\n The hypothesis for training instance {} is:\n".format(i + 1), hypothesis)

print("\n The Maximally specific hypothesis for the training instance is:")
print(hypothesis)
