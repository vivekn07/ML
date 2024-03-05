import numpy as np

# Input data
X = np.array([[2, 9], [1, 5], [3, 6]], dtype=float)  # Two inputs [sleep, study]
y = np.array([[92], [86], [89]], dtype=float)  # One output [Expected % in Exams]

# Normalize input data
X_max = np.amax(X, axis=0)  # Maximum of X array longitudinally
X /= X_max

# Sigmoid Function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivative of Sigmoid Function
def derivatives_sigmoid(x):
    return x * (1 - x)

# Variable initialization
epoch = 5000  # Setting training iterations
lr = 0.1  # Setting learning rate
inputlayer_neurons = 2  # Number of features in the dataset
hiddenlayer_neurons = 3  # Number of hidden layer neurons
output_neurons = 1  # Number of neurons at the output layer

# Weight and bias initialization
wh = np.random.uniform(size=(inputlayer_neurons, hiddenlayer_neurons))  # Weight of the link from input node to hidden node
bh = np.random.uniform(size=(1, hiddenlayer_neurons))  # Bias of the link from input node to hidden node
wout = np.random.uniform(size=(hiddenlayer_neurons, output_neurons))  # Weight of the link from hidden node to output node
bout = np.random.uniform(size=(1, output_neurons))  # Bias of the link from hidden node to output node

# Training loop
for i in range(epoch):
    # Forward Propagation
    hinpl = np.dot(X, wh)
    hinp = hinpl + bh
    hlayer_act = sigmoid(hinp)

    outinpl = np.dot(hlayer_act, wout)
    outinp = outinpl + bout
    output = sigmoid(outinp)

    # Backpropagation
    EO = y - output
    outgrad = derivatives_sigmoid(output)
    d_output = EO * outgrad

    EH = d_output.dot(wout.T)
    hiddengrad = derivatives_sigmoid(hlayer_act)
    d_hiddenlayer = EH * hiddengrad

    # Update weights and biases
    wout += hlayer_act.T.dot(d_output) * lr
    wh += X.T.dot(d_hiddenlayer) * lr

print("Input:\n", X)
print("Actual Output:\n", y)
print("Predicted Output:\n", output)