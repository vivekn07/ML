import numpy as np

# Input data
X = np.array([[2, 9], [1, 5], [3, 6]], dtype=float)
y = np.array([[92], [86], [89]], dtype=float)

# Scaling input data
X = X / np.amax(X, axis=0)
y = y / 100

# Sigmoid Function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivative of Sigmoid Function
def derivatives_sigmoid(x):
    return x * (1 - x)

# Variable initialization
epoch = 5000  # Setting training iterations
lr = 0.1  # Setting learning rate
input_layer_neurons = 2  # number of features in the data set
hidden_layer_neurons = 3  # number of hidden layers neurons
output_neurons = 1  # number of neurons at the output layer

# Weight and bias initialization
wh = np.random.uniform(size=(input_layer_neurons, hidden_layer_neurons))  # weight of the link from input node to hidden node
bh = np.random.uniform(size=(1, hidden_layer_neurons))  # bias of the link from input node to hidden node
wout = np.random.uniform(size=(hidden_layer_neurons, output_neurons))  # weight of the link from hidden node to output node
bout = np.random.uniform(size=(1, output_neurons))  # bias of the link from hidden node to output node

# Training
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

# Print results
print("Input:\n", str(X))
print("Actual Output:\n", str(y))
print("Predicted Output:\n", output)
