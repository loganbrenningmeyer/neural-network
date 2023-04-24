from math import exp
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons, load_breast_cancer, load_digits
from scipy.special import expit

class Neuron:
    def __init__(self, activation):
        # weights established in init_weights_inputs
        self.weights = list()
        self.bias = 0
        # inputs established in init_weights_inputs
        self.inputs = list()
        self.output = 0
        # weights/bias derivatives from backpropagation
        self.weights_derivatives = list()
        self.bias_derivative = 0
        # set activation function
        self.activation = activation

    def calculate_weighted_sum(self):
        # dot product weights
        self.z = np.dot(self.weights, self.inputs)
        # add bias
        self.z += self.bias

    def activation_function(self):
        if self.activation == 'relu':
            return self.relu()
        elif self.activation == 'sigmoid':
            return self.sigmoid()

    def activation_derivative(self):
        if self.activation == 'relu':
            return self.relu_derivative()
        elif self.activation == 'sigmoid':
            return self.sigmoid_derivative()

    def relu(self):
        return max(0, self.z)
    
    def relu_derivative(self):
        if self.z > 0:
            return 1
        else:
            return 0

    def sigmoid(self):
        return expit(self.z)
        #return 1/(1 + exp(-self.z))
    
    def sigmoid_derivative(self):
        return self.sigmoid() * (1 - self.sigmoid())

class Layer:
    def __init__(self, num_neurons, activation):
        self.neurons = [Neuron(activation) for _ in range(num_neurons)]

class NeuralNetwork:

    def __init__(self, num_neurons, activation):
        # sets hidden layers to the given activation function and the output layer to sigmoid
        self.layers = [Layer(num_neurons[i], activation) for i in range(len(num_neurons) - 1)] + [Layer(num_neurons[-1], 'sigmoid')]
        self.init_weights_inputs()

    def plot_data(self):
        # Separate the data points by class
        class_0 = self.X[self.y == 0]
        class_1 = self.X[self.y == 1]

        # Create a scatter plot of the data points in each class
        plt.scatter(class_0[:, 0], class_0[:, 1], label='Class 0', alpha=0.5)
        plt.scatter(class_1[:, 0], class_1[:, 1], label='Class 1', alpha=0.5)

        # Generate a grid of points
        x1_values, x2_values = np.meshgrid(np.linspace(-6, 6, 100), np.linspace(-6, 6, 100))

        # Predict the class probabilities for each point in the grid
        grid_predictions = []
        for x1, x2 in zip(x1_values.ravel(), x2_values.ravel()):
            self.layers[0].neurons[0].output = x1
            self.layers[0].neurons[1].output = x2
            self.forward()
            grid_predictions.append(self.layers[-1].neurons[0].output)

        # Reshape the predictions to match the grid
        grid_predictions = np.array(grid_predictions).reshape(x1_values.shape)

        # Create a contour plot of the decision boundaries
        plt.contourf(x1_values, x2_values, grid_predictions, levels=[0, 0.5, 1], alpha=0.2, colors=['blue', 'red'])
        
        # Set axis labels
        plt.xlabel('x1')
        plt.ylabel('x2')

        # Add a legend
        plt.legend()

        # Display the plot
        plt.pause(0.1)  # pause for 0.1 seconds
        plt.clf()       # clear the current figure

    def load_digits_data(self):
        data = load_digits(n_class=2)
        self.X, self.y = data.data, data.target

    def load_breast_cancer_data(self):
        data = load_breast_cancer()
        self.X, self.y = data.data, data.target

    def load_moons_data(self):
        self.X, self.y = make_moons(n_samples=1000, noise=0.1, random_state=0)

    def load_circle_data(self):
        num_samples = 1000
        radius_inner = 2
        radius_outer = 4
        # Initialize input and target lists
        inputs = []
        targets = []

        for _ in range(num_samples):
            # Randomly choose a class (inner circle or outer ring)
            target = np.random.randint(0, 2)

            # Generate a random angle
            angle = 2 * np.pi * np.random.rand()

            # Choose a random distance from the center based on the class
            if target == 0:  # Inner circle
                distance = radius_inner * np.sqrt(np.random.rand())
            else:  # Outer ring
                distance = radius_outer + 1.5 * np.random.rand()

            # Calculate x and y coordinates based on the angle and distance
            x1 = distance * np.cos(angle)
            x2 = distance * np.sin(angle)

            # Add the data point to the input and target lists
            inputs.append([x1, x2])
            targets.append(target)

            self.X = np.array(inputs)
            self.y = np.array(targets)


    # data [[x1, x2], [x1, x2], ...]
    # targets [y1, y2, ...]
    def load_xor_data(self):
        rng = np.random.RandomState(0)
        X = rng.randn(1000, 2)    #randomly generates dataset (x0, x1)
        X = (X > 0) * 2 - 1 + 0.2 * rng.randn(*X.shape)   #moves positive/negative data to 1 or -1 ((X > 0) * 2 - 1) then adds some noise for spread (+ 0.3 * rng.randn(*X.shape))
        y = np.logical_xor(X[:,0] > 0, X[:,1] > 0).astype(int)    #Compare X0 and X1 sign to determine XOR
        self.X = np.multiply(3, X)
        self.y = y

    def train(self):
        epochs = 500
        # train for a number of epochs
        for epoch in range(epochs):
            # iterate through each data point
            for i in range(int(len(self.X) * 0.75)):
                # send inputs into the input layer
                # allows for more than 2 input features
                for neuron in range(len(self.layers[0].neurons)):
                    self.layers[0].neurons[neuron].output = self.X[i][neuron]
                # forward prop
                self.forward()
                # calculate error (target - output)
                error = self.y[i] - self.layers[-1].neurons[0].output
                # backprop
                self.backprop(error)
                # update weights
                self.update_weights()
            # plot the data and decision boundary every 100 epochs
            if (epoch % 25 == 0):
                self.plot_data()

            # check validation accuracy
            if (epoch % 25 == 0):
                accuracy = self.validate()
                print(f"Epoch: {epoch}, Accuracy: {accuracy}")
                if (accuracy == 1):
                    break

    def validate(self):
        correct = 0
        # iterate through each data point
        for i in range(int(len(self.X) * 0.75), len(self.X)):
            # send inputs into the input layer
            # allows for more than 2 input features
            for neuron in range(len(self.layers[0].neurons)):
                self.layers[0].neurons[neuron].output = self.X[i][neuron]
            # forward prop
            self.forward()
            # calculate error (target - output)
            error = self.y[i] - self.layers[-1].neurons[0].output
            # if the output is within 0.5 of the target, count it as correct
            if (abs(error) < 0.5):
                correct += 1
        return correct / (len(self.X) * 0.25)

    def test(self):
        correct = 0
        # iterate through each data point
        for i in range(int(len(self.X) * 0.75), len(self.X)):
            # send inputs into the input layer
            # allows for more than 2 input features
            for neuron in range(len(self.layers[0].neurons)):
                self.layers[0].neurons[neuron].output = self.X[i][neuron]
            # forward prop
            self.forward()
            # calculate error (target - output)
            error = self.y[i] - self.layers[-1].neurons[0].output
            # if the output is within 0.5 of the target, count it as correct
            if (abs(error) < 0.5):
                correct += 1
        print("Accuracy: " + str(correct / (len(self.X) * 0.25)))

    def forward(self):
        # send inputs into the first hidden layer
        for neuron in self.layers[1].neurons:
            neuron.inputs = np.array([self.layers[0].neurons[neuron].output for neuron in range(len(self.layers[0].neurons))])

        # propagate through each layer excluding the input layer
        for curr_layer in range(1, len(self.layers)):
            # update weights and inputs
            for neuron in self.layers[curr_layer].neurons:
                # take inputs as the outputs of the previous layer
                neuron.inputs = np.array([neuron.output for neuron in self.layers[curr_layer - 1].neurons])
                # calculate z
                neuron.calculate_weighted_sum()
                # calculate output (sigmoid(z))
                neuron.output = neuron.activation_function()

    def backprop(self, error):
        # iterate through each layer in reverse order, excluding the input layer
        for curr_layer in range(len(self.layers) - 1, 0, -1):
            # iterate through each neuron in the current layer
            for neuron in range(len(self.layers[curr_layer].neurons)):
                # error = (target - output)
                gradient = -(error)
                # calculate bias derivative
                for layer in range(len(self.layers) - 1, curr_layer, -1):
                    # multiply by weights and sigmoid derivatives of neuron outputs
                    # in the path towards the current neuron
                    gradient *= self.layers[layer].neurons[0].activation_derivative()
                    # ensure that the weight connecting the current neuron is multiplied finally
                    # and not the weight of the first neuron in the layer as is the default in the else statement
                    if (layer == curr_layer + 1):
                        gradient *= self.layers[layer].neurons[0].weights[neuron]
                    else:
                        gradient *= self.layers[layer].neurons[0].weights[0]
                # multiply by the sigmoid derivative of the current neuron
                gradient *= self.layers[curr_layer].neurons[neuron].activation_derivative()
                # set the bias derivative
                self.layers[curr_layer].neurons[neuron].bias_derivative = gradient
                # calculate weight derivatives (bias derivative * input)
                # for each weight, multiply the gradient by the weight's corresponding input
                for weight in range(len(self.layers[curr_layer].neurons[neuron].weights)):
                    self.layers[curr_layer].neurons[neuron].weights_derivatives.append(gradient * self.layers[curr_layer].neurons[neuron].inputs[weight])

    def update_weights(self):
        # learning rate
        alpha = 0.1
        # iterate through each layer, excluding the input layer
        for curr_layer in self.layers[1:]:
            # iterate through each neuron in the current layer
            for neuron in curr_layer.neurons:
                # new weight = current weight - learning rate * weight derivative
                # update each weight in the neuron
                for weight in range(len(neuron.weights)):
                    neuron.weights[weight] = neuron.weights[weight] - alpha * neuron.weights_derivatives[weight]
                # update the bias 
                neuron.bias = neuron.bias - alpha * neuron.bias_derivative
        # reset weights/bias derivatives to prepare for the next epoch
        for layer in self.layers:
            for neuron in layer.neurons:
                neuron.weights_derivatives = list()
                neuron.bias_derivative = 0

    def init_weights_inputs(self):
        for curr_layer in range(1, len(self.layers)):
            for neuron in self.layers[curr_layer].neurons:
                # generate random weights for each neuron in the previous layer
                neuron.weights = np.random.uniform(-1, 1, len(self.layers[curr_layer - 1].neurons))
                # generate random bias
                neuron.bias = np.random.uniform(-1, 1)
                # set inputs to outputs of previous layer
                neuron.inputs = np.array([neuron.output for neuron in self.layers[curr_layer - 1].neurons])
    
    def print_neurons(self):
        for layer in range(len(self.layers)):
            if (layer == 0):
                print(f"--- Input Layer ---\n")
            elif (layer == len(self.layers) - 1):
                print(f"--- Output Layer ---\n")
            else:
                print(f"--- Hidden Layer {layer}: ---\n")
            for neuron in self.layers[layer].neurons:
                print(f"weights: {neuron.weights}")
                print(f"bias: {neuron.bias}")
                print(f"inputs: {neuron.inputs}")
                print(f"output: {neuron.output}")
                print(f"weights derivatives: {neuron.weights_derivatives}")
                print(f"bias derivative: {neuron.bias_derivative}\n")

def main():
    # create neural network with 4 layers
    # 2 neurons in the input layer, 3 in the first hidden layer, 2 in the second hidden layer, 1 in the output layer


    # load xor data into the network
    nn = NeuralNetwork([2, 3, 2, 1], 'sigmoid')
    nn.load_xor_data()
    nn.train()
    nn.test()

    # load circle data into the network
    nn = NeuralNetwork([2, 5, 3, 2, 1], 'sigmoid')
    nn.load_circle_data()
    nn.train()
    nn.test()

    # load moon data into the network
    nn = NeuralNetwork([2, 5, 3, 1], 'sigmoid')
    nn.load_moons_data()
    nn.train()
    nn.test() 

    '''
    nn = NeuralNetwork([64, 32, 16, 8, 1], 'sigmoid')
    nn.load_digits_data()
    nn.train()
    nn.test()
    '''
main()

