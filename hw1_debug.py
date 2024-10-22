import numpy as np
import torch
import sklearn
from sklearn.datasets import make_gaussian_quantiles
from sklearn.model_selection import train_test_split
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer

if torch.cuda.is_available():
    torch.set_default_device("cuda")
    device = torch.device("cuda:0")

def load_data(num_samples, num_features):
    return make_gaussian_quantiles(
        mean=None,
        cov=0.7,
        n_samples=num_samples,
        n_features=num_features, # was 2
        n_classes=2,
        shuffle=True,
        random_state=42,
    )

class NeuralNetwork:
    def __init__(self, n_in, n_hidden, n_out):
        # number of input nodes
        self.n_x = n_in
        # number of hidden nodes
        self.n_h = n_hidden
        # number of output nodes
        self.n_y = n_out
        # Define 1st weight matrix (using random initialization)
        # self.W1 = torch.rand(n_in, n_hidden) #np.sqrt(1 / n_in)
        self.W1 = torch.normal(mean = 0, std = .01, size = (n_in, n_hidden)).to(device)
        # define 1st bias vector
        # self.b1 = torch.rand(n_hidden)
        self.b1 = torch.normal(mean = 0, std = 1, size = (n_hidden,)).to(device)
        # Define 2nd weight matrix (using random initialization)
        # self.W2 = torch.rand(n_hidden, n_out) #np.sqrt(1 / n_hidden)
        self.W2 = torch.normal(mean = 0, std = .01, size = (n_hidden, n_out)).to(device)
        # Define 2nd bias vector
        # self.b2 = torch.rand(n_out)
        self.b2 = torch.normal(mean = 0, std = 1, size = (n_out,)).to(device)
              
    def forward(self, X):
        hidden = torch.sigmoid(torch.mm(X, self.W1) + self.b1)
        output = torch.sigmoid(torch.mm(hidden, self.W2) + self.b2)
        return output
        
    def backward(self, X, y, learning_rate):
        z1 = torch.mm(X, self.W1) + self.b1
        a1 = torch.sigmoid(z1)
        z2 = torch.mm(a1, self.W2) + self.b2
        a2 = torch.sigmoid(z2)
        sigmoid_prime = lambda x: torch.sigmoid(x) * (1 - torch.sigmoid(x))

        delta_loss = a2 - y.unsqueeze(1) # dL/dz2
        delta_W2 = torch.mm(a1.t(), delta_loss)
        delta_b2 = torch.sum(delta_loss, axis = 0)
        delta_z1 = torch.mm(delta_loss, self.W2.t()) * sigmoid_prime(z1)
        delta_W1 = torch.mm(X.t(), delta_z1)
        delta_b1 = torch.sum(delta_z1, axis = 0)

        N = X.shape[0]
        self.W1 -= learning_rate / N * delta_W1 # update of the 1st weight matrix
        self.b1 -= learning_rate / N * delta_b1 # update of the 1st bias vector
        self.W2 -= learning_rate / N * delta_W2 # update of the 2nd weight matrix
        self.b2 -= learning_rate / N * delta_b2 # update of the 2nd bias vector

        
    def train(self, X_train, y_train, X_valid, y_valid, epochs, learning_rate):
        for e in range(epochs):
            xe_loss = lambda y, y_pred: -torch.mean(torch.mul(torch.log(y_pred), y) + torch.mul(torch.log(1 - y_pred),  (1 - y))) # fill in the question marks 

            y_train_pred = torch.clamp(self.forward(X_train), 1e-7, 1 - 1e-7)
            training_loss = xe_loss(y_train, y_train_pred)
            

            self.backward(X_train, y_train, learning_rate)
            y_valid_pred = torch.clamp(self.forward(X_valid), 1e-7, 1 - 1e-7)
            valid_loss = xe_loss(y_valid, y_valid_pred)
            
            
            # self.W1 -= # update of the 1st weight matrix
            # self.b1 -= # update of the 1st bias vector
            # self.W2 -= # update of the 2nd weight matrix
            # self.b2 -= # update of the 2nd bias vector
            if not e % 500:
                print("Epoch %d/%d: Training Loss %.6f Validation Loss %.6f" %(e, epochs, training_loss, valid_loss))
                # print(self.W1)
    
    def score(self, X, y):
        y_pred = (self.forward(X) > .5).float()
        return 1 - (y_pred - y.unsqueeze(1)).abs().mean(axis = 0)


def main():
    n_samples = 5000; num_features = 5
    gaussian_quantiles = load_data(n_samples, num_features)
    X, y = gaussian_quantiles

    # data = load_breast_cancer()
    # X, y = data.data, data.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    X_train_torch = torch.from_numpy(X_train).float().to(device)
    y_train_torch = torch.from_numpy(y_train).float().to(device)
    X_valid_torch = torch.from_numpy(X_test).float().to(device)
    y_valid_torch = torch.from_numpy(y_test).float().to(device)

    # from sklearn.neural_network import MLPClassifier
    # from sklearn.metrics import accuracy_score
    # mlp = MLPClassifier(hidden_layer_sizes=(10,), activation='logistic', 
    #                 max_iter=25000, learning_rate_init=0.01, random_state=42)

    # mlp.fit(X_train, y_train)

    # Make predictions on the test data
    # y_pred = mlp.predict(X_test)

    # Calculate the accuracy
    # accuracy = accuracy_score(y_test, y_pred)

    # print(f'Accuracy: {accuracy:.2f}')


    nn = NeuralNetwork(X_train.shape[1], 10, 1)

    nn.train(
        X_train_torch,
        y_train_torch,
        X_valid_torch,
        y_valid_torch,
        epochs=5000,
        learning_rate=0.05
    )

    nn.score(X_train_torch, y_train_torch)

    breakpoint()

if __name__ == "__main__":
    main()