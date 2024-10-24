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

class EarlyStopper:
    def __init__(self, threshold = float("inf"), epsilon = float("inf")):
        self.threshold = threshold
        self.epsilon = epsilon
        self.counter = 0
        self.min_valid_loss = float('inf')

    def early_stop(self, valid_loss):
        if valid_loss < self.min_valid_loss:
            self.min_valid_loss = valid_loss
            self.counter = 0
        elif valid_loss > (self.min_valid_loss + self.epsilon):
            self.counter += 1
            if self.counter >= self.threshold:
                return True
        return False

class NeuralNetwork:
    def __init__(self, n_in, n_hidden, n_out):
        # number of input nodes
        self.n_x = n_in
        # number of hidden nodes
        self.n_h = n_hidden
        # number of output nodes
        self.n_y = n_out

        # Define 1st weight matrix (using random initialization)
        self.W1 = torch.rand(n_in, n_hidden) #np.sqrt(1 / n_in)
        # self.W1 = torch.normal(mean = 0, std = .01, size = (n_in, n_hidden)).to(device)

        # define 1st bias vector
        self.b1 = torch.rand(n_hidden)
        # self.b1 = torch.normal(mean = 0, std = 1, size = (n_hidden,)).to(device)
        
        # Define 2nd weight matrix (using random initialization)
        self.W2 = torch.rand(n_hidden, n_out) #np.sqrt(1 / n_hidden)
        # self.W2 = torch.normal(mean = 0, std = .01, size = (n_hidden, n_out)).to(device)

        # Define 2nd bias vector
        self.b2 = torch.rand(n_out)
        # self.b2 = torch.normal(mean = 0, std = 1, size = (n_out,)).to(device)
              
    def forward(self, X):
        hidden = torch.sigmoid(torch.mm(X, self.W1) + self.b1)
        # hidden = torch.relu(torch.mm(X, self.W1) + self.b1)
        output = torch.sigmoid(torch.mm(hidden, self.W2) + self.b2)
        return output
        
    def backward(self, X, y, learning_rate):
        z1 = torch.mm(X, self.W1) + self.b1
        a1 = torch.sigmoid(z1)
        # a1 = torch.relu(z1)
        z2 = torch.mm(a1, self.W2) + self.b2
        a2 = torch.sigmoid(z2)
        sigmoid_prime = lambda x: torch.sigmoid(x) * (1 - torch.sigmoid(x))
        relu_prime = lambda x: (x > 0).float()

        delta_loss = a2 - y.unsqueeze(1) # dL/dz2
        delta_W2 = torch.mm(a1.t(), delta_loss)
        delta_b2 = torch.sum(delta_loss, axis = 0)
        delta_z1 = torch.mm(delta_loss, self.W2.t()) * sigmoid_prime(z1)
        # delta_z1 = torch.mm(delta_loss, self.W2.t()) * relu_prime(z1)
        delta_W1 = torch.mm(X.t(), delta_z1)
        delta_b1 = torch.sum(delta_z1, axis = 0)

        N = X.shape[0]
        self.W1 -= learning_rate / N * delta_W1 # update of the 1st weight matrix
        self.b1 -= learning_rate / N * delta_b1 # update of the 1st bias vector
        self.W2 -= learning_rate / N * delta_W2 # update of the 2nd weight matrix
        self.b2 -= learning_rate / N * delta_b2 # update of the 2nd bias vector

    
    def score(self, y, y_pred, pct = True):
        # y_pred = (self.forward(X) > .5).float()
        return (1 - ((y_pred > .5).float() - y.unsqueeze(1)).abs().mean(axis = 0)).item() * (100**pct)

    def metrics(self, y, y_pred, pct = True):
        diff = 2 * (y.unsqueeze(1) + 1) - (y_pred > .5).float()
        FP = torch.sum(diff == 1.).item()
        TN = torch.sum(diff == 2.).item()
        TP = torch.sum(diff == 3.).item()
        FN = torch.sum(diff == 4.).item()
        precision = TP / (TP + FP) * (100**pct)
        recall = TP / (TP + FN) * (100**pct)
        f1 = 2 * precision * recall / (precision + recall)
        return precision, recall, f1

    def train_log(self, y_train, y_train_pred, y_valid, y_valid_pred, train_loss, valid_loss, curr_epoch, max_epoch, verbose):
        accuracy_train = self.score(y_train, y_train_pred)
        accuracy_valid = self.score(y_valid, y_valid_pred)
        if verbose: 
            print("Epoch %d/%d: Training Loss %.6f\tValidation Loss %.6f\tTraining accuracy %.2f%%\tValidation accuracy %.2f%%" %(curr_epoch, max_epoch, train_loss, valid_loss, accuracy_train, accuracy_valid))
        if verbose > 1:
            print("Precision %.2f%%\tRecall %.2f%%\tF1 Score %.6f" %self.metrics(y_train, y_train_pred))

    def train(self, X_train, y_train, X_valid, y_valid, epochs, learning_rate, stopper_args = None, verbose = 1, print_interval = 1000):
        stopper = EarlyStopper(**stopper_args)
        for e in range(epochs):
            xe_loss = lambda y, y_pred: -torch.mean(torch.mul(torch.log(y_pred), y.unsqueeze(1)) + torch.mul(torch.log(1 - y_pred),  (1 - y.unsqueeze(1)))) # fill in the question marks 

            y_train_pred = torch.clamp(self.forward(X_train), 1e-7, 1 - 1e-7)
            training_loss = xe_loss(y_train, y_train_pred)
            

            self.backward(X_train, y_train, learning_rate)
            y_valid_pred = torch.clamp(self.forward(X_valid), 1e-7, 1 - 1e-7)
            valid_loss = xe_loss(y_valid, y_valid_pred)
            
            # self.W1 -= # update of the 1st weight matrix
            # self.b1 -= # update of the 1st bias vector
            # self.W2 -= # update of the 2nd weight matrix
            # self.b2 -= # update of the 2nd bias vector

            if not (e+1) % print_interval:
                self.train_log(y_train, y_train_pred, y_valid, y_valid_pred, training_loss, valid_loss, e+1, epochs, verbose)
            if stopper.early_stop(valid_loss):
                print("[Early stopping]")
                self.train_log(y_train, y_train_pred, y_valid, y_valid_pred, training_loss, valid_loss, e+1, epochs, verbose)
                break


def main():
    n_samples = 5000; num_features = 5
    gaussian_quantiles = load_data(n_samples, num_features)
    X, y = gaussian_quantiles

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    X_train_torch = torch.from_numpy(X_train).float().to(device)
    y_train_torch = torch.from_numpy(y_train).float().to(device)
    X_valid_torch = torch.from_numpy(X_test).float().to(device)
    y_valid_torch = torch.from_numpy(y_test).float().to(device)

    nn = NeuralNetwork(X_train.shape[1], 10, 1)

    nn.train(
        X_train_torch,
        y_train_torch,
        X_valid_torch,
        y_valid_torch,
        epochs = 200000,
        learning_rate = 0.2,
        verbose = 2,
        print_interval = 20000,
        stopper_args = {"threshold": 10, "epsilon": 0}
    )

if __name__ == "__main__":
    main()