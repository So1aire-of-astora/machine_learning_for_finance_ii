import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

def load_data(path):
    data = pd.read_csv(path)
    mess = data[data.Label=="category"]
    print("Turns out the dataset is not cleaned thoroughly. Line 7111:\n{}".format(mess))
    return data.drop(mess.index).reset_index(drop = True)

class Model:
    def __init__(self, data, params):
        self.params = params
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(data.iloc[:,:-1].to_numpy(), 
                                                                                self.transform_label(data["Label"]),
                                                                                test_size = self.params["test_size"],
                                                                                random_state = self.params["random_state"])
        self.model = None

    def transform_label(self, y):
        if self.params["encoding"] == "numeric":
            encoder = LabelEncoder()
        elif self.params["encoding"] == "one-hot": 
            encoder = OneHotEncoder(sparse_output = False)
        y_out = encoder.fit_transform(y.to_numpy().reshape(-1, 1))
        return y_out

    def fit(self):
        raise NotImplementedError
    
    @staticmethod
    def get_metrics(y, y_pred, name: str, verbose = 0):
        # inverse transformation from one-hot to categorical
        # y = np.argmax(y, axis = 1)
        # y_pred = np.argmax(y_pred, axis = 1)
        precision = precision_score(y, y_pred, average = "weighted")
        recall = recall_score(y, y_pred, average = "weighted")
        f1 = f1_score(y, y_pred, average = "weighted")
        if verbose:
            print("""Set: {}\nAccuracy (overall): {:.4f}\nAccuracy (by category): {}\nPrecision (by category): {:.4f}\n
                  Recall (by category): {:.4f}\nF1 Score by category: {:.4f}""".format(name, precision, recall, f1))
    
    def evaluate(self, plot = False):
        if self.model is None:
            raise NotImplementedError
        y_train_pred = self.model.predict(self.X_train)
        y_test_pred = self.model.predict(self.X_test)
        self.get_metrics(self.y_train, y_train_pred, name = "Training")
        self.get_metrics(self.y_test, y_test_pred, name = "Test")
        if plot:
            self.plot(self.X_train, y_train_pred)
            self.plot(self.X_test, y_test_pred)

    def plot(self):
        n_classes = np.unique(self.y_train).shape[0]

class Baseline(Model):
    def __init__(self, data, params):
        super().__init__(data, params)
    
    def fit(self):
        lr = LogisticRegression()
        self.model = GridSearchCV(estimator = lr, param_grid = self.params["lr_grid"], cv = 5)
        self.model.fit(self.X_train, self.y_train)

class MLP(Model, nn.Module):
    def __init__(self, data, params, mlp_params):
        super(MLP, self).__init__(data, params)
        self.mlp_params = mlp_params
        layers = []
        prev_size = mlp_params["input_size"]
        for size in mlp_params["hidden_size"]:
            layers.append(nn.Linear(prev_size, size))
            layers.append(nn.ReLU())
            prev_size = size
        layers.append(nn.Linear(prev_size, mlp_params["output_size"]))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)
    
    @staticmethod
    def correct(y, y_pred):
        predicted_digits = y.argmax(1)
        correct_ones = (predicted_digits == y_pred).type(torch.float)
        return correct_ones.sum().item()
    
    def fit(self, criterion, optimizer, epochs):
        self.train()
        X_train_tensor = torch.tensor(self.X_train, dtype = torch.float32)
        y_train_tensor = torch.tensor(self.y_train, dtype = torch.long)
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size = self.mlp_params["batch_size"], shuffle = True)

        num_batches = len(train_loader)
        num_items = len(train_loader.dataset)

        for e in range(epochs):
            total_loss = 0.0
            total_correct = 0
            
            for X, y in train_loader:
                optimizer.zero_grad()
                pred = self(X)
                loss = criterion(y, pred)
                loss.backward()
                optimizer.step()
                total_loss += loss
                total_correct += self.correct(y, pred)
            
            training_loss = total_loss / num_batches
            accuracy = total_correct / num_items
            print("Epoch %d/%d: Training Loss: %.4fAccuracy: %.4f" %(e+1, epochs, training_loss, accuracy*100))

    def test(self, criterion):
        self.eval()
        X_test_tensor = torch.tensor(self.X_test, dtype = torch.float32)
        y_test_tensor = torch.tensor(self.y_test, dtype = torch.long)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
        test_loader = DataLoader(test_dataset, batch_size = self.mlp_params["batch_size"], shuffle = True)
        num_batches = len(test_loader)
        num_items = len(test_loader.dataset)

        test_loss = 0
        total_correct = 0

        with torch.no_grad():
            for X, y in test_loader:

                pred = self(X)

                loss = criterion(y, pred)
                test_loss += loss

                total_correct += self.correct(y, pred)
        test_loss /= num_batches
        accuracy = total_correct / num_items
        print("Test Loss %.6f\tTest Accuracy %.2f%%" %(test_loss, accuracy*100))

def main():
    data = load_data("./midterm/data_classification.csv")
    lr_model = Baseline(data, params = {"test_size": .3, "random_state": 42, "encoding": "numeric", 
                                        "lr_grid": {'C': [0.01, 0.1, 1, 10, 100],
                                                    'solver': ['lbfgs', 'saga'],
                                                    'max_iter': [100, 500, 1000]}})
    lr_model.fit()
    lr_model.evaluate()


if __name__ == "__main__":
    main()