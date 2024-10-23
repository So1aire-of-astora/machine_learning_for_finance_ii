import torch, torchvision
import numpy as np
import torch.nn as nn
from torchvision import datasets
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
import lightning as L

if torch.cuda.is_available():
    # torch.set_default_device("cuda")
    device = torch.device("cuda:0")
    # device = torch.device("cpu")

def load_data(size, njobs):
    transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = datasets.CIFAR10(root = "./data", train = True, download = True, transform = transform)
    testset = datasets.CIFAR10(root = "./data", train = False, download = True, transform = transform)

    train_set, val_set = torch.utils.data.random_split(trainset, [.8, .2])

    # batch_size = ?
    # You should use as many cores you have on your laptop
    # num_workers = ?

    # Fill in the options for both data loaders. Warning: the training dataloader should shuffle the data
    train_loader = torch.utils.data.DataLoader(train_set, batch_size = size, shuffle = True, num_workers = njobs, pin_memory = True)
    valid_loader = torch.utils.data.DataLoader(val_set, batch_size = size, shuffle = False, num_workers = njobs, pin_memory = True)
    test_loader = torch.utils.data.DataLoader(testset, batch_size = size, shuffle = False, num_workers = njobs, pin_memory = True)

    return train_loader, valid_loader, test_loader

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

class Net(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list, output_dim: int, dropout_rate = 0.):
        super(Net, self).__init__()
        layers = [nn.Flatten()]
        dims = [input_dim] + hidden_dims
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
        layers.append(nn.Linear(dims[-1], output_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

class LitNet(L.LightningModule):
    def __init__(self, model, criterion, optimizer):
        super().__init__()
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer

    def configure_optimizers(self):
        return self.optimizer

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        X, y = batch
        output = self.model(X)
        loss = self.criterion(output, y)
        self.log("Training Loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        # training_step defines the train loop.
        X, y = batch
        output = self.model(X)
        loss = self.criterion(output, y)
        self.log("Validation Loss", loss)

    def test_step(self, batch, batch_idx):
        # training_step defines the train loop.
        X, y = batch
        output = self.model(X)
        loss = self.criterion(output, y)
        self.log("Test Loss", loss)


def correct(output, target):
    predicted_digits = output.argmax(1)                            # pick digit with largest network output
    correct_ones = (predicted_digits == target).type(torch.float)  # 1.0 for correct, 0.0 for incorrect
    return correct_ones.sum().item()    

def train(train_loader, valid_loader, model, criterion, optimizer, epochs, stopper_args):
    stopper = EarlyStopper(**stopper_args)
    for e in range(epochs):
        model.train()

        num_batches = len(train_loader)
        num_items = len(train_loader.dataset)

        total_loss = 0
        total_correct = 0
        
        for data, target in train_loader:
            data = data.to(device)
            target = target.to(device)

            output = model(data)

            loss = criterion(output, target)
            total_loss += loss

            total_correct += correct(output, target)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        
        training_loss = total_loss / num_batches
        accuracy_train = total_correct / num_items
        valid_loss, accuracy_valid = test(valid_loader, model, criterion, verbose = 0)
        if not (e+1) % 10:
            print("Epoch %d/%d: Training Loss %.6f\tValidation Loss %.6f\tTraining Accuracy %.2f%%\tValidation Accuracy %.2f%%" %(e+1, epochs, training_loss, valid_loss, accuracy_train*100, accuracy_valid*100))
        if stopper.early_stop(valid_loss):
            print("[Early stopping]\nEpoch %d/%d: Training Loss %.6f\tValidation Loss %.6f\tTraining Accuracy %.2f%%\tValidation Accuracy %.2f%%" %(e+1, epochs, training_loss, valid_loss, 
                                                                                                                                                    accuracy_train*100, accuracy_valid*100))
            break       


def test(test_loader, model, criterion, verbose = 1):
    '''
    test or validate
    '''
    model.eval()
    num_batches = len(test_loader)
    num_items = len(test_loader.dataset)

    test_loss = 0
    total_correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            target = target.to(device)

            output = model(data)

            loss = criterion(output, target)
            test_loss += loss

            total_correct += correct(output, target)
    test_loss /= num_batches
    accuracy = total_correct / num_items
    if not verbose:
        return test_loss, accuracy
    print("Test Loss %.6f\tTest Accuracy %.2f%%" %(test_loss, accuracy*100))



def main():
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    input_dim = 3 * 32 * 32; hidden_dim = [1024, 512, 256]; output_dim = len(classes); learning_rate = .0001; num_epochs = 200; batch_size = 128; njobs = 32; dropout = .5; optimizer_decay = 1e-4
    train_loader, valid_loader, test_loader = load_data(batch_size, njobs)

    model = Net(input_dim, hidden_dim, output_dim, dropout).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate, weight_decay = optimizer_decay)

    print("Hidden dim: {}; lr: {}; batch_size: {}; dropout: {}".format(hidden_dim, learning_rate, batch_size, dropout))
    # lit_mlp = LitNet(model, criterion, optimizer)
    # lit_trainer = L.Trainer(max_epochs = num_epochs)
    # lit_trainer.fit(lit_mlp, train_loader)

    stopper_args = {"threshold": 20, "epsilon": 1e-4}
    train(train_loader, valid_loader, model, criterion, optimizer, num_epochs, stopper_args)
    test(test_loader, model, criterion)



if __name__ == "__main__":
    main()