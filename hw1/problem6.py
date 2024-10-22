import torch, torchvision
import numpy as np
import torch.nn as nn
from torchvision import datasets
import torchvision.transforms as transforms
from matplotlib import pyplot as plt

if torch.cuda.is_available():
    # torch.set_default_device("cuda")
    device = torch.device("cuda:0")

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
    train_loader = torch.utils.data.DataLoader(train_set, batch_size = size, shuffle = True, num_workers = njobs)
    valid_loader = torch.utils.data.DataLoader(val_set, batch_size = size, shuffle = False, num_workers = njobs)
    test_loader = torch.utils.data.DataLoader(testset, batch_size = size, shuffle = False, num_workers = njobs)

    return train_loader, valid_loader, test_loader

class Net(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Net, self).__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, x):
        return self.layers(x)

def correct(output, target):
    prediction = output.argmax(1)                            # pick digit with largest network output
    correct_ones = (prediction == target).type(torch.float)  # 1.0 for correct, 0.0 for incorrect
    return correct_ones.sum().item()  

def train(train_loader, valid_loader, model, criterion, optimizer, epochs):
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

        if not (e+1) % 500:
            valid_loss, accuracy_valid = test(valid_loader, model, criterion, verbose = 0)
            print("Epoch %d/%d: Training Loss %.6f\tValidation Loss %.6f\tTraining accuracy %.2f%%\tValidation accuracy %.2f%%" %(e+1, epochs, training_loss, valid_loss, accuracy_train*100, accuracy_valid*100))
        


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
    input_dim = 3 * 32 * 32; hidden_dim = 50; output_dim = len(classes); learning_rate = .01; num_epochs = 10000; batch_size = 64; njobs = 32
    train_loader, valid_loader, test_loader = load_data(batch_size, njobs)
    
    model = Net(input_dim, hidden_dim, output_dim).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.RMSprop(model.parameters(), lr = learning_rate)

    train(train_loader, valid_loader, model, criterion, optimizer, num_epochs)
    test(test_loader, model, criterion)



if __name__ == "__main__":
    main()