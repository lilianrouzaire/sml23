import numpy as np
import torch
from torch import nn
from torchvision import datasets

import matplotlib.pyplot as plt


def load_data(flatten=True):
    
    train_set = datasets.MNIST(root="data/", train=True, download=True)
    train_data = train_set.data
    if flatten:
        train_data = train_data.view(train_set.data.shape[0], -1)
    train_data = train_data.float()
    train_targets = convert_to_one_hot_labels(train_set.targets)
    
    test_set = datasets.MNIST(root="data/", train=False, download=True)
    test_data = test_set.data
    if flatten:
        test_data = test_data.view(test_set.data.shape[0], -1)
    test_data = test_data.float()
    test_targets = convert_to_one_hot_labels(test_set.targets)

    # Normalise inplace.
    mu, std = train_data.mean(), train_data.std()
    train_data.sub_(mu).div_(std)
    test_data.sub_(mu).div_(std)
    
    return train_data, train_targets, test_data, test_targets


def convert_to_one_hot_labels(target):
    # Same rows as target, as many columns as the number of classes in target.
    tmp = torch.zeros(target.size(0), target.max() + 1)
    # Puts 1.0 in dimension 1 (columns) in the specified indexs in target.
    tmp.scatter_(1, target.view(-1, 1), 1.0)
    return tmp

def get_linear_model():
    model = nn.Linear(784, 10)
    return model

def get_shallow_model(nb_hidden=100):
    model = nn.Sequential(
        nn.Linear(784, nb_hidden),
        nn.ReLU(),
        nn.Linear(nb_hidden, 10)
    )
    return model

def get_deep_model(nb_hidden=100):
    model = nn.Sequential(
        nn.Linear(784, nb_hidden),
        nn.ReLU(),
        nn.Linear(nb_hidden, nb_hidden),
        nn.ReLU(),
        nn.Linear(nb_hidden, nb_hidden),
        nn.ReLU(),
        nn.Linear(nb_hidden, 10)
    )
    return model

def get_dropout_model(nb_hidden=100, p=0.5):
    model = nn.Sequential(
        nn.Linear(784, nb_hidden),
        nn.ReLU(),
        nn.Dropout(p=p),
        nn.Linear(nb_hidden, nb_hidden),
        nn.ReLU(),
        nn.Dropout(p=p),
        nn.Linear(nb_hidden, nb_hidden),
        nn.ReLU(),
        nn.Linear(nb_hidden, 10)
    )
    return model


def train_model(model, nb_epochs=100, lambda_l2=0, lr=1e-1, batch_size=100, flatten_input=True):
    
    # Load all data.
    train_data, train_targets, test_data, test_targets = load_data(flatten_input)
    
    # Cross entropy loss and stochastic gradient descent.
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    # Initialise list of outputs.
    train_losses, test_losses, train_accuracies, test_accuracies = [], [], [], []

    for e in range(nb_epochs):
        print(f"\rEpoch {e+1}", end="")
        
        # Iterate over train data in batches.
        for data, targets in zip(train_data.split(batch_size), train_targets.split(batch_size)):
            
            # Pass data through the model and compute loss.
            output = model(data)
            loss = criterion(output, targets)
            
            # Apply regularisation.
            if lambda_l2 > 0:
                for p in model.parameters():
                    loss += lambda_l2 * p.pow(2).sum()
            
            # Reset gradients to zero, compute the backward pass and update parameters.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_losses.append(loss.item()) # Append loss for last batch of the epoch.

        with torch.no_grad():
            train_preds = model(train_data)
            train_correct = (train_preds.argmax(dim=1) == train_targets.argmax(dim=1)).float().sum()
            train_accuracy = train_correct / len(train_targets)
        train_accuracies.append(train_accuracy.item())

            
        test_loss, test_accuracy, _ = eval_model(model, criterion, test_data, test_targets)
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)
        
    print("\nTraining finished!")
            
    return model, train_losses, test_losses, train_accuracies, test_accuracies


def eval_model(model, criterion, test_data, test_targets):
    
    # Switch the model to eval mode (if you had any dropout or batchnorm, they are turned off).
    model.eval()
    
    # Ensure the gradients are not collected for model evaluation.
    with torch.no_grad():
        test_pred = model(test_data)
        test_loss = criterion(test_pred, test_targets).item()
        test_correct = (test_pred.argmax(dim=1) == test_targets.argmax(dim=1)).float().sum()
        test_accuracy = test_correct / len(test_targets)
        cm = confusion_matrix(test_targets.argmax(dim=1), test_pred.argmax(dim=1))

    
    # Switch back to training mode.
    model.train()
    
    return test_loss, test_accuracy.item(), cm

def plot_example(X, y):
    fig, ax = plt.subplots(3, 3, figsize=(10, 10))
    for i in range(3):
        for j in range(3):
            label = y[i*3+j].argmax()
            ax[i, j].imshow(X[i*3+j].view(28, 28), cmap="gray")
            ax[i, j].set_title(f"Label: {label}")
            ax[i, j].axis("off")
    plt.show()

def confusion_matrix(y_true, y_pred):
    classes = np.unique(np.concatenate((y_true, y_pred)))
    num_classes = len(classes)
    cm = np.zeros((num_classes, num_classes), dtype=int)
    
    for i in range(len(y_true)):
        true_class = y_true[i]
        pred_class = y_pred[i]
        cm[true_class][pred_class] += 1
    
    return cm
