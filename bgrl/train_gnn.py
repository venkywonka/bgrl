import os
import torch
from torch_geometric.data import DataLoader
from sklearn.model_selection import ParameterGrid
from torchmetrics.classification import MultilabelF1Score
from tqdm import tqdm

from bgrl.models import GCN, GATv2

EPOCHS = 500

def grid_search_gnn(train_dataset, val_dataset, test_dataset, device, logdir, writer, num_classes=None, num_features=None):
    """
    Perform a grid search over the defined hyperparameter space to find the best GNN model.

    :param train_dataset: Dataset for training
    :param val_dataset: Dataset for validation
    :param test_dataset: Dataset for testing
    :param device: Device to run the computations on (e.g. 'cpu' or 'cuda')
    :return: Test loss, Test F1 score, and the trained model
    """
    # Define the hyperparameter grid
    param_grid = {
        'lr': [1e-3],
        'layer_sizes': [[512, 128]],
        'batchnorm': [False],
        # 'weight_decay': [0.001],
    }

    grid = ParameterGrid(param_grid)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=0)

    best_val_f1 = float('-inf')
    best_params = None
    iter=0

    # Iterate through all the hyperparameter combinations
    for params in grid:
        # Create and train the GNN with the current hyperparameters
        layer_sizes = params['layer_sizes']
        layer_sizes.insert(0, num_features)
        layer_sizes.append(num_classes)

        model = GATv2(layer_sizes=layer_sizes, batchnorm=params['batchnorm'], layernorm=~params['batchnorm']).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'])

        # Train the model and evaluate its performance using cross-validation or a validation set
        val_loss, val_f1 = train_and_evaluate(model, optimizer, train_loader, val_loader, device, iter, num_classes=num_classes)
        iter += 1
        # write the val scores along with params to a json file

        # Update the best score and parameters if needed
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_params = params

    print("Best score:", best_val_f1)
    print("Best parameters:", best_params)

    os.makedirs(os.path.join(logdir, 'gnn'), exist_ok=True)
    torch.save(model.state_dict(), os.path.join(logdir, 'gnn', f'{model.__str__}_{best_params.__str__()}.pt'))
    print(f"Saved model to model_{best_params.__str__()}.pt")

    test_loss, test_f1 = eval(model, test_loader, device, num_classes=num_classes)

    return test_loss, test_f1, model

def train_and_evaluate(model, optimizer, train_loader, valid_loader, device, iter, valid_every=20, num_classes=None):
    """
    Train the model and evaluate it on the validation set.

    :param model: The GNN model
    :param optimizer: The optimizer for training the model
    :param train_loader: The DataLoader for the training dataset
    :param valid_loader: The DataLoader for the validation dataset
    :param device: The device to run the computations on (e.g. 'cpu' or 'cuda')
    :return
    :return: Validation loss and validation F1 score
    """
    for epoch in tqdm(range(1, EPOCHS+1), desc=f"grid: {iter}"):
        train(model, optimizer, train_loader, device)
        if epoch % valid_every == 0:
            val_loss, val_f1 = eval(model, valid_loader, device, num_classes=num_classes)
            print(f"Epoch: {epoch}, Val Loss: {val_loss}, Val F1: {val_f1}")

    return val_loss, val_f1

def train(model, optimizer, train_loader, device):
    """
    Train the GNN model on the given dataset.

    :param model: The GNN model
    :param optimizer: The optimizer for training the model
    :param train_loader: The DataLoader for the training dataset
    :param device: The device to run the computations on (e.g. 'cpu' or 'cuda')
    """
    model.train()

    for batch in train_loader:
        optimizer.zero_grad()
        batch = batch.to(device)
        out = model(batch)
        # import pdb; pdb.set_trace()
        if batch.y.ndim == 1:
            loss = torch.nn.CrossEntropyLoss()(out, batch.y.squeeze())
        else:
            loss = torch.nn.BCEWithLogitsLoss()(out, batch.y)
        loss.backward()
        optimizer.step()

def onehot_ifnot(tensor):
    """
    Return one-hot encoding of the last dimension of the tensor if the tensor is not one-hot encoded.
    """
    # Check if the tensor is already one-hot encoded by counting the number of unique values in each row
    # is_onehot = torch.all(torch.tensor(torch.unique(tensor, dim=-1).size(-1) == 2)).item()
    is_onehot = False if tensor.squeeze().ndim == 1 else True
    # If not, convert it to one-hot encoding using torch.nn.functional.one_hot
    if not is_onehot:
        # Get the number of classes from the maximum value in the tensor
        num_classes = tensor.max() + 1
        # Apply one-hot encoding to the last dimension
        tensor = torch.nn.functional.one_hot(tensor, num_classes=num_classes)
    # Return the tensor
    return tensor

def eval(model, valid_loader, device, num_classes=None):
    """
    Evaluate the GNN model on the given dataset.

    :param model: The GNN model
    :param valid_loader: The DataLoader for the validation dataset
    :param device: The device to run the computations on (e.g. 'cpu' or 'cuda')
    :return: Loss and F1 score
    """
    model.eval()

    total_nodes_n = 0
    total_loss = 0
    macro_f1 = MultilabelF1Score(num_labels=num_classes, average='macro').to(device)

    gt = []
    pred = []

    for batch in valid_loader:
        batch = batch.to(device)
        with torch.no_grad():
            out = model(batch)
            loss = torch.nn.functional.cross_entropy(out, batch.y)
            total_loss += loss * batch.x.shape[0]
            total_nodes_n += batch.x.shape[0]
            gt += batch.y.cpu().tolist()
            pred += out.cpu().tolist()
            macro_f1.update(out.to(device), onehot_ifnot(batch.y.long()).to(device))

    val_f1 = macro_f1.compute()
    return total_loss / total_nodes_n, val_f1
