import torch
from torch_geometric.data import DataLoader
from sklearn.model_selection import ParameterGrid
from torchmetrics.classification import MultilabelF1Score
from tqdm import tqdm

from bgrl.models import GCN, GATv2

EPOCHS = 500

def grid_search_gnn(train_dataset, val_dataset, test_dataset, device):
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
        'batchnorm': [True],
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
        layer_sizes.insert(0, train_dataset.num_node_features)
        layer_sizes.append(train_dataset.num_classes)

        model = GATv2(layer_sizes=layer_sizes, batchnorm=params['batchnorm'], layernorm=~params['batchnorm']).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'])

        # Train the model and evaluate its performance using cross-validation or a validation set
        val_loss, val_f1 = train_and_evaluate(model, optimizer, train_loader, val_loader, device, iter)
        iter += 1

        # Update the best score and parameters if needed
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_params = params

    print("Best score:", best_val_f1)
    print("Best parameters:", best_params)

    torch.save(model.state_dict(), f'model_{best_params.__str__()}.pt')
    print(f"Saved model to model_{best_params.__str__()}.pt")

    test_loss, test_f1 = eval(model, test_loader, device)

    return test_loss, test_f1, model

def train_and_evaluate(model, optimizer, train_loader, valid_loader, device, iter, valid_every=20):
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
            val_loss, val_f1 = eval(model, valid_loader, device)
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
        loss = torch.nn.BCEWithLogitsLoss()(out, batch.y)
        loss.backward()
        optimizer.step()

def eval(model, valid_loader, device):
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
    macro_f1 = MultilabelF1Score(num_labels=valid_loader.dataset.num_classes, average='macro').to(device)

    for batch in valid_loader:
        batch = batch.to(device)
        with torch.no_grad():
            out = model(batch)
            loss = torch.nn.functional.cross_entropy(out, batch.y)
            total_loss += loss * batch.x.shape[0]
            total_nodes_n += batch.x.shape[0]

            macro_f1.update(out.to(device), batch.y.long().to(device))

    val_f1 = macro_f1.compute()
    return total_loss / total_nodes_n, val_f1
