import random
from torch_geometric.transforms import RandomNodeSplit
from torch_geometric.utils import subgraph
from torch_geometric.data import InMemoryDataset

import numpy as np
import pandas as pd
import torch


def set_random_seeds(random_seed=0):
    r"""Sets the seed for generating random numbers."""
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)

def add_embeddings(dataset, writer, tag, multilabel=False):
    """
    Adds embeddings with label metadata to tensorboard

    Parameters:
        dataset: A torch_geometric.datasets.PPI dataset
        writer: A SummaryWriter instance from torch.utils.tensorboard
        tag: A string to name the embeddings in TensorBoard
    """
    # Prepare the embeddings tensor
    all_embeddings = []

    # Prepare the metadata
    if multilabel:
        metadata_header = ['label_{}'.format(i) for i in range(dataset.num_classes)]
    else:
        metadata_header = ['label']
    metadata_df = pd.DataFrame(columns=metadata_header)

    # Iterate through the dataset to collect embeddings and labels
    for data in dataset:
        all_embeddings.append(data.x)

        # Convert multi-label one-hot encoding to a list of label indices for each node
        node_labels_df = pd.DataFrame(data.y.cpu().numpy(), columns=metadata_header)
        metadata_df = pd.concat([metadata_df, node_labels_df], ignore_index=True)


    # Concatenate all the embeddings into a single tensor
    embeddings_tensor = torch.cat(all_embeddings, dim=0)

    # Write the embeddings to TensorBoard
    writer.add_embedding(embeddings_tensor, metadata=metadata_df.values.tolist(), metadata_header=metadata_header, tag=tag)

def split_transductive_dataset(dataset, train_ratio=0.7, val_ratio=0.1, seed=42):
    # Assume data is the original data object
    data = dataset[0]
    # num_classes
    num_classes = dataset.num_classes
    # Create node masks for 80/10/10 split
    data = RandomNodeSplit(split="test_rest")(data).to(data.x.device)

    # Create subgraphs for each split
    edge_index, _ = subgraph(data.train_mask, data.edge_index, relabel_nodes=True)
    train_data = data.__class__(edge_index=edge_index, num_classes=num_classes)
    train_data.x = data.x[data.train_mask]
    train_data.y = data.y[data.train_mask]

    edge_index, _ = subgraph(data.val_mask, data.edge_index, relabel_nodes=True)
    val_data = data.__class__(edge_index=edge_index, num_classes=num_classes)
    val_data.x = data.x[data.val_mask]
    val_data.y = data.y[data.val_mask]

    edge_index, _ = subgraph(data.test_mask, data.edge_index, relabel_nodes=True)
    test_data = data.__class__(edge_index=edge_index, num_classes=num_classes)
    test_data.x = data.x[data.test_mask]
    test_data.y = data.y[data.test_mask]

    return [train_data], [val_data], [test_data]
