import random

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

def add_embeddings(dataset, writer, tag):
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
    metadata_header = ['label_{}'.format(i) for i in range(dataset.num_classes)]
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

