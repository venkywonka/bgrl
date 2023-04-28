import logging

from absl import app
from absl import flags
import torch
from torch_geometric.datasets import PPI

from bgrl import *

log = logging.getLogger(__name__)
FLAGS = flags.FLAGS

# Dataset.
flags.DEFINE_string('dataset_dir', './data', 'Directory where the dataset resides.')
flags.DEFINE_string('ckpt_path', None, 'Path to checkpoint.')

def main(argv):
    """
    Load the PPI dataset, build the GraphSAGE_GCN encoder and compute enriched representations.
    Then, perform grid search on the GNN model and print the test F1-score.

    :param argv: Command-line arguments
    """
    # Use CUDA_VISIBLE_DEVICES to select GPU
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    log.info('Using {} for evaluation.'.format(device))

    # Load data
    train_dataset = PPI(FLAGS.dataset_dir, split='train')
    val_dataset = PPI(FLAGS.dataset_dir, split='val')
    test_dataset = PPI(FLAGS.dataset_dir, split='test')
    log.info('Dataset {}, graph 0: {}.'.format(train_dataset.__class__.__name__, train_dataset[0]))

    # Build networks
    input_size, representation_size = train_dataset.num_node_features, 512
    encoder = GraphSAGE_GCN(input_size, 512, 512)
    load_trained_encoder(encoder, FLAGS.ckpt_path, device)
    encoder.eval()

    # Compute enriched representations from pretrained BGRL encoder
    # train_dataset = compute_representations(encoder, train_dataset, device, return_dataset=True)
    # val_dataset = compute_representations(encoder, val_dataset, device, return_dataset=True)
    # test_dataset = compute_representations(encoder, test_dataset, device, return_dataset=True)

    test_loss, test_f1, model = grid_search_gnn(train_dataset, val_dataset, test_dataset, device)

    print('Test F1-score: %.5f' % test_f1)


if __name__ == "__main__":
    log.info('PyTorch version: %s' % torch.__version__)
    app.run(main)
