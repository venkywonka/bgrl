import logging
import os
from absl import app
from absl import flags
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from bgrl import *

log = logging.getLogger(__name__)
FLAGS = flags.FLAGS
# Dataset.
flags.DEFINE_enum('dataset', 'coauthor-cs',
                  ['amazon-computers', 'amazon-photos', 'coauthor-cs', 'coauthor-physics', 'wiki-cs'],
                  'Which graph dataset to use.')
flags.DEFINE_string('dataset_dir', './data', 'Where the dataset resides.')

# Architecture.
flags.DEFINE_multi_integer('graph_encoder_layer', None, 'Conv layer sizes.')
flags.DEFINE_string('ckpt_path', None, 'Path to checkpoint.')
flags.DEFINE_string('logdir', None, 'Where the checkpoint and logs are stored.')
flags.DEFINE_string('experiment', None, 'ssl or control')


def main(argv):
    # use CUDA_VISIBLE_DEVICES to select gpu
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    log.info('Using {} for evaluation.'.format(device))

    # create log directory
    os.makedirs(FLAGS.logdir, exist_ok=True)
    with open(os.path.join(FLAGS.logdir, 'config.cfg'), "w") as file:
        file.write(FLAGS.flags_into_string())  # save config file

    # setup tensorboard
    writer = SummaryWriter(FLAGS.logdir)

    # load data
    if FLAGS.dataset != 'wiki-cs':
        dataset = get_dataset(FLAGS.dataset_dir, FLAGS.dataset)
    else:
        dataset, train_masks, val_masks, test_masks = get_wiki_cs(FLAGS.dataset_dir)

    data = dataset[0]  # all dataset include one graph
    log.info('Dataset {}, {}.'.format(dataset.__class__.__name__, data))
    data = data.to(device)  # permanently move in gpy memory

    # build networks
    input_size, representation_size = data.x.size(1), FLAGS.graph_encoder_layer[-1]
    encoder = GCN([input_size] + FLAGS.graph_encoder_layer, batchnorm=True) # 512, 256, 128
    load_trained_encoder(encoder, FLAGS.ckpt_path, device)
    encoder.eval()

    # compute representations
    if FLAGS.experiment == 'ssl':
        dataset = compute_representations(encoder, dataset, device, return_dataset=True)

    add_embeddings(dataset, writer, tag=FLAGS.experiment)

    num_classes = dataset.num_classes
    num_features = dataset.num_features

    if FLAGS.dataset != 'wiki-cs':
        train_dataset, val_dataset, test_dataset = split_transductive_dataset(dataset)
    else:
        train_dataset, val_dataset, test_dataset = dataset[train_masks], dataset[val_masks], dataset[test_masks]
    # test_loss, test_f1, model = sequential_search_gnn(train_dataset, val_dataset, test_dataset, device, FLAGS.logdir, writer, num_classes=num_classes, num_features=num_features)
    test_loss, test_f1, model = grid_search_gnn(train_dataset, val_dataset, test_dataset, device, FLAGS.logdir, writer, num_classes=num_classes, num_features=num_features)

    print('Test loss: %.5f, Test F1: %.5f' % (test_loss, test_f1))


if __name__ == "__main__":
    log.info('PyTorch version: %s' % torch.__version__)
    app.run(main)
