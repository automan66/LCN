import argparse
import random
import yaml
from yaml import SafeLoader

import torch
import torch.nn.functional as F
import torch.nn as nn
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid, CitationFull
from torch_geometric.nn import GCNConv
import numpy as np
import os

from lib.model import Encoder
from lib.eval import label_classification, cluster_test


def test(encoder, x, edge_index, y, dataset):
    encoder.eval()

    with torch.no_grad():
        _, z = encoder(x, edge_index, dataset)

    print("===== Classification =====")
    label_classification(z, y, ratio=0.1)

    print("===== Clustering =====")
    cluster_test(
        z.cpu(),
        torch.unique(y).size(0),
        y.cpu(),
        args.cluster_random_state
    )


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='CiteSeer')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--cluster_random_state', type=int, default=12345)
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='Path of pretrained encoder.'
    )

    args = parser.parse_args()

    #################################################
    # load config
    #################################################

    config = yaml.load(
        open('lib/config.yaml'),
        Loader=SafeLoader
    )[args.dataset]

    torch.manual_seed(config['seed'])
    np.random.seed(config['np_seed'])
    random.seed(config['np_seed'])

    #################################################
    # hyper-parameters
    #################################################

    num_hidden = config['num_hidden']
    activation = ({
        'relu': F.relu,
        'prelu': nn.PReLU()
    })[config['activation']]

    base_model = ({
        'GCNConv': GCNConv
    })[config['base_model']]

    num_layers = config['num_layers']

    #################################################
    # dataset
    #################################################

    def get_dataset(path, name):
        assert name in ['Cora', 'CiteSeer', 'PubMed', 'DBLP']

        if name == 'DBLP':
            return CitationFull(path, name, T.NormalizeFeatures())
        else:
            return Planetoid(path, name, "public")

    dataset = get_dataset('./dataset/', args.dataset)
    data = dataset[0]

    device = torch.device(
        f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'
    )

    data = data.to(device)

    #################################################
    # build model
    #################################################

    encoder = Encoder(
        dataset.num_features,
        num_hidden,
        activation,
        num_layers,
        base_model=base_model
    ).to(device)

    #################################################
    # load checkpoint
    #################################################

    if args.checkpoint is None:
        checkpoint = f'checkpoints/{args.dataset}/LCN/enc.pth'
    else:
        checkpoint = args.checkpoint

    if not os.path.exists(checkpoint):
        raise FileNotFoundError(f'Checkpoint not found: {checkpoint}')

    encoder.load_state_dict(
        torch.load(checkpoint, map_location=device)
    )

    print("Load model from:", checkpoint)

    #################################################
    # test
    #################################################

    test(
        encoder,
        data.x,
        data.edge_index,
        data.y,
        args.dataset
    )