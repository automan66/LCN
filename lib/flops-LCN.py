import argparse
import random
from time import perf_counter as t
import yaml
from yaml import SafeLoader

import torch
import torch_geometric.transforms as T
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.datasets import Planetoid, CitationFull
from torch_geometric.nn import GCNConv

from lib.model import Encoder, Decoder, CL_loss
from lib.eval import label_classification
import numpy as np
import os

from fvcore.nn import FlopCountAnalysis



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Cora')
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()

    config = yaml.load(open('lib/config.yaml'),
                       Loader=SafeLoader)[args.dataset]

    torch_seed = config['seed']
    torch.manual_seed(torch_seed)
    torch.cuda.manual_seed(torch_seed)
    torch.cuda.manual_seed_all(torch_seed)

    np_seed = config['np_seed']
    random.seed(np_seed)
    np.random.seed(np_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # hyper-parameters
    learning_rate = config['learning_rate']
    num_hidden = config['num_hidden']
    num_proj_hidden = config['num_proj_hidden']
    activation = ({'relu': F.relu, 'prelu': nn.PReLU()})[config['activation']]
    base_model = ({'GCNConv': GCNConv})[config['base_model']]
    num_layers = config['num_layers']
    tau = config['tau']
    num_epochs = config['num_epochs_LCN']
    weight_decay = config['weight_decay']

    # get dataset
    def get_dataset(path, name):
        assert name in ['Cora', 'CiteSeer', 'PubMed', 'DBLP']

        if name == 'DBLP':
            return CitationFull(path, name, T.NormalizeFeatures())
        else:
            return Planetoid(path, name, "public")

    path = './dataset/'
    dataset = get_dataset(path, args.dataset)
    data = dataset[0]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = data.to(device)

    # define model
    encoder = Encoder(dataset.num_features,
                      num_hidden,
                      activation,
                      num_layers,
                      base_model=base_model).to(device)

    x = torch.randn(data.x.size()).to(device)
    flops_analysis = FlopCountAnalysis(encoder, (x, data.edge_index))
    total_flops = flops_analysis.total()
    print(f"Total FLOPs: {total_flops}")
    print(f"FLOPs: {total_flops / 1e9:.2f} GFLOPs")
