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
from lib.eval import label_classification, cluster_test
import numpy as np
import os


def train(encoder, decoder, data_trans, x, edge_index, tau, dataset):
    encoder.train()
    decoder.train()
    data_trans.train()
    optimizer.zero_grad()

    w = 0.01
    z1, z2 = encoder(x, edge_index, dataset)
    loss_feat = CL_loss(decoder, decoder, z1, z2, tau)
    loss_data = CL_loss(data_trans, decoder, x, z1, tau)
    loss = loss_feat + loss_data * w
    loss.backward()
    optimizer.step()

    return loss.item()


def test(encoder, x, edge_index, y, dataset):
    encoder.eval()
    _, z = encoder(x, edge_index, dataset)
    label_classification(z, y, ratio=0.1)
    r = cluster_test(z.detach().cpu(), torch.unique(y).size()[0], y.detach().cpu(), args.cluster_random_state)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='CiteSeer')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--cluster_random_state', type=int, default='12345')
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
    decoder = Decoder(num_hidden, num_proj_hidden, num_hidden).to(device)
    data_trans = Decoder(dataset.num_features, num_proj_hidden, num_hidden).to(device)
    params = list(encoder.parameters()) + list(decoder.parameters()) + list(data_trans.parameters())
    optimizer = torch.optim.Adam(params,
                                 lr=learning_rate,
                                 weight_decay=weight_decay)

    # start training
    start = t()
    prev = start

    for epoch in range(1, num_epochs + 1):
        loss = train(encoder, decoder, data_trans, data.x, data.edge_index, tau, args.dataset)

        if epoch % 50 == 0:
            now = t()
            print(
                f'Epoch_{epoch:04d}, loss: {loss:.4f}, '
                f'epoch_time: {now - prev:.4f}, total_time: {now - start:.4f}')
            prev = now

    # save model
    model_dir = 'checkpoints/' + args.dataset + '/LCN/'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    torch.save(encoder.state_dict(), model_dir + 'enc.pth')
    torch.save(decoder.state_dict(), model_dir + 'dec.pth')

    # start testing
    print(f'== test ==')
    test(encoder, data.x, data.edge_index, data.y, args.dataset)
