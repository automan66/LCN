import torch
import torch.nn as nn
import torch.nn.functional as F


# encoding the input graph
class Encoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels, activation, num_layers,
                 base_model):
        super(Encoder, self).__init__()
        self.base_model = base_model
        self.num_layers = num_layers

        # define 6-layer GCNs
        self.conv = [base_model(in_channels, out_channels)]
        for _ in range(num_layers - 1):
            self.conv.append(base_model(out_channels, out_channels))

        self.conv = nn.ModuleList(self.conv)
        self.dpout = nn.Dropout(0.5)
        self.activation = activation

    def forward(self, x, edge_index, dataset):

        outs = []

        x = self.dpout(x)
        x = self.activation(self.conv[0](x, edge_index))
        outs.append(x)

        for i in range(1, len(self.conv)):
            if dataset != 'PubMed':
                x = self.dpout(x)
            x = self.activation(self.conv[i](x, edge_index))
            outs.append(x)
            x = x + outs[0]

        return outs[0], outs[-1]


# decoding the feature for contrastive loss
class Decoder(torch.nn.Module):
    def __init__(self, num_inp, num_proj_hidden, num_out):
        super(Decoder, self).__init__()
        self.fc1 = torch.nn.Linear(num_inp, num_proj_hidden)
        self.fc2 = torch.nn.Linear(num_proj_hidden, num_out)

    def forward(self, x):
        z = F.leaky_relu(self.fc1(x))
        # z = F.elu(self.fc1(x))
        return self.fc2(z)


# computing the cosine similarity
def sim(z1, z2):
    z1 = F.normalize(z1)
    z2 = F.normalize(z2)
    return torch.mm(z1, z2.t())


# computing the contrastive loss
def cl_loss(z1, z2, tau):
    f = lambda x: torch.exp(x / tau)
    refl_sim = f(sim(z1, z1))
    between_sim = f(sim(z1, z2))

    # # positive samples
    # pos = between_sim.diag()  
    # # negative samples selected from the same layer
    # neg_inter = refl_sim.sum(1) - refl_sim.diag()
    # # negative samples selected from the different layer 
    # neg_cross = between_sim.sum(1) - between_sim.diag()  
    
    # return -torch.log(pos / (pos + neg_inter + neg_cross))
    
    return -torch.log(between_sim.diag() /
                      (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))


# computing the final loss
def CL_loss(projection1, projection2, z1, z2, tau):
    h1 = projection1(z1)
    h2 = projection2(z2)

    l1 = cl_loss(h1, h2, tau)
    l2 = cl_loss(h2, h1, tau)

    ret = (l1 + l2) * 0.5
    ret = ret.mean()
    return ret


# to fit the model and loss into a single gpu, we use the following loss function ...
# to train deep LCN and wide LCN on Pubmed and DBLP dataset
def half_CL_loss(projection1, projection2, z1, z2, tau):
    h1 = projection1(z1)
    h2 = projection2(z2)
    ret = cl_loss(h2, h1, tau)
    ret = ret.mean()
    return ret