# LCN
Code for "Layer-wise Contrastive Network for Unsupervised Graph Representation Learning"

This directory includes a PyTorch implementation of [Layer-wise Contrastive Network for Unsupervised Graph Representation Learning]. Note that the code is built on top of the official PyTorch implementation of GRACE (https://github.com/CRIPAC-DIG/GRACE).

### Dependencies
* Python 3.6 
* torch 1.6.0
* torch-geometric 1.6.1
* sklearn 0.21.3 

You can also install the appropriate version of torch-geometric according to your environment by referring to the official site (https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html). 

### Datasets

The datasets (e.g., Cora and Citeseer) are stored in the `dataset` directory.

### Evaluation and Training

To reproduce the results on a specific dataset, you can directly load the corresponding pretrained checkpoint from the `checkpoints` directory and run the following commands:

```bash
# Evaluate on the Cora dataset
python test-LCN.py \
    --dataset Cora \
    --gpu 0 \
    --checkpoint checkpoints/Cora/LCN/enc.pth

python test-wide-LCN.py \
    --dataset Cora \
    --gpu 0 \
    --checkpoint checkpoints/Cora/wide_LCN/enc.pth
```

To train the model from scratch on a specific dataset, run:

```bash
# Train on the Cora dataset
python train-LCN.py \
    --dataset Cora \
    --gpu 0

python train-wide-LCN.py \
    --dataset Cora \
    --gpu 0
```
