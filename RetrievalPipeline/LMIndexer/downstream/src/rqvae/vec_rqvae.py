from __future__ import print_function

import os
import json
import matplotlib.pyplot as plt
import numpy as np
import random

from sklearn.manifold import TSNE

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim

import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.utils import make_grid

from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from argparse import ArgumentParser

from vec_model import RQVAE

from IPython import embed

parser = ArgumentParser()
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--depth', type=int, default=4)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--num_epochs', type=int, default=50)
parser.add_argument('--code_embedding_dim', type=int, default=64)
parser.add_argument('--codebook_size', type=int, default=256)
parser.add_argument('--commitment_cost', type=float, default=0.25)
parser.add_argument('--decay', type=float, default=0.99)
parser.add_argument('--learning_rate', type=float, default=0.001)
parser.add_argument('--embedding_dir', type=str, required=True)
parser.add_argument('--code_save_dir', type=str, required=True)
parser.add_argument('--base_model', type=str, required=True)
parser.add_argument('--domain', type=str, required=True)
parser.add_argument('--num_workers', type=int, required=True)

args = parser.parse_args()

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic = True

# Utils
def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def show(img):
    npimg = img.numpy()
    fig = plt.imshow(np.transpose(npimg, (1,2,0)), interpolation='nearest')
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)

def id_list2str(input_list):
    input_list = [[str(s) for s in ll] for ll in input_list]
    return [','.join(s) for s in input_list]

# load data
base_model = args.base_model.split('/')[-1]
training_data = torch.load(os.path.join(args.embedding_dir, f'{base_model}-embed.pt'), map_location='cpu').numpy()
# data_variance = np.var(training_data / 255.0)
data_variance = 1
input_dim = training_data.shape[-1]

training_tensor = torch.load(os.path.join(args.embedding_dir, f'{base_model}-embed.pt'), map_location='cpu').float()
training_dataset = torch.utils.data.TensorDataset(training_tensor)

# config
depth = args.depth
num_workers = args.num_workers


now_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")


batch_size = args.batch_size
num_epochs = args.num_epochs

num_hiddens = (512, 256, 128)

embedding_dim = args.code_embedding_dim
num_embeddings = args.codebook_size

model_name = f"rqvae-{num_embeddings}-depth{depth}-{args.domain}"
tb_dir_path = './logs/%s/tb_log/%s'%(model_name, now_time)

commitment_cost = args.commitment_cost
decay = args.decay
learning_rate = args.learning_rate


## train
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

training_loader = DataLoader(training_dataset,
                             batch_size=batch_size,
                             shuffle=True,
                             pin_memory=True,
                             num_workers=num_workers)

infer_loader = DataLoader(training_dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            pin_memory=True,
                            num_workers=num_workers)

model = RQVAE(input_dim, num_hiddens, num_embeddings, embedding_dim, commitment_cost, decay=decay, depth=depth).to(device)

optimizer = optim.Adam(model.parameters(), lr=learning_rate, amsgrad=False)

model.train()
train_res_recon_error = []
train_res_perplexity = []

# train_on = not os.path.exists('./%s.pth'%model_name)
train_on = True

if train_on:
    # tb_writer = SummaryWriter(log_dir=tb_dir_path)
    # with torch.autograd.set_detect_anomaly(True):
    for epoch in range(num_epochs):
        model.train()

        # for (data, _) in tqdm(training_loader):
        for batch in training_loader:
            data = batch[0].to(device)
            optimizer.zero_grad()

            loss_commit, data_recon, perplexity = model.compute_loss(data)
            loss_recon = F.mse_loss(data_recon, data) / data_variance
            loss = loss_recon + loss_commit
            loss.backward()

            optimizer.step()

        print('[epoch %d] | loss_recon: %.3f, loss_commit: %.3f, perplexity %.3f'
              %(epoch, loss_recon.item(), loss_commit.item(), perplexity.item()))

        # tb_writer.add_scalar('loss_recon', loss_recon.item(), epoch)
        # tb_writer.add_scalar('loss_commit', loss_commit.item(), epoch)
        # tb_writer.add_scalar('perplexity', perplexity.item(), epoch)

    torch.save({'model_state_dict': model.state_dict()}, 'downstream/src/rqvae/%s.pth'%model_name)
else:
    check_point = torch.load('downstream/src/rqvae/%s.pth'%model_name, map_location='cuda')
    model.load_state_dict(check_point['model_state_dict'], strict=True)