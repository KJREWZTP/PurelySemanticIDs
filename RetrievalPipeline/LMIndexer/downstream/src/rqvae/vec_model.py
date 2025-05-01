from __future__ import print_function

import os
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

from IPython import embed
from tqdm import tqdm
import itertools

import line_profiler


def straight_through_estimator(x, c):
    # this method replaces x by c without stopping the gradient flowing through x
    return x + (c-x).detach()


def compute_perplexity(encodings):

    avg_probs = torch.mean(encodings, dim=0)
    perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
    return perplexity


class VectorQuantizerEMA(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, decay, epsilon=1e-5):
        super(VectorQuantizerEMA, self).__init__()

        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings

        # self._embedding.weight acts as a codebook
        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.requires_grad = False
        self._embedding.weight.data.normal_()

        self.register_buffer('_ema_cluster_size', torch.zeros(num_embeddings))
        self._ema_w = nn.Parameter(torch.Tensor(num_embeddings, self._embedding_dim), requires_grad=False)
        self._ema_w.data.normal_()

        self._decay = decay
        self._epsilon = epsilon

    def forward(self, inputs):
        # convert inputs from BCHW -> BHWC
        inputs = inputs.detach()
        # inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape

        # Flatten input
        flat_input = inputs.view(-1, self._embedding_dim)

        codebook = self._embedding.weight.data
        codebook_ema = self._ema_w.data

        # Calculate distances
        distances = (torch.sum(flat_input ** 2, dim=1, keepdim=True)
                     + torch.sum(codebook ** 2, dim=1)
                     - 2 * torch.matmul(flat_input, codebook.t()))

        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)

        encoding_indices_squeezed = encoding_indices.squeeze(1)

        quantized = codebook[encoding_indices_squeezed] 
        quantized = quantized.view(input_shape)


        # Use EMA to update the embedding vectors
        if self.training:
            # EMA update for codebook
            dw = torch.matmul(encodings.t(), flat_input)
            self._ema_w.data = codebook_ema * self._decay + (1 - self._decay) * dw

            self._ema_cluster_size = self._ema_cluster_size * self._decay + \
                                     (1 - self._decay) * torch.sum(encodings, 0)

            # Codebook update with Laplace smoothing of the cluster size
            n = torch.sum(self._ema_cluster_size.data)
            self._ema_cluster_size = (
                    (self._ema_cluster_size + self._epsilon)
                    / (n + self._num_embeddings * self._epsilon) * n)

            self._embedding.weight.data = self._ema_w.data / self._ema_cluster_size.unsqueeze(1)

        # convert quantized from BHWC -> BCHW
        # return quantized.permute(0, 3, 1, 2).contiguous(), encodings.detach()
        return quantized, encodings.detach()
    
    @line_profiler.profile
    def encode_to_id_return_topk(self, x, topk: int):
        # convert inputs from BCHW -> BHWC
        x = x.detach()
        # inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = x.shape

        # Flatten input
        flat_input = x.view(-1, self._embedding_dim)

        codebook = self._embedding.weight.data
        codebook_ema = self._ema_w.data

        # Calculate distances
        dist = (torch.sum(flat_input ** 2, dim=1, keepdim=True)
                     + torch.sum(codebook ** 2, dim=1)
                     - 2 * torch.matmul(flat_input, codebook.t()))
        
        _, embed_ind = (-dist).topk(topk, dim=1)

        residuals = []
        embed_ind_topks = []
        
        quantized = None

        for i in range(topk):
            embed_ind_topk = embed_ind[:, i]
            embed_ind_topk = embed_ind_topk.view(*x.shape[:-1])
            embed_ind_topks.append(embed_ind_topk)
           
            quantize = codebook[embed_ind_topk]
            if i == 0:
                quantized = quantize
            residual = x - quantize
            residuals.append(residual)

        residuals = torch.stack(residuals, dim=0)
        embed_ind_topks = torch.stack(embed_ind_topks, dim=0)
        return residuals, embed_ind_topks, quantized
    

class Encoder(nn.Module):
    def __init__(self, in_channels, num_hiddens, code_hiddens):
        super(Encoder, self).__init__()

        assert len(num_hiddens) == 3

        self._linear1 = nn.Linear(in_features=in_channels, out_features=num_hiddens[0])
        self._linear2 = nn.Linear(in_features=num_hiddens[0], out_features=num_hiddens[1])
        self._linear3 = nn.Linear(in_features=num_hiddens[1], out_features=num_hiddens[2])
        self._linear4 = nn.Linear(in_features=num_hiddens[2], out_features=code_hiddens)

    def forward(self, inputs):
        x = self._linear1(inputs)
        x = F.relu(x)

        x = self._linear2(x)
        x = F.relu(x)

        x = self._linear3(x)
        x = F.relu(x)

        x = self._linear4(x)
        return x


class Decoder(nn.Module):
    def __init__(self, in_channels, num_hiddens, code_hiddens):
        super(Decoder, self).__init__()

        assert len(num_hiddens) == 3

        self._linear1 = nn.Linear(in_features=code_hiddens, out_features=num_hiddens[0])
        self._linear2 = nn.Linear(in_features=num_hiddens[0], out_features=num_hiddens[1])
        self._linear3 = nn.Linear(in_features=num_hiddens[1], out_features=num_hiddens[2])
        self._linear4 = nn.Linear(in_features=num_hiddens[2], out_features=in_channels)

    def forward(self, inputs):
        x = self._linear1(inputs)
        x = F.relu(x)

        x = self._linear2(x)
        x = F.relu(x)

        x = self._linear3(x)
        x = F.relu(x)

        x = self._linear4(x)
        return x


class RQVAE(nn.Module):
    def __init__(self, input_dim, num_hiddens, num_embeddings, embedding_dim, commitment_cost, decay=0., depth=1):
        super(RQVAE, self).__init__()

        self._encoder = Encoder(input_dim, num_hiddens, embedding_dim)  # in_channels, num_hiddens, code_hiddens

        assert decay > 0.0
        self._quantizer = VectorQuantizerEMA(num_embeddings, embedding_dim, decay)

        self._decoder = Decoder(input_dim, num_hiddens, embedding_dim)

        self._commitment_cost = commitment_cost

        assert depth > 0
        self._depth = depth
        self.num_embeddings = num_embeddings

    def forward(self, x):
        z = self._encoder(x)

        '''
        start - residual quantization
        '''
        z_detached = z.detach().clone()

        _residue = z_detached
  
        for d in range(self._depth):
            _quantized, _code = self._quantizer(_residue)
            _code_id = _code.argmax(1).unsqueeze(-1)

            if d == 0:
                all_codes = _code_id
            else:
                all_codes = torch.cat((all_codes, _code_id), dim=1)

            if d == 0:
                z_hat = _quantized
            else:
                z_hat += _quantized

            _residue = z_detached - z_hat
       
        '''
        end - residual quantization
        '''

        z_hat_flowing = straight_through_estimator(z, z_hat)
        x_recon = self._decoder(z_hat_flowing)

        return x_recon, all_codes
    
    @line_profiler.profile
    def generate_ids_wo_conflict_by_candidates(self, x: torch.Tensor, device: torch.device, score_func="sum_norm"):
        z = self._encoder(x)
        '''
        start - residual quantization
        '''
        original_x = z.detach().clone()
        x = z.detach().clone()
        z_detached = z.detach().clone()
        _residual = z_detached

        self.id_cache = set()
        self.id_confirmed_index = torch.zeros(x.shape[0], dtype=torch.bool, device=device)

        embed_index_matrix = []
        all_layer_residuals, all_layer_embed_indices = [], []
        
        # topks = [1] * (self._depth - 1) + [min(3, self.num_embeddings)]
        # topks_backup = [10] * (self._depth - 1) + [min(10, self.num_embeddings)]

        topks = [1] * (self._depth - 1) + [min(10, self.num_embeddings)]
        topks_backup = [5] * (self._depth - 1) + [min(20, self.num_embeddings)]

        for d in range(self._depth):
            residuals, embed_indices, _quantized = self._quantizer.encode_to_id_return_topk(_residual, topk=topks_backup[d])
            if d == 0:
                z_hat = _quantized
            else:
                z_hat += _quantized

            _residual = z_detached - z_hat
            embed_index = embed_indices[0]
            embed_index_matrix.append(embed_index)
            all_layer_residuals.append(residuals)
            all_layer_embed_indices.append(embed_indices)

        # filter unique ID and corresponding indices
        embed_matrix = torch.stack(embed_index_matrix, dim=1)
        
        unique_ids, inverse_mapping = torch.unique(embed_matrix, dim=0, return_inverse=True)

        for unique_idx in range(len(unique_ids)):
            first_match_idx = (inverse_mapping == unique_idx).nonzero(as_tuple=True)[0][0]
            self.id_confirmed_index[first_match_idx] = True
            self.id_cache.add(tuple(unique_ids[unique_idx].cpu().numpy()))

        print(f"Unique IDs generated: {len(unique_ids)}")

        # Conflict resolution
        embed_matrix, self.id_confirmed_index = self._resolve_conflicts(
            embed_matrix,
            all_layer_residuals,
            all_layer_embed_indices,
            self.id_confirmed_index,
            topks,
            self._depth,
            device,
            score_func
        )

        number = torch.sum(self.id_confirmed_index, dtype=torch.int32)
        print("confirmed ones:", number)

        left_ones = x.shape[0]-number.item()
        print("left ones:", left_ones)

        # Conflict resolution
        embed_matrix, self.id_confirmed_index = self._resolve_conflicts(
            embed_matrix,
            all_layer_residuals,
            all_layer_embed_indices,
            self.id_confirmed_index,
            topks_backup,
            self._depth,
            device,
            score_func
        )

        number = torch.sum(self.id_confirmed_index, dtype=torch.int32)
        print("confirmed ones:", number)

        left_ones = x.shape[0]-number.item()
        print("left ones:", left_ones)
        
        new_recon = self.compute_recon(embed_matrix, z)
        return new_recon, embed_matrix
    
    # Faster version: explicitly switch branches based on depth
    @line_profiler.profile
    def _resolve_conflicts(
        self,
        embed_matrix,
        residuals_by_layer,
        indices_by_layer,
        id_confirmed_index,
        topks,
        n_digit,
        device,
        score_func
    ):
        if score_func == "sum_norm":
            remaining_indices = torch.where(~id_confirmed_index)[0]
            print(f"Remaining unresolved IDs: {len(remaining_indices)}")
            count = 0

            for idx in tqdm(remaining_indices, desc="Resolving conflicts"):
                layer_residuals = [residuals[:, idx].norm(p=2, dim=1).detach().cpu().numpy() for residuals in residuals_by_layer]
                layer_indices = [indices[:, idx].detach().cpu().numpy() for indices in indices_by_layer]

                id_candidates = {}
                if n_digit == 3:
                    for idx1, idx2, idx3 in itertools.product(range(topks[0]), range(topks[1]), range(topks[2])):
                        score = sum(layer_residuals[layer][idx] for layer, idx in enumerate((idx1, idx2, idx3)))
                        id_candidates[(layer_indices[0][idx1], layer_indices[1][idx2], layer_indices[2][idx3])] = score
                elif n_digit == 2:
                    for idx1, idx2 in itertools.product(range(topks[0]), range(topks[1])):
                        score = layer_residuals[0][idx1] + layer_residuals[1][idx2]
                        id_candidates[(layer_indices[0][idx1], layer_indices[1][idx2])] = score

                for candidate, _ in sorted(id_candidates.items(), key=lambda x: x[1]):
                    if candidate not in self.id_cache:
                        self.id_cache.add(candidate)
                        id_confirmed_index[idx] = True
                        embed_matrix[idx] = torch.tensor(candidate, device=device)
                        count += 1
                        break

            print(f"Resolved conflicts: {count}")
            return embed_matrix, id_confirmed_index

        elif score_func == "random":
            remaining_indices = torch.where(~id_confirmed_index)[0]
            print(f"Remaining unresolved IDs: {len(remaining_indices)}")
            count = 0

            for idx in tqdm(remaining_indices, desc="Resolving conflicts"):
                layer_residuals = [residuals[:, idx].norm(p=2, dim=1).detach().cpu().numpy() for residuals in residuals_by_layer]
                layer_indices = [indices[:, idx].detach().cpu().numpy() for indices in indices_by_layer]

                id_candidates = {}
                if n_digit == 3:
                    for idx1, idx2, idx3 in itertools.product(range(topks[0]), range(topks[1]), range(topks[2])):
                        id_candidates[(layer_indices[0][idx1], layer_indices[1][idx2], layer_indices[2][idx3])] = 1  # score is unused
                elif n_digit == 2:
                    for idx1, idx2 in itertools.product(range(topks[0]), range(topks[1])):
                        id_candidates[(layer_indices[0][idx1], layer_indices[1][idx2])] = 1

                candidates = list(id_candidates.keys())
                random.shuffle(candidates)


                for candidate in candidates:
                    if candidate not in self.id_cache:
                        self.id_cache.add(candidate)
                        id_confirmed_index[idx] = True
                        embed_matrix[idx] = torch.tensor(candidate, device=device)
                        count += 1
                        break
            print(f"Resolved conflicts: {count}")
            return embed_matrix, id_confirmed_index
        elif score_func == "combined_order":
            remaining_indices = torch.where(~id_confirmed_index)[0]
            print(f"Remaining unresolved IDs: {len(remaining_indices)}")
            count = 0

            for idx in tqdm(remaining_indices, desc="Resolving conflicts"):
                layer_residuals = [residuals[:, idx].norm(p=2, dim=1).detach().cpu().numpy() for residuals in residuals_by_layer]
                layer_indices = [indices[:, idx].detach().cpu().numpy() for indices in indices_by_layer]

                # generate candidates in lexicographic order
                if n_digit == 3:
                    candidate_iter = itertools.product(
                        range(topks[0]),
                        range(topks[1]),
                        range(topks[2])
                    )
                elif n_digit == 2:
                    candidate_iter = itertools.product(
                        range(topks[0]),
                        range(topks[1])
                    )
                else:
                    raise ValueError("Only 2 or 3 levels are supported.")

                for index_tuple in candidate_iter:
                    candidate = tuple([layer_indices[i][index_tuple[i]] for i in range(len(topks))])

                    if candidate not in self.id_cache:
                        self.id_cache.add(candidate)
                        embed_matrix[idx] = torch.tensor(candidate, device=device)
                        id_confirmed_index[idx] = True
                        count += 1
                        break
            return embed_matrix, id_confirmed_index
        else:
            return embed_matrix, id_confirmed_index
    
    def reconstruct_z_hat_from_embed_matrix(self, embed_matrix):
        """
        Reconstruct z_hat using manually provided embed_matrix (code IDs per layer).

        Args:
            embed_matrix (torch.LongTensor): shape [B, D]
            codebook (torch.Tensor): shape [D * num_embeddings, dim]
            num_embeddings (int): number of embeddings per layer
            depth (int): number of residual quantization levels

        Returns:
            Tensor: z_hat of shape [B, dim]
        """
        device = embed_matrix.device
        batch_size = embed_matrix.size(0)
        codebook = self._quantizer._embedding.weight.data
        dim = codebook.size(1)

        z_hat = torch.zeros(batch_size, dim, device=device)

        for d in range(self._depth):
            indices = embed_matrix[:, d]
            quantized_d = codebook[indices]
            z_hat += quantized_d

        return z_hat

    def compute_recon(self, embed_matrix, original_z):
        z_hat = self.reconstruct_z_hat_from_embed_matrix(embed_matrix)
        z_hat_flowing = straight_through_estimator(original_z, z_hat)
        x_recon = self._decoder(z_hat_flowing)
        return x_recon
    

    # # Generalized version to resolve conflicts
    # @line_profiler.profile
    # def _resolve_conflicts(
    #     self,
    #     embed_matrix,
    #     residuals_by_layer,
    #     indices_by_layer,
    #     id_confirmed_index,
    #     topks,
    #     n_digit,
    #     device
    # ):
    #     remaining_indices = torch.where(~id_confirmed_index)[0]
    #     print(f"Remaining unresolved IDs: {len(remaining_indices)}")
    #     count = 0

    #     for idx in tqdm(remaining_indices, desc="Resolving conflicts"):
    #         layer_residuals = [residuals[:, idx].norm(p=2, dim=1).detach().cpu().numpy() for residuals in residuals_by_layer]
    #         layer_indices = [indices[:, idx].detach().cpu().numpy() for indices in indices_by_layer]

    #         id_candidates = {}
    #         for combination in itertools.product(*(range(topks[layer]) for layer in range(n_digit))):
    #             score = sum(layer_residuals[layer][combination[layer]] for layer in range(n_digit))
    #             candidate = tuple(layer_indices[layer][combination[layer]] for layer in range(n_digit))
    #             id_candidates[candidate] = score

    #         for candidate, _ in sorted(id_candidates.items(), key=lambda x: x[1]):
    #             if candidate not in self.id_cache:
    #                 self.id_cache.add(candidate)
    #                 id_confirmed_index[idx] = True
    #                 embed_matrix[idx] = torch.tensor(candidate, device=device)
    #                 count += 1
    #                 break

    #     print(f"Resolved conflicts: {count}")
    #     return embed_matrix, id_confirmed_index



    @line_profiler.profile
    def recursively_generate_ids_wo_conflict(self, x: torch.Tensor, device: torch.device) -> torch.Tensor:
        """
        This method matches your original 'recursive' approach but uses the new 
        quantization-layer interface (encode_to_id_return_topk).
        """

        z = self._encoder(x)
        x = z.detach().clone()
        original_x = x.clone()
        z_detached = z.detach().clone()
        _residual = z_detached

        self.id_cache = set()
        self.id_confirmed_index = torch.zeros(x.shape[0], dtype=torch.bool, device=device)

        embed_indices = []

        for d in range(self._depth):
            top1_residuals, top1_indices, _quantized = self._quantizer.encode_to_id_return_topk(_residual, topk=1)
            
            if d == 0:
                z_hat = _quantized
            else:
                z_hat += _quantized

            _residual = z_detached - z_hat
            best_residual = top1_residuals[0]
            best_id = top1_indices[0]
            embed_indices.append(best_id)


        embed_matrix = torch.stack(embed_indices, dim=1)
        unique_ids, inverse_matrix = torch.unique(embed_matrix, dim=0, return_inverse=True)

        topks = [1] * (self._depth - 1) + [min(10, self.num_embeddings)]
        topks_backup = [5] * (self._depth - 1) + [min(20, self.num_embeddings)]

        for i in range(len(unique_ids)):
            first_match_idx = (inverse_matrix == i).nonzero(as_tuple=True)[0][0]
            self.id_confirmed_index[first_match_idx] = True
            self.id_cache.add(tuple(unique_ids[i].detach().cpu().numpy()))
        
        print(f"unique number: {len(unique_ids)}")

        x = original_x
        embed_matrix = self._recursive_reducing_conflict(
            x,
            embed_matrix,
            topks,
            device
        )

        number = torch.sum(self.id_confirmed_index, dtype=torch.int32)
        print("confirmed ones:", number)

        left_ones = x.shape[0]-number.item()
        print("left ones:", left_ones)

        x = original_x
        embed_matrix = self._recursive_reducing_conflict(
            x,
            embed_matrix,
            topks_backup,
            device
        )

        number = torch.sum(self.id_confirmed_index, dtype=torch.int32)
        print("confirmed ones:", number)

        left_ones = x.shape[0]-number.item()
        print("left ones:", left_ones)

        new_recon = self.compute_recon(embed_matrix, z)
        return new_recon, embed_matrix


    @line_profiler.profile
    def _recursive_reducing_conflict(
        self,
        x,
        embed_matrix,
        topks,
        device
    ) -> torch.Tensor:
        """
        For any unconfirmed samples, we iteratively try to find a new ID tuple via a
        recursive search that uses top-k candidates at each layer.
        - We do multiple passes (while-loop) until all are confirmed or no progress.
        - Each sample's ID search is done by _search_new_id_for_sample (recursive).
        """
        original_x = x.clone()
        batch_size = x.shape[0]
        n_layers = self._depth

        while not torch.all(self.id_confirmed_index):
            zero_indices = torch.where(~self.id_confirmed_index)[0]
            if len(zero_indices) == 0:
                break

            print(f"Remaining unconfirmed samples: {len(zero_indices)}")
            count = 0

            # Try to find unique IDs for each unconfirmed sample
            for i_sample in zero_indices:
                if self.id_confirmed_index[i_sample]:
                    continue  # might have been resolved mid-loop

                x_i = original_x[i_sample : i_sample + 1]  # shape: (1, dim, ...)
                # Attempt to find a new unique ID via recursion
                found_new_id, candidate_id = self._search_new_id_for_sample(
                    sample_idx=i_sample,
                    sample_x=x_i,
                    layer_idx=0,
                    current_ids=[],
                    device=device,
                    topks=topks
                )

                if found_new_id:
                    self.id_confirmed_index[i_sample] = True
                    embed_matrix[i_sample] = torch.tensor(candidate_id, dtype=torch.long, device=device)
                    self.id_cache.add(candidate_id)
                    count += 1

            print(f"Newly confirmed this pass: {count}")
            total_confirmed = torch.sum(self.id_confirmed_index).item()
            print(f"Total confirmed so far: {total_confirmed}")

            # If we can't confirm any more IDs, break to avoid infinite loop
            if count == 0:
                print("No further unique IDs found; stopping.")
                break

        return embed_matrix


    @line_profiler.profile
    def _search_new_id_for_sample(
        self,
        sample_idx: int,
        sample_x: torch.Tensor,
        layer_idx: int,
        current_ids: list,
        device: torch.device,
        topks: list
    ) -> (bool, tuple):
        """
        Recursively explore top-k candidates at each layer until we either:
        1) reach the end (layer_idx == #layers) => check if candidate_id is in id_cache
        2) or fail to find any unique ID at this layer.

        Returns:
        (found_new_id, candidate_id_or_None)
            found_new_id: bool
            candidate_id_or_None: If found_new_id == True, returns the new unique tuple.
        """
        n_layers = self._depth

        # Base Case: All layers assigned
        if layer_idx == n_layers:
            candidate_tuple = tuple(current_ids)
            if candidate_tuple not in self.id_cache:
                return True, candidate_tuple
            return False, None

        # Otherwise, we proceed with top-k at the current layer
        topk = topks[layer_idx]
        residuals, indices, _ = self._quantizer.encode_to_id_return_topk(sample_x, topk=topk)

        # Explore each top-k candidate in turn
        for k_idx in range(topk):
            next_res = residuals[k_idx]
            next_id = int(indices[k_idx].item())

            found_new_id, cand = self._search_new_id_for_sample(
                sample_idx=sample_idx,
                sample_x=next_res,
                layer_idx=layer_idx + 1,
                current_ids=current_ids + [next_id],
                device=device,
                topks=topks
            )
            if found_new_id:
                return True, cand

        return False, None


    def compute_loss(self, x):
        z = self._encoder(x)

        '''
        start - residual quantization
        '''
        z_detached = z.detach().clone()

        _residue = z_detached
        for d in range(self._depth):
            _quantized, _encodings = self._quantizer(_residue)

            if d == 0:
                z_hat = _quantized
                _perplexity = compute_perplexity(_encodings)
                _loss_commit = F.mse_loss(z_hat.clone(), z)
            else:
                z_hat += _quantized
                _perplexity += compute_perplexity(_encodings)
                _loss_commit += F.mse_loss(z_hat.clone(), z)    # TODO: figure out why clone is required

            _residue = z_detached - z_hat

        perplexity = _perplexity / self._depth
        '''
        end - residual quantization
        '''

        z_hat_flowing = straight_through_estimator(z, z_hat)
        x_recon = self._decoder(z_hat_flowing)

        loss_commit = self._commitment_cost * _loss_commit

        return loss_commit, x_recon, perplexity
