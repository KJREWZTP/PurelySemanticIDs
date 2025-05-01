import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import json
import os
from sklearn.cluster import KMeans
import itertools
from tqdm import tqdm
from argparse import ArgumentParser

import json
import os
import numpy as np
import torch
from sklearn.cluster import KMeans

def sanitize_semantic_ids(id_dict, max_segments=2, inplace=True):
    """
    Check and fix semantic IDs to ensure they have at most `max_segments` components.

    Args:
        id_dict (dict): Dictionary of IDs, e.g., {item_id: 'x,y,z'}
        max_segments (int): Maximum allowed number of segments (default: 2)
        inplace (bool): If True, modify in-place. If False, return a new dict.

    Returns:
        dict: Sanitized ID dict (only if inplace=False)
    """
    if not inplace:
        id_dict = dict(id_dict)  # shallow copy

    corrected = 0
    for k, v in id_dict.items():
        segments = v.split(',')
        
        if len(segments) > max_segments:
            id_dict[k] = ','.join(segments[:max_segments])
            corrected += 1

    print(f"Semantic ID check complete: {corrected} corrected out of {len(id_dict)}")
    
    if not inplace:
        return id_dict


class HCIndexer:
    def __init__(self, cluster_nums=None, max_depth=None, max_docs=100, device='cpu', conflict_threshold=5):
        self.cluster_nums = cluster_nums if cluster_nums is not None else [10] * 10
        self.max_depth = max_depth
        self.max_docs = max_docs
        self.device = torch.device(device)
        self.final_data = {}
        self.id_cache = set()
        self.candidate_map = {} 
        self.conflict_threshold = conflict_threshold
        self.conflict_num = 0
        self.original_ids_w_conflict = set()
    
    def clear_data(self):
        self.final_data = {}
        self.id_cache = set()
        self.candidate_map = {}
        self.conflict_num = 0
        self.original_ids_w_conflict = set()

    def count_conflict_id_nums(self):
        print("conflict id nums: ", self.conflict_num)
        return self.conflict_num

    def cluster_embeddings(self, embeddings, num_clusters):
        km = KMeans(n_clusters=num_clusters, random_state=0)
        cluster_labels = km.fit_predict(embeddings)
        return km, cluster_labels
    
    
    def generate_ids_with_cluster_nums_and_max_depth(self, embeddings, original_indices=None, depth=0, semantic_id_prefix=''):
        """ Recursively generate semantic IDs, ensuring they are at most 2 segments (x,y). """

        if original_indices is None:
            original_indices = list(range(len(embeddings)))

        if self.max_depth is not None and depth >= self.max_depth:
            raise ValueError(f"Exceeded max depth {self.max_depth} at recursion depth {depth}")

        num_clusters = self.cluster_nums[min(depth, len(self.cluster_nums) - 1)]
        actual_num_clusters = min(num_clusters, len(embeddings))  # Prevent ValueError

        if actual_num_clusters <= 1:
            # Too few points to cluster
            for idx, original_idx in enumerate(original_indices):
                self.final_data[str(original_idx + 1)] = semantic_id_prefix
            self.conflict_num += len(original_indices)
            return

        embeddings_np = np.array(embeddings)
        km, cluster_labels = self.cluster_embeddings(embeddings_np, actual_num_clusters)

        clusters = {i: [] for i in range(actual_num_clusters)}
        cluster_indices = {i: [] for i in range(actual_num_clusters)}
        for idx, label in enumerate(cluster_labels):
            clusters[label].append(embeddings[idx])
            cluster_indices[label].append(original_indices[idx])

        for label, cluster_i in clusters.items():
            new_prefix = semantic_id_prefix + (',' if semantic_id_prefix else '') + str(label)
            next_depth = depth + 1

            # Stop recursion: already reached target depth (x,y)
            if next_depth >= self.max_depth or len(cluster_i) <= self.max_docs:
                for original_idx in cluster_indices[label]:
                    self.final_data[str(original_idx + 1)] = new_prefix
                self.conflict_num += len(cluster_indices[label])
                continue

            # Otherwise, safe to go deeper
            self.generate_ids_with_cluster_nums_and_max_depth(
                cluster_i, cluster_indices[label], next_depth, new_prefix)

    def find_closest_clusters(self, embedding, clusters, topk=5):
        """
        Compute the distance between the given embedding and the center of each cluster
        (calculated as the mean of vectors in the cluster). Returns the topk (cluster_id, distance)
        pairs with the smallest distances.
        """
        distances = [
            (l, np.linalg.norm(embedding - np.mean(clusters[l], axis=0)))
            for l in clusters if len(clusters[l]) > 0
        ]
        return sorted(distances, key=lambda x: x[1])[:topk]

    def generate_id_candidates_without_conflict(
        self,
        embeddings,
        original_indices=None,
        depth: int = 0,
        semantic_id_prefix: str = ''
    ):
        """
        Assign a unique two-segment semantic ID “cluster,neighbor” to every
        embedding.  The first segment is the k-means cluster label; the second
        segment is chosen, in order, from the nearest cluster centroids to the
        document’s own centroid.  A global `self.id_cache` prevents collisions.

        After the call:
            • self.final_data[str(idx)] = "<int>,<int>"
            • len(self.final_data) == len(embeddings)  (sanity-check enforced)
        """
        if semantic_id_prefix:
            raise ValueError("This implementation always starts with an empty "
                            "semantic_id_prefix so that IDs have exactly two "
                            "segments.")

        if original_indices is None:
            original_indices = list(range(len(embeddings)))

    
        k = min(self.cluster_nums[0], len(embeddings))
        if k < 2:
            raise ValueError("Need at least two clusters to build two-segment IDs")

        km, labels = self.cluster_embeddings(np.asarray(embeddings), k)
        centers = km.cluster_centers_                    # (k, dim)

        # Build mapping: cluster_id → list[(pos_in_embeddings, original_idx)]
        clusters = {i: [] for i in range(k)}
        for pos, lab in enumerate(labels):
            clusters[lab].append((pos, original_indices[pos]))

        for clu, members in clusters.items():
            # Pre-compute neighbour clusters sorted by centroid distance
            dists = np.linalg.norm(centers - centers[clu], axis=1)
            neighbour_order = np.argsort(dists).tolist()        # closest first

            for pos, orig_idx in members:
                key = str(orig_idx)

                # Try closest centroids first
                for nb in neighbour_order:
                    cand = f"{clu},{nb}"
                    if cand not in self.id_cache:
                        self.final_data[str(int(key)+1)] = cand
                        self.id_cache.add(cand)
                        break
                else:
                    # All “clu,nb” taken – fall back to linear probing
                    probe = 0
                    while True:
                        cand = f"{clu},{probe}"
                        if cand not in self.id_cache:
                            self.final_data[str(int(key)+1)] = cand
                            self.id_cache.add(cand)
                            self.conflict_num += 1          # count real conflicts
                            break
                        probe += 1


    def _to_two_segments(self, prefix: str, suffix: str = "0") -> str:
        segs = prefix.split(',')
        if len(segs) == 1:
            return f"{segs[0]},{suffix}"
        return f"{segs[0]},{segs[1]}"        
    

    def recursively_generate_ids_wo_conflict(
            self,
            doc_embeddings,
            original_indices=None,
            depth: int = 0,
            semantic_id_prefix: str = ''
    ):
        if original_indices is None:
            original_indices = list(range(len(doc_embeddings)))

        if semantic_id_prefix.count(',') == 1:
            for local_idx, orig_idx in enumerate(original_indices):
                key  = str(orig_idx)
                cand = self._to_two_segments(semantic_id_prefix, str(local_idx))
                if cand not in self.id_cache:
                    self.final_data[str(int(key)+1)] = cand
                    self.id_cache.add(cand)
                else:
                    self.resolve_conflict_recursively(
                        original_idx=orig_idx,
                        base_prefix=semantic_id_prefix,
                        probe_start=local_idx + 1,
                        visited=set()
                    )
            return

        num_clusters = self.cluster_nums[min(depth, len(self.cluster_nums) - 1)]
        k = min(num_clusters, len(doc_embeddings))

        if k < 2:
            for local_idx, orig_idx in enumerate(original_indices):
                key  = str(orig_idx)
                cand = self._to_two_segments(semantic_id_prefix or str(local_idx))
                probe = 0
                while cand in self.id_cache:
                    probe += 1
                    cand = self._to_two_segments(semantic_id_prefix or str(local_idx), str(probe))
                self.final_data[str(int(key)+1)] = cand
                self.id_cache.add(cand)
            return

        km, labels = self.cluster_embeddings(np.asarray(doc_embeddings), k)

        clusters, cluster_indices = {}, {}
        for lab in range(k):
            clusters[lab] = []
            cluster_indices[lab] = []
        for i, lab in enumerate(labels):
            clusters[lab].append(doc_embeddings[i])
            cluster_indices[lab].append(original_indices[i])

        for lab, emb_list in clusters.items():
            idx_list   = cluster_indices[lab]
            first_seg  = f"{lab}"

            if len(emb_list) > self.max_docs:
                self.recursively_generate_ids_wo_conflict(
                    emb_list, idx_list, depth + 1, first_seg
                )
            else:
                for local_idx, orig_idx in enumerate(idx_list):
                    key  = str(orig_idx)
                    cand = self._to_two_segments(first_seg, str(local_idx))
                    if cand not in self.id_cache:
                        self.final_data[str(int(key)+1)] = cand
                        self.id_cache.add(cand)
                    else:
                        self.resolve_conflict_recursively(
                            original_idx=orig_idx,
                            base_prefix=first_seg,
                            probe_start=local_idx + 1,
                            visited=set()
                        )

    def resolve_conflict_recursively(
            self,
            original_idx,
            base_prefix: str,
            probe_start: int,
            visited: set
    ) -> bool:
        if base_prefix in visited:
            return False
        visited.add(base_prefix)

        probe = probe_start
        while True:
            cand = self._to_two_segments(base_prefix, str(probe))
            if cand not in self.id_cache:
                key = str(original_idx)
                self.final_data[str(int(key)+1)] = cand
                self.id_cache.add(cand)
                if probe != 0:
                    self.conflict_num += 1
                return True
            probe += 1
    
    
    def save_results(self, save_path):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        sorted_final_data = {k: self.final_data[k] for k in sorted(self.final_data, key=lambda x: int(x))}

        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(sorted_final_data, f, indent=4)
        print(f'Saved results to {save_path}')

if __name__ == "__main__": 

    parser = ArgumentParser()
    parser.add_argument('--embedding_dir', type=str, required=True, help='Path to the embedding file')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save clustered results')
    parser.add_argument('--cluster_nums', type=str, default=None, help='Comma-separated list of cluster numbers per layer')
    parser.add_argument('--base_model', type=str, default=None, help='Base model name')
    parser.add_argument('--semantic_id', type=str, default=None, help='Semantic ID name')
    parser.add_argument('--max_depth', type=int, default=None, help='Maximum clustering depth')
    parser.add_argument('--max_docs', type=int, default=256, help='Maximum number of documents per cluster')
    parser.add_argument('--conflict_threshold', type=int, default=5, help='Threshold for checking alternative clusters in recursive resolution')
    args = parser.parse_args()

    cluster_nums = list(map(int, args.cluster_nums.split(','))) if args.cluster_nums else None

    embeddings = torch.load(args.embedding_dir, map_location='cpu').numpy()

    indexer = HCIndexer(cluster_nums=cluster_nums, max_depth=args.max_depth, max_docs=args.max_docs, conflict_threshold=args.conflict_threshold)

    print("Start Generating ...")
    # Generate IDs using all three methods

    output_path = args.output_path + f"{args.semantic_id}-code-{args.base_model}.json"
    indexer.generate_ids_with_cluster_nums_and_max_depth(embeddings)
    indexer.count_conflict_id_nums()
    indexer.save_results(output_path)
    indexer.clear_data()

    output_path = args.output_path + f"{args.semantic_id}-noconflict-candidate-code-{args.base_model}.json"
    indexer.generate_id_candidates_without_conflict(embeddings)
    indexer.count_conflict_id_nums()
    indexer.save_results(output_path)
    indexer.clear_data()

    output_path = args.output_path + f"{args.semantic_id}-noconflict-recursive-code-{args.base_model}.json"
    indexer.recursively_generate_ids_wo_conflict(embeddings)
    indexer.count_conflict_id_nums()
    indexer.save_results(output_path)
    indexer.clear_data()

