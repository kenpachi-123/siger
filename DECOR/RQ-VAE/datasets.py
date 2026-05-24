import os

import numpy as np
import torch
from torch.utils.data import Dataset


class EmbDataset(Dataset):
    def __init__(self, data_path, embedding_dim=256, pca_dim=None):
        self.data_path = data_path
        embeddings = np.load(data_path).astype(np.float32)
        if embeddings.ndim != 2:
            raise ValueError(f"Expected 2D embeddings, got shape {embeddings.shape} from {data_path}")

        if embedding_dim is not None:
            if embedding_dim <= 0 or embedding_dim > embeddings.shape[1]:
                raise ValueError(
                    f"embedding_dim must be in [1, {embeddings.shape[1]}], got {embedding_dim}"
                )
            embeddings = embeddings[:, :embedding_dim]

        if pca_dim is not None:
            if pca_dim <= 0 or pca_dim > embeddings.shape[1]:
                raise ValueError(
                    f"pca_dim must be in [1, {embeddings.shape[1]}], got {pca_dim}"
                )
            embeddings = self._apply_pca(embeddings, pca_dim)

        self.embeddings = torch.from_numpy(embeddings).float()
        self.dim = self.embeddings.shape[-1]
        print(
            f"Loaded {len(self.embeddings)} embeddings with dimension {self.dim} from {data_path} "
            f"(embedding_dim={embedding_dim}, pca_dim={pca_dim})"
        )

    @staticmethod
    def _apply_pca(embeddings, pca_dim):
        centered = embeddings - embeddings.mean(axis=0, keepdims=True)
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        components = vt[:pca_dim]
        reduced = centered @ components.T
        return reduced.astype(np.float32, copy=False)

    def __getitem__(self, index):
        return self.embeddings[index], index

    def __len__(self):
        return len(self.embeddings)
