import numpy as np
import torch
import torch.utils.data as data


class EmbDataset(data.Dataset):

    def __init__(self,data_path):

        self.data_path = data_path
        self.embeddings = torch.from_numpy(np.load(data_path)[:,:256]).float()
        self.dim = self.embeddings.shape[-1]
        print(f"Loaded {len(self.embeddings)} embeddings with dimension {self.dim} from {data_path}")

    def __getitem__(self, index):
        return self.embeddings[index], index

    def __len__(self):
        return len(self.embeddings)
