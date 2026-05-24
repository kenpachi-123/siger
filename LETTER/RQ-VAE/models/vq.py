import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import kmeans, sinkhorn_algorithm
import random
#import wandb


class VectorQuantizer(nn.Module):

    def __init__(self, n_e, e_dim, mu = 0.25,
                 beta = 1, kmeans_init = False, kmeans_iters = 10,
                 sk_epsilon=0.01, sk_iters=100):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.mu = mu
        self.kmeans_init = kmeans_init
        self.kmeans_iters = kmeans_iters
        self.sk_epsilon = sk_epsilon
        self.sk_iters = sk_iters

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        if not kmeans_init:
            self.initted = True
            self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)
        else:
            self.initted = False
            self.embedding.weight.data.zero_()

    def get_codebook(self):
        return self.embedding.weight

    def get_codebook_entry(self, indices, shape=None):
        # get quantized latent vectors
        z_q = self.embedding(indices)
        if shape is not None:
            z_q = z_q.view(shape)

        return z_q

    def init_emb(self, data):

        # centers = kmeans(
        #     data,
        #     self.n_e,
        #     self.kmeans_iters,
        # )
        centers, _ = self.constrained_km(data, 256)
        self.embedding.weight.data.copy_(centers)
        self.initted = True
    
    def constrained_km(self, data, n_clusters=10):
        from k_means_constrained import KMeansConstrained 
        x = data.cpu().detach().numpy()

        size_min = min(len(data) // (n_clusters * 2), 50) # 50 for the very first time, 10 the latter

        clf = KMeansConstrained(n_clusters=n_clusters, size_min=size_min, size_max=size_min * 4, max_iter=10, n_init=10,
                                n_jobs=10, verbose=False) # 'size_min * 4' for the very first time, 'n_clusters * 4' for the latter
        clf.fit(x)
        t_centers = torch.from_numpy(clf.cluster_centers_)
        t_labels = torch.from_numpy(clf.labels_).tolist()
        value_counts = {}
        return t_centers, t_labels


    def diversity_loss(self, x_q, indices, indices_cluster, indices_list):
        emb = self.embedding.weight
        temp = 1

        pos_candidates = indices_list[indices]
        pos_candidate_sizes = indices_cluster.clamp_min(1)
        sample_offsets = torch.floor(
            torch.rand(indices.shape[0], device=x_q.device) * pos_candidate_sizes
        ).long()
        y_true = pos_candidates[
            torch.arange(indices.shape[0], device=x_q.device),
            sample_offsets,
        ]

        # sim = F.cosine_similarity(x_q, emb, dim=-1)
        sim = torch.matmul(x_q, emb.t())

        sim = sim.scatter(
            1,
            indices.unsqueeze(1),
            sim.gather(1, indices.unsqueeze(1)) - 1e12,
        )
        sim = sim / temp
        loss = F.cross_entropy(sim, y_true)

        return loss

    def diversity_loss_main_entry(self, x, x_q, indices, labels):
        if isinstance(labels, dict):
            label_tensor = labels["labels"]
            positive_pool = labels["positive_pool"]
            positive_pool_sizes = labels["positive_pool_sizes"]
        else:
            label_tensor = torch.as_tensor(labels, device=x_q.device, dtype=torch.long)
            positive_pool, positive_pool_sizes = self.build_positive_pool(label_tensor)

        indices_cluster = positive_pool_sizes.index_select(0, indices)
        indices_list = positive_pool
        diversity_loss = self.diversity_loss(x_q, indices, indices_cluster, indices_list)

        return diversity_loss

    @staticmethod
    def build_positive_pool(label_tensor):
        label_tensor = label_tensor.long()
        num_codes = label_tensor.shape[0]
        device = label_tensor.device
        code_indices = torch.arange(num_codes, device=device)
        cluster_matches = label_tensor.unsqueeze(0) == label_tensor.unsqueeze(1)
        cluster_matches.fill_diagonal_(False)
        positive_pool_sizes = cluster_matches.sum(dim=1)
        max_pool_size = int(positive_pool_sizes.max().item()) if num_codes > 0 else 1
        max_pool_size = max(max_pool_size, 1)

        positive_pool = code_indices.unsqueeze(1).repeat(1, max_pool_size)
        for code_idx in range(num_codes):
            members = code_indices[cluster_matches[code_idx]]
            if members.numel() == 0:
                continue
            positive_pool[code_idx, :members.numel()] = members
            if members.numel() < max_pool_size:
                positive_pool[code_idx, members.numel():] = members[0]

        return positive_pool, positive_pool_sizes
                    
    
    @staticmethod
    def center_distance_for_constraint(distances):
        # distances: B, K
        max_distance = distances.max()
        min_distance = distances.min()

        middle = (max_distance + min_distance) / 2
        amplitude = max_distance - middle + 1e-5
        assert amplitude > 0
        centered_distances = (distances - middle) / amplitude
        return centered_distances
    
    def vq_init(self, x, use_sk=True):
        latent = x.view(-1, self.e_dim)

        if not self.initted:
            self.init_emb(latent)

        _distance_flag = 'distance'    
        
        if _distance_flag == 'distance':
            d = torch.sum(latent**2, dim=1, keepdim=True) + \
                torch.sum(self.embedding.weight**2, dim=1, keepdim=True).t()- \
                2 * torch.matmul(latent, self.embedding.weight.t())
        else:    
        # Calculate Cosine Similarity 
            d = latent@self.embedding.weight.t()


        if not use_sk or self.sk_epsilon <= 0:
            if _distance_flag == 'distance':
                indices = torch.argmin(d, dim=-1)
            else:    
                indices = torch.argmax(d, dim=-1)
        else:
            d = self.center_distance_for_constraint(d)

            Q = sinkhorn_algorithm(d,self.sk_epsilon,self.sk_iters)
            if torch.isnan(Q).any() or torch.isinf(Q).any():
                print(f"Sinkhorn Algorithm returns nan/inf values.")
            indices = torch.argmax(Q, dim=-1)

        x_q = self.embedding(indices).view(x.shape)

        return x_q
    
    def forward(self,  x, label, idx, use_sk=True):
        # Flatten input
        latent = x.view(-1, self.e_dim)

        if not self.initted and self.training:
            self.init_emb(latent)

        # Calculate the L2 Norm between latent and Embedded weights
        _distance_flag = 'distance'    
        
        if _distance_flag == 'distance':
            d = torch.sum(latent**2, dim=1, keepdim=True) + \
                torch.sum(self.embedding.weight**2, dim=1, keepdim=True).t()- \
                2 * torch.matmul(latent, self.embedding.weight.t())
        else:    
        # Calculate Cosine Similarity 
            d = latent@self.embedding.weight.t()
        if not use_sk or self.sk_epsilon <= 0:
            if _distance_flag == 'distance':
                if idx != -1:
                    indices = torch.argmin(d, dim=-1)
                else:
                    temp = 1.0
                    prob_dist = F.softmax(-d/temp, dim=1)  
                    indices = torch.multinomial(prob_dist, 1).squeeze()
            else:    
                indices = torch.argmax(d, dim=-1)
        else:
            d = self.center_distance_for_constraint(d)

            Q = sinkhorn_algorithm(d,self.sk_epsilon,self.sk_iters)
            # print(Q.sum(0)[:10])
            if torch.isnan(Q).any() or torch.isinf(Q).any():
                print(f"Sinkhorn Algorithm returns nan/inf values.")
            indices = torch.argmax(Q, dim=-1)

        # indices = torch.argmin(d, dim=-1)

        x_q = self.embedding(indices).view(x.shape)

        # Diversity
        diversity_loss = self.diversity_loss_main_entry(x, x_q, indices, label)
        # wandb.log({'diversity_loss': diversity_loss})

        # compute loss for embedding
        commitment_loss = F.mse_loss(x_q.detach(), x)
        codebook_loss = F.mse_loss(x_q, x.detach())

        loss = codebook_loss + self.mu * commitment_loss + self.beta * diversity_loss


        # preserve gradients
        x_q = x + (x_q - x).detach()

        indices = indices.view(x.shape[:-1])

        return x_q, loss, indices


