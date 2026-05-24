import logging
import json
import numpy as np
import torch
import random
from time import time
from torch import optim
from tqdm import tqdm

import torch.nn.functional as F
from utils import ensure_dir, set_color, get_local_time
import os
from datasets import EmbDataset
from torch.utils.data import DataLoader


class Trainer(object):
    def __init__(self, args, model):
        self.args = args
        self.model = model
        self.logger = logging.getLogger()

        self.lr = args.lr
        self.learner = args.learner
        self.weight_decay = args.weight_decay
        self.epochs = args.epochs
        self.eval_step = min(args.eval_step, self.epochs)
        self.device = self._resolve_device(args.device)
        self.ckpt_dir = args.ckpt_dir
        ensure_dir(self.ckpt_dir)
        self.best_loss = np.inf
        self.best_collision_rate = np.inf
        self.best_loss_ckpt = "best_loss_model.pth"
        self.best_collision_ckpt = "best_collision_model.pth"
        self.optimizer = self._build_optimizer()
        self.model = self.model.to(self.device)
        self.trained_loss = {"total": [], "rqvae": [], "recon": []}
        self.valid_collision_rate = {"val": []}

    def _resolve_device(self, device_str):
        if device_str == "cpu":
            return torch.device("cpu")
        if device_str.startswith("cuda"):
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA requested but not available")
            try:
                requested_index = int(device_str.split(":", 1)[1]) if ":" in device_str else 0
            except ValueError:
                requested_index = 0
            visible_count = torch.cuda.device_count()
            if requested_index >= visible_count:
                if visible_count == 1:
                    self.logger.warning(
                        "Requested device %s is invalid after CUDA_VISIBLE_DEVICES remapping; falling back to cuda:0",
                        device_str,
                    )
                    return torch.device("cuda:0")
                raise RuntimeError(
                    f"Invalid CUDA device ordinal: requested {device_str}, visible device count is {visible_count}"
                )
        return torch.device(device_str)

    def _build_optimizer(self):
        params = self.model.parameters()
        learner = self.learner
        learning_rate = self.lr
        weight_decay = self.weight_decay

        if learner.lower() == "adam":
            optimizer = optim.Adam(params, lr=learning_rate, weight_decay=weight_decay)
        elif learner.lower() == "sgd":
            optimizer = optim.SGD(params, lr=learning_rate, weight_decay=weight_decay)
        elif learner.lower() == "adagrad":
            optimizer = optim.Adagrad(
                params, lr=learning_rate, weight_decay=weight_decay
            )
            for state in optimizer.state.values():
                for key, value in state.items():
                    if torch.is_tensor(value):
                        state[key] = value.to(self.device)
        elif learner.lower() == "rmsprop":
            optimizer = optim.RMSprop(
                params, lr=learning_rate, weight_decay=weight_decay
            )
        elif learner.lower() == 'adamw':
            optimizer = optim.AdamW(
                params, lr=learning_rate, weight_decay=weight_decay
            )
        else:
            self.logger.warning(
                "Received unrecognized optimizer, set default Adam optimizer"
            )
            optimizer = optim.Adam(params, lr=learning_rate)
        return optimizer

    def _check_nan(self, loss):
        if torch.isnan(loss):
            raise ValueError("Training loss is nan")

    def vq_init(self):
        self.model.eval()
        original_data = EmbDataset(
            self.args.data_path,
            embedding_dim=getattr(self.args, "embedding_dim", 256),
            pca_dim=getattr(self.args, "pca_dim", None),
        )
        init_loader = DataLoader(
            original_data,
            num_workers=self.args.num_workers,
            batch_size=len(original_data),
            shuffle=True,
            pin_memory=True,
        )
        iter_data = tqdm(
            init_loader,
            total=len(init_loader),
            ncols=100,
            desc=set_color("Initialization of vq", "pink"),
        )
        for _, data in enumerate(iter_data):
            batch_data = data[0].to(self.device)
            self.model.vq_initialization(batch_data)

    def _train_epoch(self, train_data, epoch_idx):
        self.model.train()

        total_loss = torch.zeros((), device=self.device)
        total_recon_loss = torch.zeros((), device=self.device)
        total_cf_loss = torch.zeros((), device=self.device)
        total_quant_loss = torch.zeros((), device=self.device)
        iter_data = tqdm(
            train_data,
            total=len(train_data),
            ncols=100,
            desc=set_color(f"Train {epoch_idx}", "pink"),
        )

        for _, data in enumerate(iter_data):
            batch_data, emb_idx = data[0], data[1]
            batch_data = batch_data.to(self.device)
            emb_idx = emb_idx.to(self.device)
            self.optimizer.zero_grad(set_to_none=True)
            out, rq_loss, indices, dense_out = self.model(batch_data)

            loss, cf_loss, loss_recon, quant_loss = self.model.compute_loss(
                out, rq_loss, emb_idx, dense_out, xs=batch_data
            )
            self._check_nan(loss)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.detach()
            total_recon_loss += loss_recon.detach()
            total_cf_loss += cf_loss.detach()
            total_quant_loss += quant_loss.detach()

        return (
            total_loss.item(),
            total_recon_loss.item(),
            total_cf_loss.item(),
            total_quant_loss.item(),
        )

    @torch.no_grad()
    def _valid_epoch(self, valid_data):
        self.model.eval()

        iter_data = tqdm(
            valid_data,
            total=len(valid_data),
            ncols=100,
            desc=set_color("Evaluate   ", "pink"),
        )
        indices_set = set()

        num_sample = 0
        for _, data in enumerate(iter_data):
            batch_data, emb_idx = data[0], data[1]
            num_sample += len(batch_data)
            batch_data = batch_data.to(self.device)
            indices = self.model.get_indices(batch_data)
            indices = indices.view(-1, indices.shape[-1]).cpu().numpy()
            for index in indices:
                code = "-".join([str(int(_)) for _ in index])
                indices_set.add(code)

        collision_rate = (num_sample - len(indices_set)) / num_sample
        return collision_rate

    def _save_checkpoint(self, epoch, collision_rate=1, ckpt_file=None):
        ckpt_path = os.path.join(self.ckpt_dir, ckpt_file) if ckpt_file else os.path.join(
            self.ckpt_dir, 'epoch_%d_collision_%.4f_model.pth' % (epoch, collision_rate)
        )
        state = {
            "args": self.args,
            "epoch": epoch,
            "best_loss": self.best_loss,
            "best_collision_rate": self.best_collision_rate,
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        torch.save(state, ckpt_path, pickle_protocol=4)

        self.logger.info(
            set_color("Saving current", "blue") + f": {ckpt_path}"
        )

    def _generate_train_loss_output(self, epoch_idx, s_time, e_time, loss, recon_loss, cf_loss):
        train_loss_output = (
            set_color("epoch %d training", "green")
            + " ["
            + set_color("time", "blue")
            + ": %.2fs, "
        ) % (epoch_idx, e_time - s_time)
        train_loss_output += set_color("train loss", "blue") + ": %.4f" % loss
        train_loss_output += ", "
        train_loss_output += set_color("reconstruction loss", "blue") + ": %.4f" % recon_loss
        train_loss_output += ", "
        train_loss_output += set_color("cf loss", "blue") + ": %.4f" % cf_loss
        return train_loss_output + "]"

    def fit(self, data):
        cur_eval_step = 0
        self.vq_init()
        for epoch_idx in range(self.epochs):
            training_start_time = time()
            train_loss, train_recon_loss, cf_loss, quant_loss = self._train_epoch(data, epoch_idx)

            training_end_time = time()
            train_loss_output = self._generate_train_loss_output(
                epoch_idx, training_start_time, training_end_time, train_loss, train_recon_loss, cf_loss
            )
            self.logger.info(train_loss_output)

            if train_loss < self.best_loss:
                self.best_loss = train_loss

            if (epoch_idx + 1) % self.eval_step == 0:
                valid_start_time = time()
                collision_rate = self._valid_epoch(data)

                if collision_rate < self.best_collision_rate:
                    self.best_collision_rate = collision_rate
                    cur_eval_step = 0
                    self._save_checkpoint(
                        epoch_idx,
                        collision_rate=collision_rate,
                        ckpt_file=self.best_collision_ckpt,
                    )
                else:
                    cur_eval_step += 1

                valid_end_time = time()
                valid_score_output = (
                    set_color("epoch %d evaluating", "green")
                    + " ["
                    + set_color("time", "blue")
                    + ": %.2fs, "
                    + set_color("collision_rate", "blue")
                    + ": %f]"
                ) % (epoch_idx, valid_end_time - valid_start_time, collision_rate)

                self.logger.info(valid_score_output)

                if epoch_idx > 2500:
                    self._save_checkpoint(epoch_idx, collision_rate=collision_rate)

            if epoch_idx + 1 == self.epochs:
                self._save_checkpoint(epoch_idx, collision_rate=self.best_collision_rate, ckpt_file='epoch_9999_collision_forced_model.pth')

        return self.best_loss, self.best_collision_rate
