import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import wandb
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

from ssv2a.model.modules import PositionalEmbedding, TransEncoder, MLP, sample_normal
# from ssv2a.train.loss import kld


class Aggregator(nn.Module):
    def __init__(self, emb_dim=512, device='cuda'):
        super().__init__()
        self.device = device

        self.pe = PositionalEmbedding(emb_dim, resolution=1024, inject_method='add', device=device)
        self.encoder = TransEncoder(num_layers=1, embed_dim=emb_dim, nhead=8, dropout=.2, exp_rate=2)
        self.pred_token = nn.Parameter(torch.zeros(2, emb_dim))
        nn.init.normal_(self.pred_token, mean=0, std=1)
        self.head_mu = MLP(layers=[emb_dim] * 2, dropout=.2)
        self.head_sigma = MLP(layers=[emb_dim] * 2, dropout=.2)
        self.out_mu = nn.Linear(emb_dim, emb_dim)
        self.out_sigma = nn.Linear(emb_dim, emb_dim)

        self.to(device)
        self.float()

    def forward(self, x):
        """
        :param x: [B, L, E]
        """
        x = x.to(self.device)
        x = torch.cat([torch.tile(self.pred_token, (x.shape[0], 1, 1)), x], dim=1)
        x = self.pe(x)
        x = self.encoder(x)
        return self.out_mu(self.head_mu(x[:, 0, :])), self.out_sigma(self.head_sigma(x[:, 1, :]))

    @torch.no_grad()
    def sample(self, x, var_samples=1):
        mu, sigma = self.forward(x)
        clap = torch.zeros(mu.shape).to(self.device)
        for i in range(var_samples):
            clap += sample_normal(mu, sigma) / var_samples
        return clap / clap.norm(p=2, dim=-1, keepdim=True)


class VideoCLAPDataset(Dataset):
    def __init__(self, claps_dir):
        self.clap_fs = [str(p) for p in Path(claps_dir).glob('*.npy')]

    def __len__(self):
        return len(self.clap_fs)

    def __getitem__(self, idx):
        claps = np.load(self.clap_fs[idx])
        return torch.from_numpy(claps[1:]), torch.from_numpy(claps[0])


def collate_claps(data):
    frame_claps = torch.stack([d[0] for d in data])
    gt_claps = torch.stack([d[1] for d in data])
    return frame_claps, gt_claps


class AggTrainer:
    def __init__(self, model:Aggregator, claps_dir, ckpt_dir, batch_size=64, var_samples=1):
        claps_dir = Path(claps_dir)
        self.name = 'N.A.'
        self.ckpt_dir = Path(ckpt_dir) / self.name

        self.model = model

        self.train_loader = DataLoader(VideoCLAPDataset(claps_dir / 'train'),
                                       batch_size=batch_size, collate_fn=collate_claps)
        self.val_loader = DataLoader(VideoCLAPDataset(claps_dir / 'val'),
                                     batch_size=batch_size, collate_fn=collate_claps)
        self.test_loader = DataLoader(VideoCLAPDataset(claps_dir / 'test'),
                                      batch_size=batch_size, collate_fn=collate_claps)
        self.var_samples = var_samples

    def compute_loss(self, frame_claps, gt_claps):
        mu, sigma = self.model(frame_claps)
        # kl = torch.mean(kld(mu, sigma))
        kl = 0

        mu = mu.tile(self.var_samples, 1)
        sigma = sigma.tile(self.var_samples, 1)
        gt_claps = gt_claps.tile(self.var_samples, 1).to(self.model.device)

        gen_claps = sample_normal(mu, sigma)
        gen_claps = gen_claps / gen_claps.norm(p=2, dim=-1, keepdim=True)
        gen_loss = torch.mean((1 - F.cosine_similarity(gt_claps, gen_claps)) ** 2)

        return gen_loss + .001 * kl


    def train(self, epochs=64, report_interval=1):
        best_val_loss = 1e6

        # wandb
        run = wandb.init(project='SDV2A-Agg')
        self.name = run.name
        self.ckpt_dir = self.ckpt_dir.parent / self.name
        os.makedirs(self.ckpt_dir, exist_ok=True)

        # optimizer
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4)

        # epoch
        for epoch in tqdm(range(epochs)):
            wandb_log = {}

            # step
            train_loss = []
            for batch, (frame_claps, gt_claps) in enumerate(self.train_loader):
                loss = self.compute_loss(frame_claps, gt_claps)
                # back propagation
                loss.backward()
                # optimize
                optimizer.step()
                optimizer.zero_grad()
                train_loss.append(loss.detach().cpu().item())
            train_loss = np.mean(train_loss)
            torch.save(self.model.state_dict(), self.ckpt_dir / 'latest.pth')

            # evaluate
            val_loss = []
            for batch, (frame_claps, gt_claps) in enumerate(self.val_loader):
                loss = self.compute_loss(frame_claps, gt_claps).detach().cpu().item()
                val_loss.append(loss)
            val_loss = np.mean(val_loss)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), self.ckpt_dir / 'best_val.pth')

            test_loss = []
            for batch, (frame_claps, gt_claps) in enumerate(self.test_loader):
                loss = self.compute_loss(frame_claps, gt_claps).detach().cpu().item()
                test_loss.append(loss)
            test_loss = np.mean(test_loss)

            # report
            if epoch % report_interval == 0:
                print(f"Epoch {epoch + 1} - train loss: {train_loss:.5f} "
                      f"validation loss: {val_loss:.5f} test loss: {test_loss:.5f}")

            wandb_log['train_loss'] = train_loss
            wandb_log['val_loss'] = val_loss
            wandb_log['test_loss'] = test_loss
            wandb.log(wandb_log)

        wandb.finish()

