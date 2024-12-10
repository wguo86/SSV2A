import random

import torch
import torch.nn as nn

from ssv2a.data.utils import random_mute
from ssv2a.model.modules import TransEncoder, MLP, sample_normal


class PanauralGate(nn.Module):
    def __init__(self, gate=None, slot=64, clip_dim=512):
        super().__init__()
        self.slot = slot
        self.arch = gate['arch']

        hidden_dim = clip_dim
        if self.arch == 'transformer':
            self.embed_dim = gate['embed_dim']
            self.in_proj = nn.Linear(clip_dim, self.embed_dim)

            self.encoder = TransEncoder(**gate)
            self.pred_token = nn.Parameter(torch.zeros(1, clip_dim))
            nn.init.normal_(self.pred_token, mean=0, std=1)

            self.head = MLP(**gate)

            hidden_dim = gate['layers'][-1]

        else:
            raise NotImplementedError('Architecture is not supported.')

        self.out = nn.Linear(hidden_dim, self.slot)

    def forward(self, x):
        """
        x: [B, slot, clip_dim]
        """
        if self.arch == 'transformer':
            ws = torch.cat([torch.tile(self.pred_token, (x.shape[0], 1, 1)), x], dim=1)
            ws = self.in_proj(x)
            ws = self.encoder(x)[:, 0, :]
            ws = self.head(ws)

        else:
            raise NotImplementedError('Architecture is not supported.')

        ws = self.out(ws)  # [B, slot]
        ws = ws * torch.sum(x, dim=-1)  # zero out empty conditions
        ws = torch.nn.functional.relu(ws)
        ws = ws / ws.norm(p=2, dim=-1, keepdim=True)
        return ws


class Styler(nn.Module):
    def __init__(self, styler=None, slot=64, clap_dim=512, clip_dim=512, manifold_dim=512, device='cuda'):
        super().__init__()
        self.slot = slot
        self.variational = styler['variational']
        self.arch = styler['arch']
        self.device = device

        if self.arch == 'transformer':
            self.embed_dim = styler['embed_dim']

            self.in_proj = nn.Linear(manifold_dim + clip_dim, self.embed_dim)

            self.style_encoder = TransEncoder(**styler)

            if self.variational:
                self.pred_token = nn.Parameter(torch.zeros(2, self.embed_dim))
                self.head_mu = MLP(layers=[self.embed_dim] * 2, dropout=styler['dropout'])
                self.head_sigma = MLP(layers=[self.embed_dim] * 2, dropout=styler['dropout'])
            else:
                self.pred_token = nn.Parameter(torch.zeros(1, self.embed_dim))
                self.head_mu = MLP(layers=[self.embed_dim] * 2, dropout=styler['dropout'])
            nn.init.normal_(self.pred_token, mean=0, std=1)

            hidden_dim = self.embed_dim

        else:
            raise NotImplementedError('Architecture is not supported.')

        if self.variational:
            self.out_mu = nn.Linear(hidden_dim, clap_dim)
            self.out_sigma = nn.Linear(hidden_dim, clap_dim)
        else:
            self.out = nn.Linear(hidden_dim, clap_dim)

    def forward(self, src, src_clips):
        """
        src: [B, L, manifold_dim] (manifold queried by style semantics)
        src_clips: [B, L, clip_dim] (semantic clip embeddings)
        locality: [B, L]  (not used for now)
        """
        if self.arch == 'transformer':
            # src = torch.zeros(src.shape).to(self.device)  # suppress manifold embed for ablation
            # src_clips = torch.zeros(src_clips.shape).to(self.device)  # suppress clip embed for ablation

            src = torch.cat([src, src_clips], dim=-1)
            src = self.in_proj(src)

            src = torch.cat([torch.tile(self.pred_token, (src.shape[0], 1, 1)), src], dim=1)
            src = self.style_encoder(src)

            if self.variational:
                mu = self.head_mu(src[:, 0, :])
                sigma = self.head_sigma(src[:, 1, :])
            else:
                mu = self.head_mu(src[:, 0, :])

        else:
            raise NotImplementedError('Architecture is not supported.')

        if self.variational:
            return self.out_mu(mu), self.out_sigma(sigma)
        return self.out(mu)

    def sample(self, src, src_clips, var_samples=1):
        if self.variational:
            mu, log_sigma = self.forward(src, src_clips)
            clap = torch.zeros(mu.shape).to(self.device)
            for i in range(var_samples):
                clap += sample_normal(mu, log_sigma) / var_samples
        else:
            clap = self.forward(src, src_clips)
        return clap


class Remixer(nn.Module):
    def __init__(self, remixer, clip_dim=512, clap_dim=512, manifold_dim=512, device='cuda', **_):
        super().__init__()
        self.slot = remixer['slot']
        self.kl_weight = remixer['styler']['kl_weight']
        self.variational = remixer['styler']['variational']
        self.guidance = remixer["guidance"]

        self.cfg = remixer['cfg']

        if self.guidance == 'manifold+generator':
            manifold_dim += clap_dim
        elif self.guidance == 'generator':
            manifold_dim = clap_dim
        self.styler = Styler(styler=remixer['styler'], slot=self.slot, clap_dim=clap_dim, clip_dim=clip_dim,
                             manifold_dim=manifold_dim, device=device)

        self.to(device)
        self.float()

    def forward(self, src, src_clips):
        """
        src: [B, L, manifold_dim] (manifold queried by style semantics)
        style: [B, L, clip_dim] (unannotated style semantics)
        """
        if self.training:  # classifier free guidance
            src, src_mask_idx = random_mute(src, p=self.cfg)
            src_clips, src_mask_idx = random_mute(src_clips, p=self.cfg)

        return self.styler(src, src_clips)

    def sample(self, src, src_clips, var_samples=1, normalize=True, shuffle=True):
        if shuffle:
            idx = random.sample(range(self.slot), self.slot)
            src = src[:, idx, :]
            src_clips = src_clips[:, idx, :]
        clap = self.styler.sample(src, src_clips, var_samples=var_samples)
        if normalize:
            clap = clap / clap.norm(p=2, dim=-1, keepdim=True)
        return clap

