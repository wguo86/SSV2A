import math

import torch
import torch.nn as nn

from ssv2a.data.utils import read_classes
from ssv2a.model.modules import MLP, LinearProjection, sample_normal, TransEncoder, PositionalEmbedding, MoE
# from ssv2a.train.loss import contrastive_loss, kld, rrl


class ManifoldEncoder(nn.Module):
    def __init__(self, in_dim, manifold_dim, manifold=None, device='cuda'):
        super().__init__()
        manifold['device'] = device
        self.variational = manifold['variational']
        self.model = None
        hidden_dim = in_dim

        self.arch = manifold['arch']
        if self.arch == 'mlp':
            self.model = nn.Sequential()
            self.model.append(nn.Linear(in_dim, manifold['layers'][0], bias=False))  # in projection
            self.model.append(MLP(**manifold))
            hidden_dim = manifold['layers'][-1]

        elif self.arch == 'linear':
            self.model = LinearProjection(in_dim, manifold_dim)
            hidden_dim = manifold_dim

        elif self.arch == 'transformer':
            self.embed_dim = manifold['embed_dim']
            self.out_dim = manifold['out_dim']
            if manifold['pe_inject'] == 'cat':
                self.embed_dim = 2 * self.embed_dim
            self.patches = manifold['patches']

            # learnable manifold embedding, attached to start of the sequence
            self.fold_embedding = nn.Parameter(torch.zeros(self.out_dim // self.embed_dim, self.embed_dim))
            nn.init.normal_(self.fold_embedding, mean=0.0, std=1.0)

            self.pos_embed = PositionalEmbedding(self.embed_dim,
                                                 resolution=manifold['pe_res'],
                                                 inject_method=manifold['pe_inject'],
                                                 device=device)

            self.in_proj = nn.Linear(in_dim // self.patches, self.embed_dim)  # in projection
            self.transformer = TransEncoder(**manifold)

            self.model = MLP(**manifold)  # model is the prediction head
            hidden_dim = manifold['layers'][-1]

        elif self.arch == 'moe':
            self.experts = manifold['experts']
            self.moe = MoE(manifold, experts=self.experts,
                           diverse_experts=manifold['diverse_experts'], device=device)

            self.rrl_weight = manifold['rrl_weight']
            self.router = MLP([in_dim, (in_dim + self.experts) // 2, self.experts])

            self.model = nn.Linear(manifold['layers'][-1], manifold['layers'][-1], bias=False)  # head is out projection

            hidden_dim = manifold['layers'][-1]

        if self.model is None:
            raise Exception('Illegal config for manifold encoder, abort.')

        # out projection
        if self.variational:
            self.out_mu = nn.Linear(hidden_dim, manifold_dim)
            self.out_sigma = nn.Linear(hidden_dim, manifold_dim)
        elif manifold['arch'] != 'linear':
            self.out = nn.Linear(hidden_dim, manifold_dim)

    def forward(self, x):
        router_reg_loss = 0

        if self.arch == 'transformer':
            x = self.in_proj(x.reshape(x.shape[0], self.patches, -1))  # patching
            x = torch.cat([self.fold_embedding.tile((x.shape[0], 1, 1)), x], dim=1)
            x = self.pos_embed(x)
            x = self.transformer(x)[:, :self.fold_embedding.shape[0], :]  # extract prediction token
            x = torch.flatten(x, start_dim=1)

        elif self.arch == 'moe':
            ws = self.router(x)
            ws = torch.softmax(ws, dim=-1)

            if self.training:  # router regularization loss
                # router_reg_loss = rrl(ws) * self.rrl_weight
                router_reg_loss = 0

            x = self.moe(x, ws)

        elif self.arch == 'linear':
            return self.model(x), router_reg_loss

        x = self.model(x)
        if not self.variational:
            return self.out(x), router_reg_loss
        mu = self.out_mu(x)
        log_sigma = self.out_sigma(x)

        return mu, log_sigma, router_reg_loss


class Manifold(nn.Module):
    def __init__(self, clip_dim=512, clap_dim=512, manifold_dim=128, classes='', manifold=None, device='cuda', **_):
        super().__init__()
        self.device = device
        self.model_id = 'manifold'
        self.variational = manifold['variational']
        self.kl_weight = manifold['kl_weight']
        self.clap_dim = clap_dim
        self.clip_encoder = ManifoldEncoder(clip_dim, manifold_dim, manifold, device=device)
        self.clap_encoder = ManifoldEncoder(clap_dim, manifold_dim, manifold, device=device)

        # randomly initialize learnable contrastive temperatures
        self.logit_scale = nn.Parameter(torch.ones([]) * math.log(1 / 0.07))
        self.self_logit_scale = nn.Parameter(torch.ones([]) * math.log(1 / 0.07))
        self.logit_scale_min = math.log(1)
        self.logit_scale_max = math.log(100)

        self.cr_weight = manifold['cr_weight']

        self.to(device)
        self.float()

    def forward(self, clips, claps, contrast_mask=None, kl_weight=.001, return_loss=False, var_samples=1):
        kl = 0
        loss = None
        self.logit_scale.data.clamp_(self.logit_scale_min, self.logit_scale_max)
        self.self_logit_scale.data.clamp_(self.logit_scale_min, self.logit_scale_max)

        if self.variational:
            clip_mu, clip_log_sigma, clip_rrl = self.clip_encoder(clips)
            clap_mu, clap_log_sigma, clap_rrl = self.clap_encoder(claps)

            clip_embeds = torch.empty(var_samples, clip_mu.shape[0], clip_mu.shape[1]).to(self.device)
            clap_embeds = torch.empty(var_samples, clap_mu.shape[0], clap_mu.shape[1]).to(self.device)
            for i in range(var_samples):
                emb1 = sample_normal(clip_mu, clip_log_sigma)
                clip_embeds[i, :, :] = emb1
                emb2 = sample_normal(clap_mu, clap_log_sigma)
                clap_embeds[i, :, :] = emb2

            # calculate kl distance together since clip and clap will be blended on the manifold
            if return_loss:
                mu = torch.cat([clip_mu, clap_mu])
                log_sigma = torch.cat([clip_log_sigma, clap_log_sigma])
                # kl = torch.mean(kld(mu, log_sigma))
                kl = 0

                n_clip_embeds = clip_embeds / clip_embeds.norm(dim=-1, keepdim=True)
                n_clap_embeds = clap_embeds / clap_embeds.norm(dim=-1, keepdim=True)
                n_clip_embeds_t = torch.einsum('bij->bji', n_clip_embeds)
                n_clap_embeds_t = torch.einsum('bij->bji', n_clap_embeds)

                # random permutation for monte carlo contrastive loss
                clip_loss = 0
                clap_loss = 0
                B = n_clip_embeds.shape[1]
                for i in range(var_samples):  # logits: [var_samples, B, B]
                    p1 = torch.randperm(var_samples).to(self.device)
                    p2 = torch.randperm(var_samples).to(self.device)
                    p3 = torch.randperm(var_samples).to(self.device)
                    p4 = torch.randperm(var_samples).to(self.device)

                    logits_per_clap = torch.einsum('bij,bjk->bik', n_clip_embeds[p1], n_clap_embeds_t[p2])
                    logits_per_clip = torch.einsum('bij,bjk->bik', n_clap_embeds[p3], n_clip_embeds_t[p4])

                    if contrast_mask is not None:
                        logits_per_clap *= contrast_mask
                        logits_per_clip *= contrast_mask

                    logits_per_clap *= self.logit_scale.exp()
                    logits_per_clip *= self.logit_scale.exp()

                    clap_loss += (
                        nn.functional.cross_entropy(
                            logits_per_clap.reshape(-1, B),
                            torch.arange(B).tile(var_samples).to(self.device)))
                    clip_loss += (
                        nn.functional.cross_entropy(
                            logits_per_clip.reshape(-1, B),
                            torch.arange(B).tile(var_samples).to(self.device)))

                loss = (clap_loss + clip_loss) / var_samples * .5 + kl_weight * kl + clip_rrl + clap_rrl

            return (clip_embeds.reshape(-1, clip_embeds.size(-1)),
                    clap_embeds.reshape(-1, clap_embeds.size(-1)), loss)

        else:
            clip_embeds, clip_rrl = self.clip_encoder(clips)
            clap_embeds, clap_rrl = self.clap_encoder(claps)

            # normalized features
            n_clip_embeds = clip_embeds / clip_embeds.norm(dim=-1, keepdim=True)
            n_clap_embeds = clap_embeds / clap_embeds.norm(dim=-1, keepdim=True)

            if return_loss:
                # cosine similarity as logits
                logits_per_clap = n_clap_embeds @ n_clip_embeds.t() * self.logit_scale.exp()
                logits_per_clip = n_clip_embeds @ n_clap_embeds.t() * self.logit_scale.exp()

                if contrast_mask is not None:
                    logits_per_clap *= contrast_mask
                    logits_per_clip *= contrast_mask

                # clap_loss = contrastive_loss(logits_per_clap)
                # clip_loss = contrastive_loss(logits_per_clip.t())
                clap_loss = 0
                clip_loss = 0

                loss = (clap_loss + clip_loss) * .5 + kl_weight * kl + clip_rrl + clap_rrl

            return clip_embeds, clap_embeds, loss

    def fold_clips(self, clips, var_samples=1, normalize=False):
        if self.variational:
            mu, sigma, _ = self.clip_encoder(clips)
            fold_clips = torch.zeros(mu.shape).to(self.device)
            for i in range(var_samples):  # repetitive sampling
                fold_clips += sample_normal(mu, sigma) / var_samples
        else:
            fold_clips, _ = self.clip_encoder(clips)

        if normalize:
            fold_clips = fold_clips / fold_clips.norm(p=2, dim=-1, keepdim=True)

        return fold_clips

    def fold_claps(self, claps, var_samples=1, normalize=False):
        if self.variational:
            mu, sigma, _ = self.clap_encoder(claps)
            fold_claps = torch.zeros(mu.shape).to(self.device)
            for i in range(var_samples):  # repetitive sampling
                fold_claps += sample_normal(mu, sigma) / var_samples
        else:
            fold_claps, _ = self.clap_encoder(claps)

        if normalize:
            fold_claps = fold_claps / fold_claps.norm(p=2, dim=-1, keepdim=True)

        return fold_claps

