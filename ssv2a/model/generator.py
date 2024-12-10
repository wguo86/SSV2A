import torch
import torch.nn as nn

from ssv2a.model.modules import MLP, TransEncoder, PositionalEmbedding, sample_normal, MoE
# from ssv2a.train.loss import rrl


class Generator(nn.Module):
    def __init__(self, clap_dim=512, manifold_dim=128, generator=None, device='cuda', **_):
        super().__init__()
        self.device = device
        generator['device'] = device
        self.model = None
        self.model_id = 'generator'
        self.variational = generator['variational']

        hidden_dim = manifold_dim
        self.kl_weight = generator['kl_weight']

        self.arch = generator['arch']
        if self.arch == 'mlp':
            self.model = nn.Sequential()
            self.model.append(nn.Linear(manifold_dim, generator['layers'][0]))
            self.model.append(MLP(**generator))
            hidden_dim = generator['layers'][-1]

        elif self.arch == 'transformer':
            self.embed_dim = generator['embed_dim']
            self.out_dim = generator['out_dim']
            if generator['pe_inject'] == 'cat':
                self.embed_dim = 2 * self.embed_dim
            self.patches = generator['patches']

            # learnable generator embedding, attached to start of the sequence
            self.gen_embedding = nn.Parameter(torch.zeros(self.out_dim // self.embed_dim, self.embed_dim))
            nn.init.normal_(self.gen_embedding, mean=0.0, std=1.0)

            self.pos_embed = PositionalEmbedding(self.embed_dim,
                                                 resolution=generator['pe_res'],
                                                 inject_method=generator['pe_inject'],
                                                 device=device)

            self.in_proj = nn.Linear(manifold_dim // self.patches, self.embed_dim)  # in projection
            self.transformer = TransEncoder(**generator)

            self.model = MLP(**generator)  # model is the prediction head
            hidden_dim = generator['layers'][-1]

        elif self.arch == 'moe':
            self.experts = generator['experts']
            self.moe = MoE(generator, experts=self.experts,
                           diverse_experts=generator['diverse_experts'], device=device)

            self.rrl_weight = generator['rrl_weight']
            self.router = MLP([manifold_dim, (manifold_dim + self.experts) // 2, self.experts])

            self.model = nn.Linear(generator['layers'][-1], generator['layers'][-1], bias=False)  # head is out projection

            hidden_dim = generator['layers'][-1]

        if self.model is None:
            raise Exception('Illegal config for generator, abort.')

        # out projection
        if generator['variational']:
            self.out_mu = nn.Linear(hidden_dim, clap_dim)
            self.out_sigma = nn.Linear(hidden_dim, clap_dim)
        else:
            self.out = nn.Linear(hidden_dim, clap_dim)

        self.to(device)
        self.float()

    def forward(self, x):
        # tile embedding into [B, L, E] if arch is transformer
        router_reg_loss = 0

        if self.arch == 'transformer':
            x = self.in_proj(x.reshape(x.shape[0], self.patches, -1))
            x = torch.cat([self.gen_embedding.tile((x.shape[0], 1, 1)), x], dim=1)
            x = self.pos_embed(x)
            x = self.transformer(x)[:, :self.gen_embedding.shape[0], :]  # extract prediction token
            x = torch.flatten(x, start_dim=1)

        elif self.arch == 'moe':
            ws = self.router(x)
            ws = torch.softmax(ws, dim=-1)

            if self.training:  # router regularization loss
                # router_reg_loss = rrl(ws) * self.rrl_weight
                router_reg_loss = 0

            x = self.moe(x, ws)

        if not self.variational:
            return self.out(self.model(x)), router_reg_loss
        x = self.model(x)
        mu = self.out_mu(x)
        log_sigma = self.out_sigma(x)

        return mu, log_sigma, router_reg_loss

    def fold2claps(self, folds, var_samples=64):
        if self.variational:
            mu, sigma, _ = self.forward(folds)
            gen_claps = torch.zeros(mu.shape).to(self.device)
            for i in range(var_samples):
                gen_claps += sample_normal(mu, sigma) / var_samples
        else:
            gen_claps, _ = self.forward(folds)

        gen_claps = gen_claps / gen_claps.norm(p=2, dim=-1, keepdim=True)
        return gen_claps

