import copy
import math

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from typing import Callable, List, Optional


# generate a Gaussian sample given a batch of means and variances
def sample_normal(mu, log_sigma):
    std = torch.exp(0.5 * log_sigma)
    eps = std.data.new(std.size()).normal_()
    return eps.mul(std) + mu


# good old positional embedding
# adapted from https://pytorch.org/tutorials/beginner/transformer_tutorial.html
class PositionalEmbedding(nn.Module):
    def __init__(self, embed_dims, resolution=1024, inject_method='cat', device='cuda'):
        super().__init__()
        self.inject_method = inject_method
        self.resolution = resolution
        self.device = device
        pe = torch.zeros(resolution, embed_dims)
        position = torch.arange(resolution, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dims, 2).float() * (-math.log(2 * resolution) / embed_dims))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x, *args):
        # x: [B, L, E] steps: [B, L]
        ge = self.pe[:x.shape[1]].squeeze().tile(x.shape[0], 1, 1)

        if self.inject_method == 'add':
            return x + ge
        return torch.cat([x, ge], dim=-1)


class LocalityEmbedding(PositionalEmbedding):
    def forward(self, x, *args):
        # x: [B, L, E] localities: [B, L]
        localities = torch.round(args[0] * (self.resolution - 1)).int()
        ge = torch.empty(x.shape).to(self.device)
        for i in range(localities.shape[0]):
            ge[i, :, :] = self.pe[localities[i]].squeeze()

        if self.inject_method == 'add':
            return x + ge
        return torch.cat([x, ge], dim=-1)


'''
Transformer
'''


# efficient attention, adapted from
# https://github.com/mingyuan-zhang/MotionDiffuse/blob/main/text2motion/models/transformer.py
class EfficientSelfAttention(nn.Module):
    def __init__(self, embed_dim, nhead, dropout):
        super().__init__()
        self.nhead = nhead
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(p=dropout)
        self.norm = nn.LayerNorm(embed_dim)
        self.highway = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        # x: B, T, D
        B, T, D = x.shape
        H = self.nhead

        # linear projections and split into multiheads (B, T, H, D//H)
        nx = self.norm(x)
        query = self.query(nx).view(B, T, H, -1)
        key = self.key(nx).view(B, T, H, -1)
        value = self.value(nx).view(B, T, H, -1)

        # attention (B, T, H, D//H) -> (B, T, D)
        query = F.softmax(query, dim=-1)
        key = F.softmax(key, dim=-1)
        attention = self.dropout(torch.einsum('bnhd,bnhl->bhdl', key, value))
        y = torch.einsum('bnhd,bhdl->bnhl', query, attention).reshape(B, T, D)

        # residual
        y = self.highway(x) + y
        return y


class EfficientCrossAttention(nn.Module):
    def __init__(self, embed_dim, cond_dim, nhead, dropout):
        super().__init__()
        self.nhead = nhead
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(cond_dim, embed_dim)
        self.value = nn.Linear(cond_dim, embed_dim)
        self.dropout = nn.Dropout(p=dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(cond_dim)
        self.highway = nn.Linear(embed_dim, embed_dim)

    def forward(self, x1, x2):
        """
        x1: B, T, D
        x2: B, N, L
        """
        B, T, D = x1.shape
        N = x2.shape[1]
        H = self.nhead

        # linear projections and split into multiheads (B, T, H, D//H)
        nx1 = self.norm1(x1)
        nx2 = self.norm2(x2)
        query = self.query(nx1).view(B, T, H, -1)
        key = self.key(nx2).view(B, N, H, -1)
        value = self.value(nx2).view(B, N, H, -1)

        # attention (B, T, H, D//H), (B, N, H, D//H) -> (B, T, D)
        query = F.softmax(query, dim=-1)
        key = F.softmax(key, dim=-1)
        attention = self.dropout(torch.einsum('bnhd,bnhl->bhdl', key, value))
        y = torch.einsum('bnhd,bhdl->bnhl', query, attention).reshape(B, T, D)

        # residual
        y = self.highway(x1) + y
        return y


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


class FFN(nn.Module):
    def __init__(self, embed_dim, hidden_dim, dropout):
        super().__init__()
        self.linear1 = nn.Linear(embed_dim, hidden_dim)
        self.linear2 = zero_module(nn.Linear(hidden_dim, embed_dim))
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(p=dropout)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        y = self.linear2(self.dropout(self.activation(self.linear1(self.norm(x)))))
        y = x + y
        return y


class ESALayer(nn.Module):
    def __init__(self, embed_dim, ffn_dim, nhead, dropout):
        super().__init__()
        self.esa = EfficientSelfAttention(embed_dim, nhead, dropout)
        self.ffn = FFN(embed_dim, ffn_dim, dropout)

    def forward(self, x):
        return self.ffn(self.esa(x))


class ECALayer(nn.Module):
    def __init__(self, embed_dim, cond_dim, ffn_dim, nhead, dropout):
        super().__init__()
        self.eca = EfficientCrossAttention(embed_dim, cond_dim, nhead, dropout)
        self.ffn = FFN(embed_dim, ffn_dim, dropout)

    def forward(self, x1, x2):
        return self.ffn(self.eca(x1, x2))


class TransEncoder(nn.Module):
    def __init__(self, num_layers=3, embed_dim=512, nhead=8, dropout=.2, exp_rate=1, **_):
        super().__init__()
        self.encoder = nn.Sequential()
        for i in range(num_layers):
            self.encoder.append(ESALayer(embed_dim, embed_dim * exp_rate, nhead, dropout))

    def forward(self, src):
        return self.encoder(src)


class TransDecoder(nn.Module):
    def __init__(self, num_layers=6, embed_dim=512, cond_dim=512, nhead=8, dropout=.2, **_):
        super().__init__()
        self.decoder = nn.ModuleList()
        for i in range(num_layers):
            self.decoder.append(ECALayer(embed_dim, cond_dim, embed_dim * 2, nhead, dropout))

    def forward(self, src, cond):
        for mod in self.decoder:
            src = mod(src, cond)
        return src


'''
Linear Projection
'''


class LinearProjection(nn.Module):
    def __init__(self, in_dim, out_dim, bias=True):
        super().__init__()
        self.A = nn.Parameter(torch.randn(out_dim, in_dim))
        # SVD
        self.U = None
        self.S = None
        self.Vh = None
        if bias:
            self.b = nn.Parameter(torch.zeros(1))
        else:
            self.b = 0

    def forward(self, x):
        return torch.einsum('ij,bj->bi', self.A, x) + self.b

    def svd(self):
        if self.U is None:  # solve once
            with torch.no_grad():
                self.U, self.S, self.Vh = torch.linalg.svd(self.A, full_matrices=False)
        return self.U, self.S, self.Vh

    def solve(self, x):
        # x : [B, E]
        self.svd()
        # broadcast inversion
        return torch.einsum('ij,bj->bi',
                            self.Vh.t() @ torch.diag(1.0 / self.S) @ self.U.t(),
                            x - self.b)


'''
MLPs
'''


# wrap a given module with residual shortcut
class ResWrapper(nn.Module):
    def __init__(self, mod, proj):
        super().__init__()
        self.mod = mod
        self.proj = proj

    def forward(self, x):
        return self.proj(x) + self.mod(x)


class MLP(nn.Module):
    def __init__(self, layers, resnet=False, dropout=.1, activation=nn.GELU, **_):  # pass in layers as a list of ints
        super().__init__()
        resnet = resnet and len(layers) > 2
        self.model = nn.Sequential()
        idx = 0
        if resnet:
            for i in range(0, (len(layers) - 1) // 2):
                a, b, c = i * 2, i * 2 + 1, i * 2 + 2
                idx = c
                mod = nn.Sequential()
                mod.append(nn.Linear(layers[a], layers[b]))
                mod.append(nn.LayerNorm(layers[b]))
                if dropout:
                    mod.append(nn.Dropout(p=dropout))
                mod.append(activation())
                mod.append(nn.Linear(layers[b], layers[c]))
                mod.append(nn.LayerNorm(layers[c]))
                if dropout:
                    mod.append(nn.Dropout(p=dropout))
                self.model.append(ResWrapper(mod, nn.Linear(layers[a], layers[c])))
                self.model.append(activation())

        for i in range(idx, len(layers) - 1):
            self.model.append(nn.Linear(layers[i], layers[i + 1]))
            self.model.append(nn.LayerNorm(layers[i + 1]))
            if dropout:
                self.model.append(nn.Dropout(p=dropout))
            self.model.append(activation())

    def forward(self, x):
        return self.model(x)


class MoE(nn.Module):
    def __init__(self, config, experts=8, diverse_experts=False, device='cuda'):
        super().__init__()
        cfg = copy.deepcopy(config)
        self.out_dim = cfg['layers'][-1]

        if diverse_experts:
            exps = []
            in_dim = cfg['layers'][0]
            hidden_dims = copy.deepcopy(cfg['layers'][1:-1])
            for i in range(len(hidden_dims)):
                cfg['layers'] = [in_dim] + hidden_dims[:i] + [self.out_dim]
                expn = experts // len(hidden_dims)
                if i == len(hidden_dims) - 1:
                    expn = experts - i * expn
                exps += [MLP(**cfg) for _ in range(expn)]

        else:
            exps = [MLP(**cfg) for _ in range(experts)]

        self.experts = nn.ModuleList(exps)

        self.device = device

    def forward(self, x, weights, sequential=False):
        # x: [B, E] is not sequential, else [B, L, E]; weights: [B, experts], else [B, L, experts]
        if not sequential:
            y = torch.empty(x.shape[0], len(self.experts), self.out_dim).to(self.device)  # y: [B, experts, E]
            for i, e in enumerate(self.experts):
                ep = e(x)
                y[:, i, :] = ep
            y = torch.einsum('bij,bi->bj', y, weights)  # y: [B, E]
        else:
            # y: [B, L, experts, E]
            y = torch.empty(x.shape[0], x.shape[1], len(self.experts), self.out_dim).to(self.device)
            for i, e in enumerate(self.experts):
                ep = e(x)  # [B, L, E]
                y[:, :, i, :] = ep
            y = torch.einsum('bijk,bij->bk', y, weights)  # y: [B, E]
        return y


'''
The snippet below defines AudioResNet
Adapted from https://pytorch.org/vision/main/_modules/torchvision/models/resnet.html
Pretty much the same old resnet, but modified to fit clap audio representation (1d conv).
'''


def conv3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv1d:
    """3 convolution with padding"""
    return nn.Conv1d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation
    )


def conv1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv1d:
    """1 convolution"""
    return nn.Conv1d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition" https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
            self,
            inplanes: int,
            planes: int,
            stride: int = 1,
            downsample: Optional[nn.Module] = None,
            groups: int = 1,
            base_width: int = 64,
            dilation: int = 1,
            norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm1d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class AudioResNet(nn.Module):
    def __init__(
            self,
            layers: List[int],
            planes: List[int],
            in_ch: int = 1,
            out_dim: int = 1024,
            zero_init_residual: bool = False,
            groups: int = 1,
            width_per_group: int = 64,
            norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm1d
        self._norm_layer = norm_layer

        self.inplanes = planes[0]
        self.dilation = 1

        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv1d(in_ch, self.inplanes, kernel_size=9, stride=4, padding=4, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(planes[0], layers[0])
        self.layers = nn.Sequential()
        for i in range(1, len(layers)):
            self.layers.append(self._make_layer(planes[i], layers[1], stride=2, dilate=False))
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(planes[-1] * Bottleneck.expansion, out_dim)

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm1d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if m.bn3.weight is not None:
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]

    def _make_layer(
            self,
            planes: int,
            blocks: int,
            stride: int = 1,
            dilate: bool = False,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * Bottleneck.expansion:
            downsample = nn.Sequential(
                conv1(self.inplanes, planes * Bottleneck.expansion, stride),
                norm_layer(planes * Bottleneck.expansion),
            )

        layers = []
        layers.append(
            Bottleneck(
                self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer
            )
        )
        self.inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(
                Bottleneck(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layers(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)

