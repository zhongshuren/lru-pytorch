import math
import torch
import torch.nn as nn
from torch import rand, randn, complex, log, exp, abs, sqrt, stack, split
import torch.nn.functional as F


class LRU(nn.Module):
    def __init__(self, d_model, d_hidden, r_min=0, r_max=1, dropout=0.0):
        super(LRU, self).__init__()
        B = complex(randn(d_hidden, d_model), randn(d_hidden, d_model))
        C = complex(randn(d_model, d_hidden), randn(d_model, d_hidden))
        self.B = nn.Parameter(B / math.sqrt(2 * d_model))
        self.C = nn.Parameter(C / math.sqrt(d_hidden))
        self.D = nn.Parameter(randn(d_model) / math.sqrt(d_model))

        nu_log = log(-0.5 * log(rand(d_hidden) * (r_max - r_min) + r_min))
        th_log = log(rand(d_hidden) * 6.28)
        gma_log = log(sqrt(1 - abs(exp(complex(-exp(nu_log), exp(th_log)))) ** 2))
        self.params_log = nn.Parameter(stack([nu_log, th_log, gma_log]).unsqueeze(1))

        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def scan(x, h0, lmb):
        h = x if h0 is None else torch.cat([h0, x])
        l_p, p = lmb, 1
        while p < h.shape[0]:
            h1, h2, h3 = h[:p], h[p:], h[:-p]
            h = torch.cat([h1, h2 + h3 * l_p])
            l_p, p = l_p * l_p, p * 2
        return h if h0 is None else h[1:]

    def forward(self, inp, h0=None):
        nu, th, gma = split(exp(self.params_log), 1)
        lmb = exp(complex(-nu, th))

        x = self.layer_norm(inp)  # pre norm
        x = gma * F.linear(x.cfloat(), self.B)
        h = self.scan(x, h0, lmb)
        x = F.gelu(F.linear(h, self.C).real)

        return self.dropout(x) + inp * self.D


class StackedLRU(nn.Module):
    def __init__(self, args, input_dim, out_dim, multidim=1):
        super(StackedLRU, self).__init__()
        self.encoder = nn.Linear(input_dim, args.d_model)
        self.lru_layers = nn.Sequential(*(LRU(
            d_model=args.d_model, d_hidden=args.d_hidden,
            r_min=args.r_min, r_max=args.r_max, dropout=args.p_dropout
        ) for _ in range(args.n_layers)))
        self.head = nn.Linear(args.d_model, out_dim * multidim)
        self.multidim = multidim

    def forward(self, x):
        x = self.encoder(x)
        x = x.transpose(0, 1)
        x = self.lru_layers(x)
        x = x.transpose(0, 1)
        out = self.head(x)
        if self.multidim > 1:
            out = out.unflatten(-1, (-1, 2))
        return out
