import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from typing import Optional
from omegaconf import DictConfig
from torch_geometric.data import Batch as PygBatch
import math
from torch_geometric.nn import global_mean_pool

class LayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-8):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
        x_hat = (x - mean) / torch.sqrt(var + self.eps)
        return self.gamma * x_hat + self.beta

class MultiHeadAttention(nn.Module):
    def __init__(self,
                 num_units: int,
                 num_heads: int = 8,
                 dropout_rate: float = 0.0,
                 activation=F.relu,
                 use_layernorm: bool = True):
        super().__init__()
        self.num_units = num_units
        self.num_heads = num_heads
        self.activation = activation
        self.q_lin = nn.Linear(num_units, num_units, bias=True)
        self.k_lin = nn.Linear(num_units, num_units, bias=True)
        self.v_lin = nn.Linear(num_units, num_units, bias=True)
        self.attn = nn.MultiheadAttention(num_units, num_heads, dropout=dropout_rate, batch_first=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.ln = LayerNorm(num_units) if use_layernorm else nn.Identity()

    def forward(self,
                queries: torch.Tensor,
                keys: torch.Tensor,
                query_masks: Optional[torch.Tensor] = None,
                key_masks: Optional[torch.Tensor] = None,
                causality: bool = False,
                cloze: bool = False,
                is_training: bool = True,
                debug: bool = False):

        Q = self.activation(self.q_lin(queries))
        K = self.activation(self.k_lin(keys))
        V = self.activation(self.v_lin(keys))

        B, Tq, _ = Q.shape
        Tk = K.shape[1]

        key_padding_mask = None
        if key_masks is not None:
            key_padding_mask = (key_masks == 0)
        attn_mask = None
        if causality:
            causal = torch.triu(torch.ones(Tq, Tk, device=Q.device), diagonal=1).bool()
            attn_mask = causal
        if cloze:
            rq = torch.arange(Tq, device=Q.device).unsqueeze(1)
            rk = torch.arange(Tk, device=Q.device).unsqueeze(0)
            diff = rq - rk  # [Tq,Tk]
            cloze_mask = (diff == -1)
            attn_mask = cloze_mask if attn_mask is None else (attn_mask | cloze_mask)

        if attn_mask is not None and attn_mask.dtype != torch.bool:
            attn_mask = attn_mask.bool()

        attn_output, attn_weights = self.attn(Q, K, V,
                                              key_padding_mask=key_padding_mask,
                                              attn_mask=attn_mask)
        attn_output = self.dropout(attn_output)
        out = self.ln(attn_output + queries)

        debug_dict = {}
        if debug:
            debug_dict = {
                "Q": Q, "K": K, "V": V,
                "attn_weights": attn_weights
            }
        return (out, debug_dict) if debug else out


class InducedMultiHeadAttention(nn.Module):
    def __init__(self,
                 num_units,
                 num_heads=8,
                 num_induced=10,
                 dropout_rate=0.0,
                 activation=F.relu):
        super().__init__()
        self.num_induced = num_induced
        self.inducing = nn.Parameter(torch.randn(num_induced, num_units) * 0.1)
        self.attn1 = MultiHeadAttention(num_units, num_heads, dropout_rate, activation)
        self.attn2 = MultiHeadAttention(num_units, num_heads, dropout_rate, activation)

    def forward(self,
                queries,
                keys,
                query_masks=None,
                key_masks=None,
                causality=False,
                cloze=False):
        B = queries.size(0)
        I = self.inducing.unsqueeze(0).expand(B, -1, -1)
        H = self.attn1(I, keys, key_masks=key_masks, causality=False, cloze=False)
        O = self.attn2(queries, H, query_masks=query_masks, key_masks=None, causality=causality, cloze=cloze)
        return O
    
class FFNBlock(nn.Module):
    def __init__(self, hidden, dropout, activation=F.relu):
        super().__init__()
        self.lin1 = nn.Linear(hidden, 4*hidden)
        self.lin2 = nn.Linear(4*hidden, hidden)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(hidden)

        self.act = activation
    def forward(self, x):
        residual = x
        y = self.act(self.lin1(x))
        y = self.lin2(y)
        y = self.dropout(y)
        return self.norm1(residual + y)
    
class GraphRegressor(nn.Module):
    def __init__(self, cfg: DictConfig):
        super().__init__()

        self.hidden_units = cfg["hidden_units"]
        self.embed_size = cfg["embed_size"]
        self.dropout_rate = cfg["dropout_rate"]
        self.num_blocks = cfg["num_blocks"]
        self.num_heads = cfg["num_heads"]
        self.num_induced = cfg["num_induced"]

        self.abstract_layers = nn.ModuleList()
        self.abstract_layers.append(nn.Linear(self.embed_size, 4 * self.hidden_units))
        self.abstract_layers.append(nn.Linear(4 * self.hidden_units, self.hidden_units))
        self.abstract_norm = nn.LayerNorm(self.hidden_units)

        self.dropout = nn.Dropout(self.dropout_rate)
        self.activation_fn = F.relu

        self.blocks = nn.ModuleList()
        for _ in range(self.num_blocks):
            self.blocks.append(InducedMultiHeadAttention(num_units=self.hidden_units,
                                            num_heads=self.num_heads,
                                            num_induced=self.num_induced,
                                            dropout_rate=self.dropout_rate,
                                            activation=self.activation_fn))

        self.ffns = nn.ModuleList([FFNBlock(self.hidden_units, self.dropout_rate, self.activation_fn)for _ in range(self.num_blocks)])
        self.out_linear = nn.Linear(self.hidden_units, 1)

    def forward(self, emb: Tensor) -> Tensor:
        x = emb
        x = self.activation_fn(self.abstract_layers[0](x))
        x = self.activation_fn(self.abstract_layers[1](x))
        x = self.abstract_norm(x)
        x = self.dropout(x)


        for i in range(self.num_blocks):
            x = self.blocks[i](queries=x, keys=x)
            x = self.ffns[i](x)

        logits = self.out_linear(x).squeeze(-1)  # [B,T]
        return logits
    





# ================================= Regression decoder with local features ========================================== #
class Regressor_Local_Feature(nn.Module):
    def __init__(self):
        super().__init__()
        self.pooling = global_mean_pool
        self.fusion = define_trifusion(fusion_type="trifusion", skip=True, use_bilinear=True, 
                                gate1=True, gate2=True, gate3=True, 
                                dim1=128, dim2=128, dim3=128,
                                scale_dim1=2, scale_dim2=2, scale_dim3=2,
                                mmhid=128, dropout_rate=0.25)
        self.regressor = nn.LazyLinear(out_features=1)

    def forward(self, B_z: Tensor, G_z: Tensor, batch: PygBatch) -> Tensor:
        """
        Args:
            B_z: (Tensor) shape (Nb, C)
            G_z: (Tensor) shape (Ng, C)
            batch: (PygBatch) batched data returned by PyG DataLoader
        Returns:
            affinity_pred: (Tensor) shape (B, 1)
        """
        assert not torch.all(batch.lfh == -1, dim=1).any().item()
        assert not torch.all(batch.lfp == -1, dim=1).any().item()

        h_b = self.pooling(B_z, batch.x_b_batch)  # shape (B, C), doesn't change the embedding dimension
        h_g = self.pooling(G_z, batch.x_g_batch)  # shape (B, C), doesn't change the embedding dimension
        x = torch.cat([h_b, h_g], dim=1)  # shape (B, C*2)

        lfp = batch.lfp
        lfh = batch.lfh
        fused_h = self.fusion(x, lfp, lfh)

        return fused_h
def init_max_weights(module):
    for m in module.modules():
        if type(m) == nn.Linear:
            stdv = 1. / math.sqrt(m.weight.size(1))
            m.weight.data.normal_(0, stdv)
            m.bias.data.zero_()
def define_trifusion(fusion_type, skip=True, use_bilinear=True, 
                     gate1=True, gate2=True, gate3=True, gated_fusion=True,
                     dim1=32, dim2=32, dim3=32,
                     scale_dim1=1, scale_dim2=1, scale_dim3=1,
                     mmhid=64, dropout_rate=0.25):
    fusion = None
    if fusion_type == 'trifusion':
        fusion = TripleBilinearFusion(skip=skip, use_bilinear=use_bilinear, 
                                      gate1=gate1, gate2=gate2, gate3=gate3, gated_fusion=gated_fusion,
                                      dim1=dim1, dim2=dim2, dim3=dim3,
                                      scale_dim1=scale_dim1, scale_dim2=scale_dim2, scale_dim3=scale_dim3,
                                      mmhid=mmhid, dropout_rate=dropout_rate)
    else:
        raise NotImplementedError('fusion type [%s] is not found' % fusion_type)
    return fusion
class TripleBilinearFusion(nn.Module):
    def __init__(self, skip=True, use_bilinear=True, 
                 gate1=True, gate2=True, gate3=True, gated_fusion=True,
                 dim1=32, dim2=32, dim3=32,
                 scale_dim1=1, scale_dim2=1, scale_dim3=1,
                 mmhid=64, dropout_rate=0.25):
        super(TripleBilinearFusion, self).__init__()
        self.skip = skip
        self.use_bilinear = use_bilinear
        self.gate1 = gate1
        self.gate2 = gate2
        self.gate3 = gate3
        self.gated_fusion = gated_fusion

        dim1_og, dim2_og, dim3_og = dim1, dim2, dim3
        dim1, dim2, dim3 = int(dim1/scale_dim1), int(dim2/scale_dim2), int(dim3/scale_dim3)

        skip_dim = dim1 + dim2 + dim3 + 3 if skip else 0

        # Feature 1
        self.linear_h1 = nn.Sequential(nn.Linear(dim1_og, dim1), nn.ReLU())
        self.linear_z1 = nn.Bilinear(dim1_og, dim2_og, dim1) if use_bilinear else nn.Sequential(nn.Linear(dim1_og + dim2_og, dim1))
        self.linear_o1 = nn.Sequential(nn.Linear(dim1, dim1), nn.ReLU(), nn.Dropout(p=dropout_rate))

        # Feature 2
        self.linear_h2 = nn.Sequential(nn.Linear(dim2_og, dim2), nn.ReLU())
        self.linear_z2 = nn.Bilinear(dim2_og, dim3_og, dim2) if use_bilinear else nn.Sequential(nn.Linear(dim2_og + dim3_og, dim2))
        self.linear_o2 = nn.Sequential(nn.Linear(dim2, dim2), nn.ReLU(), nn.Dropout(p=dropout_rate))

        # Feature 3
        self.linear_h3 = nn.Sequential(nn.Linear(dim3_og, dim3), nn.ReLU())
        self.linear_z3 = nn.Bilinear(dim1_og, dim3_og, dim3) if use_bilinear else nn.Sequential(nn.Linear(dim1_og + dim3_og, dim3))
        self.linear_o3 = nn.Sequential(nn.Linear(dim3, dim3), nn.ReLU(), nn.Dropout(p=dropout_rate))

        # Fusion layers
        self.post_fusion_dropout = nn.Dropout(p=dropout_rate)
        self.encoder1 = nn.Sequential(nn.Linear((dim1 + 1) * (dim2 + 1) * (dim3 + 1), mmhid), nn.ReLU(), nn.Dropout(p=dropout_rate))
        self.encoder2 = nn.Sequential(nn.Linear(mmhid + skip_dim, mmhid), nn.ReLU(), nn.Dropout(p=dropout_rate))
        self.encoder3 = nn.Sequential(nn.Linear(dim1 + dim2 + dim3, mmhid), nn.ReLU(), nn.Dropout(p=dropout_rate))

        init_max_weights(self)

    def forward(self, vec1, vec2, vec3):
        device = vec1.device

        ### Feature 1
        if self.gate1:
            h1 = self.linear_h1(vec1)
            z1 = self.linear_z1(vec1, vec2) if self.use_bilinear else self.linear_z1(torch.cat((vec1, vec2), dim=1))
            o1 = self.linear_o1(nn.Sigmoid()(z1) * h1)
        else:
            o1 = self.linear_h1(vec1)

        ### Feature 2
        if self.gate2:
            h2 = self.linear_h2(vec2)
            z2 = self.linear_z2(vec2, vec3) if self.use_bilinear else self.linear_z2(torch.cat((vec2, vec3), dim=1))
            o2 = self.linear_o2(nn.Sigmoid()(z2) * h2)
        else:
            o2 = self.linear_h2(vec2)

        ### Feature 3
        if self.gate3:
            h3 = self.linear_h3(vec3)
            z3 = self.linear_z3(vec1, vec3) if self.use_bilinear else self.linear_z3(torch.cat((vec1, vec3), dim=1))
            o3 = self.linear_o3(nn.Sigmoid()(z3) * h3)
        else:
            o3 = self.linear_h3(vec3)

        ### Fusion
        if self.gated_fusion:
            o1 = torch.cat((o1, torch.FloatTensor(o1.shape[0], 1).fill_(1).to(device)), 1)
            o2 = torch.cat((o2, torch.FloatTensor(o2.shape[0], 1).fill_(1).to(device)), 1)
            o3 = torch.cat((o3, torch.FloatTensor(o3.shape[0], 1).fill_(1).to(device)), 1)

            o123 = torch.bmm(o1.unsqueeze(2), o2.unsqueeze(1))  # BATCH_SIZE x dim1 x dim2
            o123 = o123.unsqueeze(3) @ o3.unsqueeze(1).unsqueeze(2)  # BATCH_SIZE x dim1 x dim2 x dim3
            o123 = o123.flatten(start_dim=1)  # Flatten to BATCH_SIZE x (dim1*dim2*dim3)

            out = self.post_fusion_dropout(o123)
            out = self.encoder1(out)
            if self.skip: out = torch.cat((out, o1, o2, o3), 1)
            out = self.encoder2(out)
        else:
            out = torch.cat((o1, o2, o3), 1)
            out = self.encoder3(out)
        return out
