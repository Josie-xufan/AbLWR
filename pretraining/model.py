from typing import Any, Dict, List, Optional, Tuple
import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.data import Batch as PygBatch
from torch_geometric.nn import GCNConv
from torch_geometric.nn import Sequential as PyGSequential
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool
from torch.autograd import Variable

def to_var(x, requires_grad=True):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, requires_grad=requires_grad)


class MetaModule(nn.Module):
    def params(self):
        for name, param in self.named_params(self):
            yield param

    def named_leaves(self):
        return []

    def named_submodules(self):
        return []

    def named_params(self, curr_module=None, memo=None, prefix=""):
        if memo is None:
            memo = set()

        if hasattr(curr_module, "named_leaves"):
            for name, p in curr_module.named_leaves():
                if p is not None and p not in memo:
                    memo.add(p)
                    yield prefix + ("." if prefix else "") + name, p
        else:
            for name, p in curr_module._parameters.items():
                if p is not None and p not in memo:
                    memo.add(p)
                    yield prefix + ("." if prefix else "") + name, p

        for mname, module in curr_module.named_children():
            submodule_prefix = prefix + ("." if prefix else "") + mname
            for name, p in self.named_params(module, memo, submodule_prefix):
                yield name, p

    def update_params(
        self,
        lr_inner,
        first_order=False,
        source_params=None,
        detach=False,
        identifier=None,
    ):
        if source_params is not None:
            if identifier is None:
                named_params = self.named_params(self)
            else:
                named_params = []
                for name, p in self.named_params(self):
                    if (identifier in name) and len(p.shape) > 1:
                        named_params.append((name, p))
            for tgt, src in zip(named_params, source_params):
                name_t, param_t = tgt
                grad = src
                if first_order:
                    grad = to_var(grad.detach().data)
                if grad is not None:
                    tmp = param_t - lr_inner * grad
                    self.set_param(self, name_t, tmp)
        else:
            for name, param in self.named_params(self):
                if not detach:
                    grad = param.grad
                    if first_order:
                        grad = to_var(grad.detach().data)
                    tmp = param - lr_inner * grad
                    self.set_param(self, name, tmp)
                else:
                    param = param.detach_()
                    self.set_param(self, name, param)

    def set_param(self, curr_mod, name, param):
        if "." in name:
            n = name.split(".")
            module_name = n[0]
            rest = ".".join(n[1:])
            for name, mod in curr_mod.named_children():
                if module_name == name:
                    self.set_param(mod, rest, param)
                    break
        else:
            setattr(curr_mod, name, param)

    def detach_params(self):
        for name, param in self.named_params(self):
            self.set_param(self, name, param.detach())

    def copy(self, other, same_var=False):
        for name, param in other.named_params():
            if not same_var:
                param = to_var(param.data.clone(), requires_grad=True)
            self.set_param(name, param)


class MetaLinear(MetaModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        ignore = nn.Linear(*args, **kwargs)

        self.register_buffer("weight", to_var(ignore.weight.data, requires_grad=True))
        self.register_buffer("bias", to_var(ignore.bias.data, requires_grad=True))

    def forward(self, x):
        return F.linear(x, self.weight, self.bias)

    def named_leaves(self):
        return [("weight", self.weight), ("bias", self.bias)]
    

class RegressionGCNAbAgIntLM(MetaModule):
    def __init__(self, cfg):
        super().__init__()
        self.B_encoder_block = self.create_encoder_block(**cfg["encoder"]["ab"])
        self.G_encoder_block = self.create_encoder_block(**cfg["encoder"]["ag"])
        self.pooling = global_mean_pool

        self.num_classes = cfg.get("num_classes", 3)
        self.input_dim = cfg["encoder"]["ab"]["dim_list"][-1] + cfg["encoder"]["ag"]["dim_list"][-1]
        self.feat_dim = cfg.get("feat_dim", 64)

        self.classifier = MetaLinear(self.input_dim, self.num_classes)
        self.fc4 = MetaLinear(self.input_dim, self.input_dim)
        self.fc5 = MetaLinear(self.input_dim, self.feat_dim)
        self.head = nn.Sequential(self.fc4, nn.ReLU(), self.fc5)

    
    def create_encoder_block(
        self,
        node_feat_name: str,
        edge_index_name: str,
        input_dim: int,
        input_act: str,
        dim_list: List[int],
        act_list: List[str],
        gcn_kwargs: Dict[str, Any],
    ):
        def _create_gcn_layer(i: int, j: int, in_channels: int, out_channels: int) -> Tuple[GCNConv, str]:
            """
            Generate a GCN layer

            Args:
                i (int): input layer index
                j (int): output layer index
                in_channels (int): input channels
                out_channels (int): output channels

            Returns:
                GCNConv: GCN layer
                str: a string to map the input args to the output, e.g. "x_b,
                edge_index_b -> x_b_1" means use node features `x_b` and edge_index
                `edge_index_b` to derive updated node features named `x_b_1`. This
                is used in Sequential module to map the input args to the output of
                the layer. See the reference for more details.
                Ref: https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html?highlight=sequential#torch_geometric.nn.sequential.Sequential
            """
            if i == 0:
                mapping = f"{node_feat_name}, {edge_index_name} -> {node_feat_name}_{j}"
            else:
                mapping = (f"{node_feat_name}_{i}, {edge_index_name} -> {node_feat_name}_{j}")

            return (GCNConv(in_channels=in_channels, out_channels=out_channels, **gcn_kwargs,),mapping,)

        def _create_act_layer(act_name: Optional[str]) -> nn.Module:
            # assert act_name is either None or str
            assert act_name is None or isinstance(act_name, str), f"act_name must be None or str, got {act_name}"

            if act_name is None:
                # return identity
                return (nn.Identity(),)
            elif act_name.lower() == "relu":
                return (nn.ReLU(inplace=True),)
            elif act_name.lower() == "leakyrelu":
                return (nn.LeakyReLU(inplace=True),)
            else:
                raise ValueError(f"activation {act_name} not supported, please choose from ['relu', 'leakyrelu', None]")

        modules = [_create_gcn_layer(0, 1, input_dim, dim_list[0]), _create_act_layer(input_act)]

        for i in range(len(dim_list) - 1):
            modules.extend(
                [
                    _create_gcn_layer(
                        i + 1, i + 2, dim_list[i], dim_list[i + 1]
                    ),  # i+1 increment due to the input layer
                    _create_act_layer(act_list[i]),
                ]
            )

        return PyGSequential(
            input_args=f"{node_feat_name}, {edge_index_name}", modules=modules
        )



    def encode(self, ab_batch: PygBatch, ag_batch: PygBatch) -> Tuple[Tensor, Tensor]:
        B_z = self.B_encoder_block(ab_batch.x_b, ab_batch.edge_index_b)
        G_z = self.G_encoder_block(ag_batch.x_g, ag_batch.edge_index_g)
        
        h_b = self.pooling(B_z, ab_batch.x_b_batch)
        h_g = self.pooling(G_z, ag_batch.x_g_batch)
        x = torch.cat([h_b, h_g], dim=1)
        return x
    

    def forward(self, ab_batch, ag_batch, flag_feature=False):
        emb = self.encode(ab_batch, ag_batch)

        y = self.classifier(emb)
        feat_cl = F.normalize(self.head(emb), dim=1)
        if flag_feature:
            return y, feat_cl
        else:
            return y