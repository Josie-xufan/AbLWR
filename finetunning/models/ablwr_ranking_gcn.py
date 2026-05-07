from typing import Any, Dict, List, Optional, Tuple
import seaborn as sns
import torch
import torch.nn as nn
from loguru import logger
from omegaconf import DictConfig
from torch import Tensor
from torch_geometric.data import Batch as PygBatch
from torch_geometric.nn import GCNConv
from torch_geometric.nn import Sequential as PyGSequential
from math import log
from itertools import combinations
from torch_geometric.nn import global_mean_pool
from finetunning.models.components.graph_regressor import GraphRegressor, Regressor_Local_Feature

sns.set_theme(style="whitegrid")
sns.set_context("paper", font_scale=1.5)
sns.set_palette("Set2")



class RankingCompositeLoss(nn.Module):
    def __init__(self, slate_length=5, padded_value_indicator=100,
                 use_listwise = True, 
                 use_pos_weight_norm=True,
                 list_weight=1.0,
                 tie_noise_std=1e-4):
        super().__init__()
        self.slate_length = slate_length
        self.padded_value_indicator = padded_value_indicator


        self.use_listwise = use_listwise
        self.use_pos_weight_norm = use_pos_weight_norm
        self.tie_noise_std = tie_noise_std
        self.list_weight = list_weight


    def _mask_padding(self, y_true):
        return (y_true == self.padded_value_indicator)

    def _enumerate_pairs(self, L):
        return list(combinations(range(L), 2))

    def _compute_pair_full_acc(self, y_pred, y_true):
        B, L = y_pred.shape
        pairs = self._enumerate_pairs(L)
        pad_mask = self._mask_padding(y_true)        


        correct_pairs = 0
        valid_pairs = 0

        for (i, j) in pairs:
            t_i = y_true[:, i]
            t_j = y_true[:, j]
            s_i = y_pred[:, i]
            s_j = y_pred[:, j]
            pad_any = pad_mask[:, i] | pad_mask[:, j]

            valid = (~pad_any)
            if not valid.any():
                continue
            
            sign = torch.sign((t_i - t_j)[valid])  # +1 or -1
            pred_gap = (s_i - s_j)[valid]

            correct_pairs += (sign * pred_gap > 0).float().sum()
            valid_pairs += valid.sum()

        pair_acc = correct_pairs / (valid_pairs + 1e-8)

        pred_order = torch.argsort(y_pred, dim=1, descending=True)
        true_order = torch.argsort(y_true, dim=1, descending=True)
        correct = (pred_order == true_order).all(dim=1).float()
        full_acc = correct.mean()
        return {"pair_acc": pair_acc.detach(), "full_acc": full_acc.detach()}


    def listmle_loss(self, y_pred, y_true, eps=1e-10):
        """
        Weighted ListMLE:
          1) Add tiny noise to y_true (only where not padding) to break ties per row
          2) Sort by (y_true + noise) descending
          3) Standard ListMLE with optional position weights (DCG discount)
        """
        y_pred = -y_pred
        y_true = -y_true


        B, L = y_pred.shape
        pad_mask = self._mask_padding(y_true)

        if self.tie_noise_std > 0:
            noise = torch.zeros_like(y_true).normal_(mean=0.0, std=self.tie_noise_std)
            noise = noise.masked_fill(pad_mask, 0.0)
        else:
            noise = torch.zeros_like(y_true)
        noisy_true = y_true.clone().masked_fill(pad_mask, float("-inf")) + noise


        sorted_true, indices = noisy_true.sort(dim=1, descending=True)
        preds_sorted = torch.gather(y_pred, dim=1, index=indices)
        preds_sorted = preds_sorted.masked_fill(pad_mask.gather(1, indices), float("-inf"))
        shifted = preds_sorted
        log_cumsumexp_rev = torch.logcumsumexp(shifted.flip(dims=[1]), dim=1).flip(dims=[1])
        observation_loss = (log_cumsumexp_rev - shifted)
        observation_loss = observation_loss.masked_fill(torch.isinf(shifted), 0.0)


        ranks = torch.arange(1, L + 1, device=y_pred.device, dtype=y_pred.dtype)
        pos_weights = 1.0 / torch.log2(ranks + 1.0)  # [L]
        pos_weights = pos_weights.unsqueeze(0).expand(B, L)
        valid_mask_sorted = ~pad_mask.gather(1, indices)
        pos_weights = pos_weights * valid_mask_sorted.to(y_pred.dtype)

        if self.use_pos_weight_norm:
            denom = pos_weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
            pos_weights = pos_weights / denom

        loss_per_row = (observation_loss * pos_weights).sum(dim=1)
        return loss_per_row.mean()

    def forward(self, y_pred, y_true):
        """
        y_pred: [B,L]
        y_true: [B,L]
        """
        assert y_pred.shape == y_true.shape
        logs = {}

        # ListMLE
        if self.use_listwise: l_list = self.listmle_loss(y_pred, y_true)
        else: l_list = y_pred.new_tensor(0.)
        logs["listmle_loss"] = l_list.detach()

        total = (self.list_weight * l_list)
        acc_info = self._compute_pair_full_acc(y_pred, y_true)
        logs.update(acc_info)
        return total, logs


class RegressionGCNAbAgIntLM(nn.Module):
    def __init__(self, cfg: DictConfig):
        """
        Args:
            cfg: (DictConfig) configuration
        The configuration should contain the following keys:
        - encoder: (DictConfig) encoder configuration
        - regressor: (DictConfig) regressor configuration
        - loss_func: (DictConfig) loss function configuration
        """
        super().__init__()
        self.config = cfg


        # store the loss for each batch in an epoch
        self.training_loss_epoch = []
        self.training_pred_values_epoch = []
        self.training_true_values_epoch = []


        self.validation_loss_epoch = []
        self.validation_pred_values_epoch = []
        self.validation_true_values_epoch = []

        self.test_loss_epoch = []
        self.test_pred_values_epoch = []
        self.test_true_values_epoch = []

        self.dbid_epoch = []


        self.B_encoder_block = self.create_encoder_block(**cfg["encoder"]["ab"])
        self.G_encoder_block = self.create_encoder_block(**cfg["encoder"]["ag"])
        self.pooling = global_mean_pool


        self.regressor = GraphRegressor(self.config["regression"])
        self.regressor_local_feature = Regressor_Local_Feature()
        self.loss_func_dict = self.configure_loss_func_dict()
        self.metric_func_dict = self.configure_metric_func_dict()

    

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



    def configure_loss_func_dict(self):
        loss_cfg_flat = {}
        for sub in self.config["loss_fun"].values():
            loss_cfg_flat.update(sub)
        return {"combine_loss": RankingCompositeLoss(**loss_cfg_flat)}

    def configure_metric_func_dict(self):
        """
        Configure the metric function dictionary
        """

        def _accuracy(pred_rank_label: Tensor, true_rank_label: Tensor) -> Tensor:
            return (pred_rank_label == true_rank_label).float().mean()

        def _full_rank_accuracy(y_pred: Tensor, y_true: Tensor) -> Tensor:
            """
            Compute the full rank accuracy, i.e. the predicted rank matches the true rank
            """
            assert y_pred.shape == y_true.shape
            pred_order = torch.argsort(y_pred, dim=1, descending=True)
            true_order = torch.argsort(y_true, dim=1, descending=True)
            correct = (pred_order == true_order).all(dim=1).float()
            return correct.mean()

        def _pairwise_rank_accuracy(y_pred: Tensor, y_true: Tensor) -> Tensor:
            B, L = y_pred.shape
            pairs = list(combinations(range(L), 2))
            pad_mask = (y_true == 100)

            correct_pairs = 0
            valid_pairs = 0

            for (i, j) in pairs:
                t_i = y_true[:, i]
                t_j = y_true[:, j]
                s_i = y_pred[:, i]
                s_j = y_pred[:, j]
                pad_any = pad_mask[:, i] | pad_mask[:, j]

                delta_t = t_i - t_j
                valid = ~pad_any
                if not valid.any():
                    continue

                sign = torch.sign(delta_t[valid])  # +1 or -1
                pred_gap = (s_i - s_j)[valid]

                correct_pairs += (sign * pred_gap > 0).float().sum()
                valid_pairs += valid.sum()

            pair_acc = correct_pairs / valid_pairs
            return pair_acc
        
        return {"accuracy": _accuracy, "full_rank_accuracy": _full_rank_accuracy, "pairwise_rank_accuracy": _pairwise_rank_accuracy}


    def compute_loss(self, y_pred: Tensor, y_truth: Tensor, stage: str) -> Dict[str, Tensor]:
        k = "combine_loss"
        combined, extra = self.loss_func_dict[k](y_pred, y_truth)

        loss_dict = {f"{stage}/loss/{k}": combined}
        loss_dict[f"{stage}/loss/listmle_loss"] = extra["listmle_loss"]

        loss_dict[f"{stage}/acc/pair_acc"] = extra["pair_acc"]
        loss_dict[f"{stage}/acc/full_acc"] = extra["full_acc"]
        return loss_dict


    def encode(self, batch: PygBatch) -> Tuple[Tensor, Tensor]:
        """
        Args:
            batch: (PygBatch) batched data returned by PyG DataLoader
        Returns:
            B_z: (Tensor) shape (Nb, C)
            G_z: (Tensor) shape (Ng, C)
        """
        B_z = self.B_encoder_block(batch.x_b, batch.edge_index_b)
        G_z = self.G_encoder_block(batch.x_g, batch.edge_index_g)
        

        h_b = self.pooling(B_z, batch.x_b_batch)
        h_g = self.pooling(G_z, batch.x_g_batch)
        x = torch.cat([h_b, h_g], dim=1)  
        return x
    

    def forward(self, batch1: PygBatch, batch2: PygBatch, batch3: PygBatch, batch4: PygBatch, batch5: PygBatch) -> Tensor:
        emb1 = self.encode(batch1)
        emb2 = self.encode(batch2)
        emb3 = self.encode(batch3)
        emb4 = self.encode(batch4)
        emb5 = self.encode(batch5)
        emb = torch.stack([emb1, emb2, emb3, emb4, emb5], dim=1)
        affinity_pred = self.regressor(emb)
        return affinity_pred

    

    def one_step(self, batch1: PygBatch, batch2: PygBatch, batch3: PygBatch, batch4: PygBatch, batch5: PygBatch, ranking_values: Tensor, stage: str) -> Tuple[Dict[str, Tensor], Tensor]:
        assert stage in ["train", "val", "test"], f"stage must be either 'train' or 'val', got {stage}"
        y_pred = self.forward(batch1, batch2, batch3, batch4, batch5)
        y_truth = ranking_values
        loss_dict = self.compute_loss(y_pred=y_pred, y_truth=y_truth, stage=stage)
        loss_values = loss_dict[f"{stage}/loss/combine_loss"]
        loss_dict["loss"] = loss_values
        return loss_dict, y_pred

    def training_step(self, batch, step) -> Tensor:
        loss_dict, value_pred = self.one_step(
            batch1=batch[0],
            batch2=batch[1],
            batch3=batch[2],
            batch4=batch[3],
            batch5=batch[4],
            ranking_values=batch[5],
            stage="train",
        )
        self.training_loss_epoch.append(loss_dict["loss"])
        self.training_pred_values_epoch.append(value_pred.squeeze(dim=1))  # (B,)
        self.training_true_values_epoch.append(batch[5].squeeze(dim=1))  # (B,)
        return loss_dict["loss"], loss_dict["train/acc/full_acc"], loss_dict["train/acc/pair_acc"]

    def validation_step(self, batch, step) -> Tensor:
        loss_dict, value_pred = self.one_step(
            batch1=batch[0],
            batch2=batch[1],
            batch3=batch[2],
            batch4=batch[3],
            batch5=batch[4],
            ranking_values=batch[5],
            stage="val",
        )


        self.validation_loss_epoch.append(loss_dict["loss"])
        self.validation_pred_values_epoch.append(value_pred.squeeze(dim=1))
        self.validation_true_values_epoch.append(batch[5].squeeze(dim=1))
        return loss_dict["loss"], loss_dict["val/acc/full_acc"], loss_dict["val/acc/pair_acc"]
    

    def test_step(self, batch, step) -> Tensor:
        loss_dict, value_pred = self.one_step(
            batch1=batch[0],
            batch2=batch[1],
            batch3=batch[2],
            batch4=batch[3],
            batch5=batch[4],
            ranking_values=batch[5],
            stage="test",
        )


        self.test_loss_epoch.append(loss_dict["loss"])
        self.test_pred_values_epoch.append(value_pred.squeeze(dim=1))
        self.test_true_values_epoch.append(batch[5].squeeze(dim=1))
        return loss_dict["loss"], loss_dict["test/acc/full_acc"], loss_dict["test/acc/pair_acc"]

    def pred_step(self, batch) -> Tensor:
        loss_dict, value_pred = self.one_step(
            batch1=batch[0],
            batch2=batch[1],
            batch3=batch[2],
            batch4=batch[3],
            batch5=batch[4],
            ranking_values=batch[5],
            stage="test",
        )


        self.test_loss_epoch.append(loss_dict["loss"])
        self.test_pred_values_epoch.append(value_pred.squeeze(dim=1))
        self.test_true_values_epoch.append(batch[5].squeeze(dim=1))

        self.dbid_epoch.append([item.abdbid.tolist() for item in batch[0:5]]) # 对于casestudy来说是：item.abdbid
        return loss_dict["loss"], loss_dict["test/acc/full_acc"], loss_dict["test/acc/pair_acc"]



    def on_train_epoch_end(self, epoch) -> None:
        avg_loss = torch.stack(self.training_loss_epoch).mean()
        pred_values = torch.cat(self.training_pred_values_epoch, dim=0)
        true_values = torch.cat(self.training_true_values_epoch, dim=0)


        full_rank_acc = self.metric_func_dict["full_rank_accuracy"](pred_values, true_values)
        pairwise_rank_acc = self.metric_func_dict["pairwise_rank_accuracy"](pred_values, true_values)


        self.training_loss_epoch.clear()
        self.training_pred_values_epoch.clear()
        self.training_true_values_epoch.clear()


        return avg_loss, full_rank_acc, pairwise_rank_acc, pred_values, true_values


    def on_validation_epoch_end(self, epoch) -> None:
        avg_loss = torch.stack(self.validation_loss_epoch).mean()
        pred_values = torch.cat(self.validation_pred_values_epoch, dim=0)
        true_values = torch.cat(self.validation_true_values_epoch, dim=0)


        full_rank_acc = self.metric_func_dict["full_rank_accuracy"](pred_values, true_values)
        pairwise_rank_acc = self.metric_func_dict["pairwise_rank_accuracy"](pred_values, true_values)


        self.validation_loss_epoch.clear()
        self.validation_pred_values_epoch.clear()
        self.validation_true_values_epoch.clear()


        pred_values_np = pred_values.clone().cpu().numpy()
        true_values_np = true_values.clone().cpu().numpy()


        return avg_loss, full_rank_acc, pairwise_rank_acc, pred_values_np, true_values_np



    def on_test_epoch_end(self, epoch) -> None:
        avg_loss = torch.stack(self.test_loss_epoch).mean()
        pred_values = torch.cat(self.test_pred_values_epoch, dim=0)
        true_values = torch.cat(self.test_true_values_epoch, dim=0)


        full_rank_acc = self.metric_func_dict["full_rank_accuracy"](pred_values, true_values)
        pairwise_rank_acc = self.metric_func_dict["pairwise_rank_accuracy"](pred_values, true_values)


        self.test_loss_epoch.clear()
        self.test_pred_values_epoch.clear()
        self.test_true_values_epoch.clear()


        pred_values_np = pred_values.clone().cpu().numpy()
        true_values_np = true_values.clone().cpu().numpy()

        return avg_loss, full_rank_acc, pairwise_rank_acc, pred_values_np, true_values_np
    

    def on_pred_epoch_end(self) -> None:
        avg_loss = torch.stack(self.test_loss_epoch).mean()
        pred_values = torch.cat(self.test_pred_values_epoch, dim=0)
        true_values = torch.cat(self.test_true_values_epoch, dim=0)


        full_rank_acc = self.metric_func_dict["full_rank_accuracy"](pred_values, true_values)
        pairwise_rank_acc = self.metric_func_dict["pairwise_rank_accuracy"](pred_values, true_values)


        self.test_loss_epoch.clear()
        self.test_pred_values_epoch.clear()
        self.test_true_values_epoch.clear()


        pred_values_np = pred_values.clone().cpu().numpy()
        true_values_np = true_values.clone().cpu().numpy()

        dbid0 = sum([item[0] for item in self.dbid_epoch], [])
        dbid1 = sum([item[1] for item in self.dbid_epoch], [])
        dbid2 = sum([item[2] for item in self.dbid_epoch], [])
        dbid3 = sum([item[3] for item in self.dbid_epoch], [])
        dbid4 = sum([item[4] for item in self.dbid_epoch], [])


        return avg_loss, full_rank_acc, pairwise_rank_acc, pred_values_np, true_values_np, dbid0, dbid1, dbid2, dbid3, dbid4
