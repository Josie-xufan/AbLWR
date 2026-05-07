import argparse
import os
import random
import time
import warnings
import torch
import torch.nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import numpy as np
import yaml
import json
from model import *
from utils.utils_algo import *
from dataset import *
from loss import *
import warnings
warnings.filterwarnings("ignore")
torch.set_printoptions(precision=2, sci_mode=False)

# ==============================================================================
# REFERENCE & ATTRIBUTION
# ==============================================================================
# This code is adapted/referenced from the following research works:
#
# Papers:
# 1. Pang, Liang, et al. "Setrank: Learning a permutation-invariant ranking model for information retrieval." 
#    Proceedings of the 43rd international ACM SIGIR conference on research and development in information retrieval. 2020.
#
# Source Repositories:
# - https://github.com/pl8787/SetRank
#
# License: 
# Apache-2.0
# (See: https://opensource.org/license/apache-2-0)
#
# Description: 
# This function is used to utilize unlabeled Ab-Ag pair to pretraining a robust representations based on PUL.
# ==============================================================================




parser = argparse.ArgumentParser(description="PU Pretraining for Ab-Ag Binding Affinity Classification")
parser.add_argument("--graph_root", default="..dataset/pairgraph", type=str, help="path to antigen or antibody graph data",)
parser.add_argument("--exp-dir",default="experiment",type=str,help="experiment directory for saving checkpoints and logs",)
parser.add_argument("--config", default="config.yaml", type=str, help="path to config file",)
parser.add_argument("--ab_col", default="Ab_seq_idx", type=str, help="column name of antibody IDs",)
parser.add_argument("--ag_col", default="Ag_seq_idx", type=str, help="column name of antigen IDs",)
parser.add_argument("--label_col", default="label", type=str, help="column name of label",)
parser.add_argument("--type_col", default="type", type=str, help="column name of data type",)
parser.add_argument("--ent_loss", action="store_true", help="whether enable entropy loss")
parser.add_argument("--using_cont", type=int, default=1, help="whether using contrastive loss")


parser.add_argument("--seed", default=42, type=int, help="seed for initializing training. ")
parser.add_argument("--gpu", default=0, type=int, help="GPU id to use.")
# 0-4: corresponding to the cv； >=5 means not using cv, using hard ag/ab split；-1 refers to need data splitting
parser.add_argument("--fold_idx", default=4, type=int, help="folds idx for cross validation ")  
parser.add_argument("--n_folds", default=5, type=int, help="folds for cross validation ")
parser.add_argument("--num_workers", default=4, type=int, help="number of workers for dataloader")
parser.add_argument("-b","--batch-size",default=64, type=int,help="mini-batch size (default: 64), this is the total ""batch size of all GPUs on the current node when ""using Data Parallel or Distributed Data Parallel",)
parser.add_argument("--epochs", default=400, type=int, help="number of total epochs to run")
parser.add_argument("--lr","--learning-rate",default=1e-3,type=float,metavar="LR",help="initial learning rate",dest="lr",)
parser.add_argument("--meta_lr","--meta-learning-rate",default=0.001,type=float,metavar="Meta_LR",help="meta learning rate",dest="meta_lr",)
parser.add_argument("--lr_decay_epochs",type=str,default="250,300,350",help="where to decay lr, can be a list",)
parser.add_argument("--lr_decay_rate", type=float, default=0.1, help="decay rate for learning rate")
parser.add_argument("--wd","--weight-decay",default=1e-4,type=float,metavar="W",help="weight decay (default: 1e-4)",dest="weight_decay",)
parser.add_argument("--momentum", default=0.9, type=float, metavar="M", help="momentum of SGD solver")
parser.add_argument("--rho_range", default="0.95,0.8", type=str, help="momentum updating parameter")
parser.add_argument("--temperature", default=0.07, type=float, help="mixup loss weight")
parser.add_argument("--mix_weight", default=1.0, type=float, help="mixup loss weight")


parser.add_argument("--warmup_epoch", default=20, type=int, help="epoch number of warm up") 
parser.add_argument("--contrastive_clustering",default=1,type=int,help="whether using contrastive clustering",)
parser.add_argument("--num_cluster", default=5, type=int, help="number of clusters")
parser.add_argument("--cont_cutoff", action="store_true", help="whether cut off by classifier")
parser.add_argument("--knn_aug", action="store_true", help="whether using kNN for CL")
parser.add_argument("--start_knn_aug", default=50, type=int, help="epoch to start kNN augmentation")
parser.add_argument("--num_neighbors", default=10, type=int, help="number of neighbors")
parser.add_argument("--identifier",default="classifier",type=str,help="identifier for meta layers, e.g. classifier",)


parser.add_argument("-p", "--print-freq", default=100, type=int, help="print frequency (default: 100)")
parser.add_argument("--tag", default="", type=str, help="special identifier")
parser.add_argument("--no_verbose", action="store_true", help="disable showing running statics")
parser.add_argument("--cosine", action="store_true", default=False, help="use cosine lr schedule")
parser.add_argument("--reverse", default=0, type=int, help="whether inverse label")


class Trainer:
    def __init__(self, args):
        self.args = args

        model_path = f"AbAg_PUL_ep{args.epochs}_wp{args.warmup_epoch}_rho{args.rho_start}~{args.rho_end}_co{args.cont_cutoff}_knn{args.knn_aug}{args.num_neighbors}_sd_{args.seed}"
        args.exp_dir = os.path.join(args.exp_dir, model_path)
        if not os.path.exists(args.exp_dir):
            os.makedirs(args.exp_dir)

        if args.seed is not None:
            random.seed(args.seed)
            torch.manual_seed(args.seed)
            np.random.seed(args.seed)
            cudnn.deterministic = True
        self.device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

        train_loader, valid_loader, test_loader, eval_loader, meta = load_dataset(args=args)
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        self.eval_loader = eval_loader
        self.meta = meta

        with open(args.config, 'r') as f:
            self.cfg = yaml.safe_load(f)
        self.model = RegressionGCNAbAgIntLM(self.cfg).to(self.device)
        self.optimizer = torch.optim.SGD(self.model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay,)

        self.classify_loss = CrossEntropyLossCustom(args.ent_loss)
        self.contrastive_loss = ContLoss(
            temperature=args.temperature,
            cont_cutoff=args.cont_cutoff,
            knn_aug=args.knn_aug,
            num_neighbors=args.num_neighbors,
            contrastive_clustering=args.contrastive_clustering,
            device=self.device
        )

    def train(self):
        args = self.args
        optimizer = self.optimizer

        best_test_topk_acc = 0
        best_val_topk_acc = 0

        for epoch in range(0, args.epochs):
            adjust_learning_rate(args, optimizer, epoch)

            if epoch < args.warmup_epoch or args.using_cont == 0:
                self.train_loop(epoch)
            else:
                features = self.compute_features(self.cfg)
                cluster_result = run_kmeans(features, args)
                self.train_loop(epoch, cluster_result)

            val_topk_acc, val_auc, val_f1, val_recall, val_precision = self.eval(self.valid_loader, "valid")
            test_topk_acc, test_auc, test_f1, test_recall, test_precision = self.eval(self.test_loader, "test")


            with open(os.path.join(args.exp_dir, "result.log"), "a+") as f:
                f.write("Epoch {}:\n".format(epoch))
                f.write(
                    "Validation - Top-K Acc: {}, AUC: {:.4f}, F1: {}, Recall: {}, Precision: {}\n".format(
                        val_topk_acc, val_auc, val_f1, val_recall, val_precision
                    )
                )

                f.write(
                    "Test - Top-K Acc: {}, AUC: {:.4f}, F1: {}, Recall: {}, Precision: {}\n".format(
                        test_topk_acc, test_auc, test_f1, test_recall, test_precision
                    )
                )

                f.write(
                    "Best - Test Acc {:.2f}. Learning Rate: {:.5f}\n\n".format(best_test_topk_acc, optimizer.param_groups[0]["lr"])
                    )

            if val_topk_acc > best_val_topk_acc:
                best_val_topk_acc = val_topk_acc
                ckpt_path = os.path.join(args.exp_dir, f"best_epoch{epoch:03d}.pt")
                torch.save(self.model.state_dict(), ckpt_path)

        args_dict = vars(args)
        with open(os.path.join(args.exp_dir, "args.json"), "w") as f:
            json.dump(args_dict, f, indent=4)

    def train_loop(self, epoch, cluster_result=None):
        args = self.args
        train_loader = self.train_loader
        model = self.model
        optimizer = self.optimizer
        classify_loss = self.classify_loss
        contrastive_loss = self.contrastive_loss

        batch_time = AverageMeter("Time", ":1.2f")
        data_time = AverageMeter("Data", ":1.2f")
        loss_cls_log = AverageMeter("Loss@Cls", ":2.2f")
        loss_cont_log = AverageMeter("Loss@Cont", ":2.2f")
        progress = ProgressMeter(len(train_loader), [batch_time, data_time, loss_cls_log, loss_cont_log], prefix="Epoch: [{}]".format(epoch))
        model.train()

        updated_label_list = []
        index_list = []
        ema_param = (1.0 * epoch / args.epochs * (args.rho_end - args.rho_start) + args.rho_start)

        end = time.time()

        for i, (ab_batch_orig, ag_batch_orig, ab_batch_w, ag_batch_w, ab_batch_s, ag_batch_s, label_batch, type_batch, idx_batch) in enumerate(train_loader):
            data_time.update(time.time() - end)

            if torch.argmax(label_batch, dim=1).sum() == 0:
                continue
            index_list.append(idx_batch)

            ab_batch_w, ag_batch_w, ab_batch_s, ag_batch_s, label_batch, idx_batch = (ab_batch_w.to(self.device), ag_batch_w.to(self.device), ab_batch_s.to(self.device), ag_batch_s.to(self.device), label_batch.to(self.device), idx_batch.to(self.device))
            bs = len(idx_batch)
            cluster_idxes = (None if cluster_result is None else cluster_result["im2cluster"][idx_batch])

            if epoch < args.warmup_epoch:
                labels_final = label_batch
            else:
                meta_model = RegressionGCNAbAgIntLM(self.cfg).to(self.device)
                meta_model.load_state_dict(model.state_dict())

                preds_meta = meta_model(ab_batch_w, ag_batch_w)
                eps = to_var(torch.zeros(bs, preds_meta.size(1)).to(self.device))
                labels_meta = label_batch.float() + eps
                loss = classify_loss(preds_meta, labels_meta)

                meta_model.zero_grad()
                params = []
                for name, p in meta_model.named_params(meta_model):
                    if args.identifier in name and len(p.shape) > 1:
                        params.append(p)
                grads = torch.autograd.grad(loss, params, create_graph=True, allow_unused=True)
                meta_model.update_params(args.meta_lr, source_params=grads, identifier=args.identifier)

                try:
                    ab_batch_orig_v, ag_batch_orig_v, label_batch_v, type_batch_v, idx_batch_v = next(valid_loder_iter)
                except:
                    valid_loder_iter = iter(self.valid_loader)
                    ab_batch_orig_v, ag_batch_orig_v, label_batch_v, type_batch_v, idx_batch_v = next(valid_loder_iter)
                ab_batch_orig_v, ag_batch_orig_v, label_batch_v, idx_batch_v = (ab_batch_orig_v.to(self.device), ag_batch_orig_v.to(self.device), label_batch_v.to(self.device), idx_batch_v.to(self.device))

                preds_v = meta_model(ab_batch_orig_v, ag_batch_orig_v)
                loss_meta_v = classify_loss(preds_v, label_batch_v.float())
                grad_eps = torch.autograd.grad(loss_meta_v, eps, only_inputs=True, allow_unused=True)[0]

                eps = eps - grad_eps
                meta_detected_labels = eps.argmax(dim=1)
                meta_detected_labels[torch.argmax(label_batch, dim=1).squeeze() == 1] = 1
                meta_detected_labels[torch.argmax(label_batch, dim=1).squeeze() == 2] = 2
                meta_detected_labels = F.one_hot(meta_detected_labels, eps.size(1))
                meta_detected_labels = meta_detected_labels.detach()

                updated_labels = label_batch.float()
                updated_labels = updated_labels * ema_param + meta_detected_labels * (1 - ema_param)
                labels_final = updated_labels.detach()
                updated_label_list.append(updated_labels.cpu())

                del grad_eps, grads, params

            preds_final, feat_cont = model(ab_batch_w, ag_batch_w, flag_feature=True)
            loss_cls = classify_loss(preds_final, labels_final)
            loss_cls_log.update(loss_cls.item())
            loss_final = loss_cls

            if args.using_cont:
                _, feat_cont_s = model(ab_batch_s, ag_batch_s, flag_feature=True)
                loss_cont = contrastive_loss(feat_cont, feat_cont_s, cluster_idxes, preds_final, start_knn_aug=epoch>args.start_knn_aug)
                loss_cont_log.update(loss_cont.item())
                loss_final = loss_final + loss_cont
            
            optimizer.zero_grad()
            loss_final.backward()
            optimizer.step()
            batch_time.update(time.time() - end)
            end = time.time()
            if i % args.print_freq == 0:
                progress.display(i)

        if epoch >= args.warmup_epoch and not args.no_verbose:
            updated_label_list = torch.cat(updated_label_list, dim=0)
            index_list = torch.cat(index_list, dim=0)
            self.train_loader.dataset.update_targets(updated_label_list.numpy(), index_list)





    def eval(self, data_loader, mode):
        model = self.model
        eval_loader = data_loader

        with torch.no_grad():
            print(f"==> {mode} Evaluation...")
            model.eval()
            pred_list = []
            true_list = []
            
            for _, (ab_batch_orig, ag_batch_orig, label_batch, type_batch, idx_batch) in enumerate(eval_loader):
                ab_batch_orig, ag_batch_orig, label_batch, idx_batch = (ab_batch_orig.to(self.device), ag_batch_orig.to(self.device), label_batch.to(self.device), idx_batch.to(self.device))
                outputs = model(ab_batch_orig, ag_batch_orig)
                
                pred = F.softmax(outputs, dim=1)
                pred_list.append(pred.cpu())
                true_list.append(torch.argmax(label_batch, dim=1).cpu())

            pred_list = torch.cat(pred_list, dim=0)
            true_list = torch.cat(true_list, dim=0)

            topk_acc, auc, f1, recall, precision = accuracy_triple(pred_list, true_list, topk=(1,))
            print(f"{mode}: Top-K Acc: {topk_acc}, AUC: {auc}, F1: {f1}, Recall: {recall}, Precision: {precision}")
        return topk_acc, auc, f1, recall, precision

    def compute_features(self, cfg):
        model = self.model
        model.eval()
        feat_list = torch.zeros(len(self.eval_loader.dataset), cfg.get("feat_dim", 64))
        
        with torch.no_grad():
            for i, (ab_batch_orig, ag_batch_orig, label_batch, type_batch, idx_batch) in enumerate(self.eval_loader):
                ab_batch_orig, ag_batch_orig, label_batch, idx_batch = (ab_batch_orig.to(self.device), ag_batch_orig.to(self.device), label_batch.to(self.device), idx_batch.to(self.device))

                _, feat = model(ab_batch_orig, ag_batch_orig, flag_feature=True)
                feat_list[idx_batch.cpu()] = feat.cpu()
        return feat_list.numpy()



if __name__ == "__main__":
    args = parser.parse_args() #args=["--cont_cutoff", "--knn_aug"]
    [args.rho_start, args.rho_end] = [float(item) for item in args.rho_range.split(",")]
    iterations = args.lr_decay_epochs.split(",")
    args.lr_decay_epochs = list([])
    for it in iterations:
        args.lr_decay_epochs.append(int(it))
    print(args)

    trainer = Trainer(args)
    trainer.train()
