import os
import torch
import os
from pathlib import Path
import rootutils
from loguru import logger
import argparse
from pathlib import Path
import rootutils
import torch
from loguru import logger
import sys
import random
import yaml
from datetime import datetime
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import gc
import json
import nni
import pandas as pd
import numpy as np
from itertools import product
import time
sys.path.insert(0, str(Path(__file__).parent.parent))

from finetunning.models.ablwr_ranking_gcn import RegressionGCNAbAgIntLM
from finetunning.data.ablwr_datamodule import AbLWRDataModule
from finetunning.callbacks import ModelCheckpoint_val, EarlyStopping, LearningRateMonitor

# ==================== Configuration ====================
torch.set_float32_matmul_precision(precision="medium")
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

# ============ functions for use  =============
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=None, help='Path to config file')
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _num_training_steps(train_dataloader, n_epochs, accumulate_grad_batches=1, num_devices=1):
    total_batches = len(train_dataloader) * n_epochs
    effective_batch_size = accumulate_grad_batches * num_devices
    total_steps = total_batches // effective_batch_size
    return total_steps


# =============================== Main Training Loop =============================
def main(cfg):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    cfg['save_dir'] = f"{cfg['save_dir']}/{timestamp}_{cfg['exp_name']}_bs{cfg['batch_size']}_lr{cfg['optimizer']['lr']}_wd{cfg['optimizer']['weight_decay']}_dr{cfg['regression']['dropout_rate']}_bl{cfg['regression']['num_blocks']}_hd{cfg['regression']['num_heads']}_ind{cfg['regression']['num_induced']}"
    set_seed(cfg["seed"])
    device = torch.device(cfg["device"] if torch.cuda.is_available() else 'cpu')
    os.makedirs(cfg["save_dir"], exist_ok=True)

    # ==== data preparation ====
    data_module = AbLWRDataModule(save_dir_path=cfg["data_dir"],
                                   n_folds=cfg["n_fold"], seed=cfg["seed"], num_workers=cfg["num_workers"], batch_size=cfg["batch_size"],
                                   pin_memory=True, persistent_workers=True, shuffle=True,)
    data_module.prepare_data()



    # ==== training ====
    best_metric_dict = {}
    print(f"\033[31mStarting {cfg['n_fold']}-Fold Cross Validation...\033[0m")
    for fold_idx in range(cfg["n_fold"]):
        fold_dir = os.path.join(cfg["save_dir"], f"kfold_{fold_idx}")
        os.makedirs(fold_dir, exist_ok=True)


        # ===== callback =====
        cfg_model_ckpt = cfg["callbacks"]["model_checkpoint"]
        cfg_early_stop = cfg["callbacks"]["early_stopping"]
        model_val_ckpt = ModelCheckpoint_val(monitor=cfg_model_ckpt["monitor"], mode=cfg_model_ckpt["mode"], save_top_k=cfg_model_ckpt["save_top_k"], dirpath=fold_dir, filename='bestval', logger=logger, verbose=True)
        early_stop = EarlyStopping(monitor=cfg_early_stop["monitor"], patience=cfg_early_stop["patience"], min_delta=cfg_early_stop["min_delta"], mode=cfg_early_stop["mode"], logger=logger, verbose=True)
        lr_monitor = LearningRateMonitor()    


        # ==== dataset ====
        data_module.setup(fold_idx)
        train_loader = data_module.train_dataloader()
        val_loader = data_module.val_dataloader()
        test_loader = data_module.test_dataloader()



        # ==== model initialization ====
        model = RegressionGCNAbAgIntLM(cfg)
        model = model.to(device)


        # ==== pretrained weight loading ====
        if os.path.exists(cfg["model_init"]["pretrained_model_ckpt"]):
            try:
                pret_model_dict = torch.load(cfg["model_init"]["pretrained_model_ckpt"], map_location=device)["model_state_dict"]
                model_dict = model.state_dict()
                for k, v in pret_model_dict.items():
                    if "decoder" not in k: model_dict[k] = v
                pt_path = cfg["model_init"]["pretrained_model_ckpt"]
                print(f"Pretrained model load from {pt_path}")
            except:
                pret_model_dict = torch.load(cfg["model_init"]["pretrained_model_ckpt"], map_location=device)
                model_dict = model.state_dict()
                for k, v in pret_model_dict.items():
                    if "encoder" in k: model_dict[k] = v
                pt_path = cfg["model_init"]["pretrained_model_ckpt"]
                print(f"Pretrained model load from {pt_path}")
            model.load_state_dict(model_dict)
        else:
            print("No pretrained model ckpt provided, training from scratch.")


        # ==== opt and loss ====
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["optimizer"]["lr"], weight_decay=cfg["optimizer"]["weight_decay"])
        num_steps = _num_training_steps(train_loader, cfg["trainer"]["max_epochs"], cfg["trainer"]["accumulate_grad_batches"], num_devices=1)
        scheduler = CosineAnnealingLR(optimizer, T_max=num_steps, eta_min=1e-6)


        # ==== start training ====
        cfg_trainer = cfg["trainer"]
        gradient_clip_algorithm = cfg_trainer["gradient_clip_algorithm"]
        gradient_clip_val = cfg_trainer["gradient_clip_val"]
        accumulate_grad_batches = cfg_trainer["accumulate_grad_batches"]
        best_metric_test = 0
        best_metric_val = 0
        
        for epoch in range(cfg["trainer"]["max_epochs"]):
            model.train()
            optimizer.zero_grad()
            for batch_idx, batch in enumerate(tqdm(train_loader, desc="Training", leave=False, total=len(train_loader))):
                global_step = epoch * len(train_loader) + batch_idx
                batch = [batch_.to(device) for batch_ in batch]

                loss, full_acc, pair_acc = model.training_step(batch, global_step)
                loss = loss / accumulate_grad_batches
                loss.backward()

                
                if (batch_idx + 1) % accumulate_grad_batches == 0 or (batch_idx + 1) == len(train_loader):
                    if gradient_clip_algorithm == "norm":
                        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_val)
                    elif gradient_clip_algorithm == "value":
                        torch.nn.utils.clip_grad_value_(model.parameters(), gradient_clip_val)

                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()
                    lr_monitor.step(optimizer, epoch, batch_idx)

                # logger.info(f"[Training Epoch [{epoch+1}/{cfg['trainer']['max_epochs']}], "
                #             f"Step [{batch_idx+1}/{len(train_loader)}], "
                #             f"Loss: {loss.item()*accumulate_grad_batches:.4f}, "
                #             f"Full Acc: {full_acc:.4f}, Pair Acc: {pair_acc:.4f}")
            
            avg_loss, full_rank_acc, pairwise_rank_acc, _, _ = model.on_train_epoch_end(epoch=epoch)
            logger.info(f"[Finish Training Epoch [{epoch+1}/{cfg['trainer']['max_epochs']}], "
                        f"Avg Loss: {avg_loss:.4f}, "
                        f"Full Rank Acc: {full_rank_acc:.4f}, Pairwise Rank Acc: {pairwise_rank_acc:.4f}")

            # ==== validation ====
            model.eval()
            with torch.no_grad():
                for batch_idx, batch in enumerate(tqdm(val_loader, desc="validation", leave=False, total=len(val_loader))):
                    global_step = epoch * len(val_loader) + batch_idx
                    batch = [batch_.to(device) for batch_ in batch]
                    loss, val_full_acc, val_pair_acc = model.validation_step(batch, global_step)
            val_avg_loss, val_full_rank_acc, val_pairwise_rank_acc, val_pred_values_np, val_true_values_np = model.on_validation_epoch_end(epoch=epoch)
            logger.info(f"[Finish Validation Epoch [{epoch+1}/{cfg['trainer']['max_epochs']}], "
                        f"Avg Loss: {val_avg_loss:.4f}, "
                        f"Full Rank Acc: {val_full_rank_acc:.4f}, Pairwise Rank Acc: {val_pairwise_rank_acc:.4f}")
            val_rank_acc = val_full_rank_acc + val_pairwise_rank_acc
            


            # ==== callback ====
            model_val_ckpt.step(model, val_rank_acc, epoch)
            early_stop.step(val_rank_acc)
            lr_monitor.log_last()



            # ==== test ====
            model.eval()
            with torch.no_grad():
                for batch_idx, batch in enumerate(tqdm(test_loader, desc="test", leave=False, total=len(test_loader))):
                    global_step = epoch * len(test_loader) + batch_idx
                    batch = [batch_.to(device) for batch_ in batch]
                    loss, test_full_acc, test_pair_acc = model.test_step(batch, global_step)
            test_avg_loss, test_full_rank_acc, test_pairwise_rank_acc, test_pred_values_np, test_true_values_np = model.on_test_epoch_end(epoch=epoch)
            logger.info(f"[Finish Test Epoch [{epoch+1}/{cfg['trainer']['max_epochs']}], "
                        f"Avg Loss: {test_avg_loss:.4f}, "
                        f"Full Rank Acc: {test_full_rank_acc:.4f}, Pairwise Rank Acc: {test_pairwise_rank_acc:.4f}")
            test_rank_acc = test_full_rank_acc + test_pairwise_rank_acc


            if val_rank_acc > best_metric_val:
                best_metric_val = val_rank_acc

                csv_filename = f"best_val_results_epoch_{epoch:03d}.csv"
                pred_columns = [f"pred_{i}" for i in range(test_pred_values_np.shape[1])]
                true_columns = [f"true_{i}" for i in range(test_true_values_np.shape[1])]
                test_pred_df = pd.DataFrame(data=np.hstack((test_pred_values_np, test_true_values_np)),columns=pred_columns + true_columns)
                test_pred_df.to_csv(os.path.join(fold_dir, csv_filename), index=False)
                print(f"Saved validation results to {csv_filename}")


            if early_stop.should_stop:
                logger.info("Early stopping triggered.")
                break

        gc.collect()
        torch.cuda.empty_cache()

        with open(fold_dir + "/config.json", 'w', encoding='utf-8') as f:
            json.dump(cfg, f, indent=4, ensure_ascii=False)

        best_metric_dict[f"fold_{fold_idx}"] = best_metric_test    
    return best_metric_dict[f"fold_{fold_idx}"]



if __name__ == "__main__":
    args = get_args()
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)
    best_metric = main(cfg)
    print(f"\nFinal best test metric: {best_metric}")

    torch.cuda.empty_cache()
    print("\nWaiting for GPU to clear...")
    time.sleep(5)



    


