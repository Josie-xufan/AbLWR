import os
import torch
import sys
import gc
import json
import rootutils
import argparse
import random
import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))
from finetunning.data.ablwr_datamodule import AbLWRDataModule
from finetunning.models.ablwr_ranking_gcn import RegressionGCNAbAgIntLM
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
BASE = Path(__file__).parent
torch.set_float32_matmul_precision(precision="medium")
os.environ["CUDA_VISIBLE_DEVICES"] = "4"


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pt_path', type=str, default=None, help='Path to config file')
    parser.add_argument('--fold_idx', type=int, default=0, help='select one fold for CV')
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main(cfg, pt_path, fold_idx):
    set_seed(cfg["seed"])
    device = torch.device(cfg["device"] if torch.cuda.is_available() else 'cpu')
    data_module = AbLWRDataModule(save_dir_path=cfg["data_dir"],
                                   n_folds=cfg["n_fold"], seed=cfg["seed"], num_workers=cfg["num_workers"], batch_size=cfg["batch_size"],
                                   pin_memory=True, persistent_workers=True, shuffle=True,)
    print(f"\033[31mStarting Testing...\033[0m")
    data_module.prepare_data()
    data_module.setup(fold_idx)
    test_loader = data_module.test_dataloader()


    model = RegressionGCNAbAgIntLM(cfg)
    model = model.to(device)
    pret_model_dict = torch.load(pt_path, map_location=device)
    model.load_state_dict(pret_model_dict)


    model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc="Prediction", leave=False, total=len(test_loader))):
            batch = [batch_.to(device) for batch_ in batch]
            loss, test_full_acc, test_pair_acc = model.pred_step(batch)
    test_avg_loss, test_full_rank_acc, test_pairwise_rank_acc, test_pred_values_np, test_true_values_np, dbid0, dbid1, dbid2, dbid3, dbid4 = model.on_pred_epoch_end()
    logger.info(f"[Finish Prediction, "
                f"Avg Loss: {test_avg_loss:.4f}, "
                f"Full Rank Acc: {test_full_rank_acc:.4f}, Pairwise Rank Acc: {test_pairwise_rank_acc:.4f}")
    
    csv_filename = os.path.dirname(pt_path) + "/Prediction.csv"
    pred_columns = [f"pred_{i}" for i in range(test_pred_values_np.shape[1])]
    true_columns = [f"true_{i}" for i in range(test_true_values_np.shape[1])]
    test_pred_df = pd.DataFrame(data=np.hstack((test_pred_values_np, test_true_values_np)),columns=pred_columns + true_columns)
    
    test_pred_df["dbid0"] = dbid0
    test_pred_df["dbid1"] = dbid1
    test_pred_df["dbid2"] = dbid2
    test_pred_df["dbid3"] = dbid3
    test_pred_df["dbid4"] = dbid4
    
    test_pred_df.to_csv(csv_filename, index=False)
    print(f"Saved results to {csv_filename}")
    gc.collect()
    torch.cuda.empty_cache()



if __name__ == "__main__":
    args = get_args()
    pt_path = args.pt_path

    config_path = os.path.dirname(pt_path) + "/config.json"
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = json.load(f)
    
    fold_idx = args.fold_idx
    main(cfg, pt_path, fold_idx)





    


