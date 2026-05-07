import os
import numpy as np
import pandas as pd
import pickle
import re
import torch
from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader as PyGDataLoader
from typing import Dict, Any
from augmentation import *

valid_pattern = re.compile(r'^[ACDEFGHIKLMNPQRSTVWYX]+$')
def is_valid_sequence(seq):
    return bool(valid_pattern.match(seq))


def load_all_data():
    """
    Step1: data loading
    """
    Ab_Ag_Aff_data = pd.read_csv('../dataset/Ab_Ag_Pairs.csv')


    """
    Step2: prepare dbID_matrix, |Ag_seq|(4265) x |Ab_seq|(22691)
    """
    dbID_filepath = "../dataset/matrix_dbID.npy"
    if not os.path.exists(dbID_filepath):
        Ag_seq_set = sorted(list(set(Ab_Ag_Aff_data["Ag_seq"])))
        Ab_seq_set = sorted(list(set(Ab_Ag_Aff_data["Ab_seq"])))
        Ab_Ag_Aff_data["Ag_seq_idx"] = pd.Categorical(Ab_Ag_Aff_data["Ag_seq"], categories=Ag_seq_set).codes
        Ab_Ag_Aff_data["Ab_seq_idx"] = pd.Categorical(Ab_Ag_Aff_data["Ab_seq"], categories=Ab_seq_set).codes
        N_Ag = len(Ag_seq_set)
        N_Ab = len(Ab_seq_set)

        dbID_matrix = np.full((N_Ag, N_Ab), -1, dtype=np.int32)
        ag_indices = Ab_Ag_Aff_data["Ag_seq_idx"].to_numpy()
        ab_indices = Ab_Ag_Aff_data["Ab_seq_idx"].to_numpy()
        db_values  = Ab_Ag_Aff_data["dbID"].to_numpy(dtype=np.int32)
        dbID_matrix[ag_indices, ab_indices] = db_values
        np.save(dbID_filepath, dbID_matrix)
        print(f"Finished saving matrix_dbID.npy")            
    else:
        Ag_seq_set = sorted(list(set(Ab_Ag_Aff_data["Ag_seq"])))
        Ab_seq_set = sorted(list(set(Ab_Ag_Aff_data["Ab_seq"])))
        Ab_Ag_Aff_data["Ag_seq_idx"] = pd.Categorical(Ab_Ag_Aff_data["Ag_seq"], categories=Ag_seq_set).codes
        Ab_Ag_Aff_data["Ab_seq_idx"] = pd.Categorical(Ab_Ag_Aff_data["Ab_seq"], categories=Ab_seq_set).codes
        dbID_matrix = np.load(dbID_filepath)
        print(f"Load matrix_dbID.npy")

    return Ag_seq_set, Ab_seq_set, dbID_matrix, Ab_Ag_Aff_data


def ternarize_labels(y: np.ndarray, lo: float=-0.5, hi: float=0.5) -> np.ndarray:
    out = np.zeros_like(y, dtype=np.int64)
    out[y < lo] = -1
    out[y > hi] = 1
    out = out + 1  # -1 -> 0, 0 -> 1, 1 -> 2
    return out


def count_classes(labels: np.ndarray) -> Dict[int, int]:
    uniq, cnt = np.unique(labels, return_counts=True)
    return {int(k): int(v) for k, v in zip(uniq, cnt)}


def load_graph_pt(root_dir: str, kind: str, mol_id: str):
    path = os.path.join(root_dir, kind, mol_id + ".pt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Graph file not found: {path}")
    g = torch.load(path, weights_only=False, map_location=torch.device('cpu'))
    return g

class LabeledPairDataset(Dataset):
    def __init__(self,
                 df: pd.DataFrame,
                 graph_root: str,
                 ab_col: str="Ab_ID",
                 ag_col: str="Ag_ID",
                 label_col: str="label",
                 type_col: str="type",
                 augment: bool=False):
        self.df = df.reset_index(drop=True)
        self.graph_root = graph_root
        self.ab_col = ab_col
        self.ag_col = ag_col
        self.label_col = label_col
        self.type_col = type_col
        self.augment = augment

        self.pu_labels = self.df[self.label_col].values
        self.pu_labels = np.eye(self.pu_labels.max()+1)[self.pu_labels]

    def __len__(self):
        return len(self.df)
    

    def update_targets(self, new_labels, idxes):
        self.pu_labels[idxes] = new_labels

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        ab_id = int(row[self.ab_col])
        ab_id = f"Ab_{ab_id:05d}"

        ag_id = int(row[self.ag_col])
        ag_id = f"Ag_{ag_id:05d}"

        # y = int(row[self.label_col])
        y = self.pu_labels[idx]
        t = str(row[self.type_col])

        ab_g = load_graph_pt(self.graph_root, "antibody", ab_id)
        ag_g = load_graph_pt(self.graph_root, "antigen", ag_id)

        if self.augment:
            ab_g_w = weak_augmentation(ab_g, edge_name="edge_index_b")
            ag_g_w = weak_augmentation(ag_g, edge_name="edge_index_g")

            ab_g_s = strong_augmentation(ab_g, edge_name="edge_index_b")
            ag_g_s = strong_augmentation(ag_g, edge_name="edge_index_g")

            return ab_g, ag_g, ab_g_w, ag_g_w, ab_g_s, ag_g_s, torch.tensor(y, dtype=torch.long), t, idx
        else: 
            return ab_g, ag_g, torch.tensor(y, dtype=torch.long), t, idx



def make_dataloader(dataset: Dataset,
                    batch_size: int,
                    shuffle: bool=True,
                    num_workers: int=4) -> PyGDataLoader:
    follow_batch = ["x_b", "x_g"]
    exclude_keys = []

    return PyGDataLoader(dataset,
                      batch_size=batch_size,
                      shuffle=shuffle,
                      num_workers=num_workers,
                      follow_batch=follow_batch,
                      exclude_keys=exclude_keys,
                      pin_memory=(num_workers > 0),
                      persistent_workers=(num_workers > 0))



def make_splits(df_labeled: pd.DataFrame,
                fold_idx: int=-1,
                n_folds: int=5, 
                seed: int=42,
                val_test_ratio: float=0.2) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    if fold_idx==-1:
        print("Spliting data...")
        rng = np.random.default_rng(seed)
        all_dbID = df_labeled["dbID"].values.copy()
        rng.shuffle(all_dbID)

        n_total = len(df_labeled)
        n_val = int(n_total * val_test_ratio)
        n_test = int(n_total * val_test_ratio)
        n_train = n_total - n_val - n_test

        train_ids = set(all_dbID[:n_train])
        val_ids = set(all_dbID[n_train:n_train+n_val])
        test_ids = set(all_dbID[n_train+n_val:])

        train_df = df_labeled[df_labeled["dbID"].isin(train_ids)].reset_index(drop=True)
        val_df = df_labeled[df_labeled["dbID"].isin(val_ids)].reset_index(drop=True)
        test_df = df_labeled[df_labeled["dbID"].isin(test_ids)].reset_index(drop=True)
    elif fold_idx >=0 and fold_idx < n_folds:
        save_dir_path = "../dataset/runs_CV"
        print(f"Using general Split from {save_dir_path}...")

        val_folds = fold_idx
        test_folds = (fold_idx + 1) % n_folds
        train_folds = [i for i in range(n_folds) if i not in [val_folds, test_folds]]

        val_indices = np.load(os.path.join(save_dir_path, f"fold_{val_folds}_indices.npy"))
        test_indices = np.load(os.path.join(save_dir_path, f"fold_{test_folds}_indices.npy"))
        train_indices = []
        for i in train_folds:
            idx = np.load(os.path.join(save_dir_path, f"fold_{i}_indices.npy"))
            train_indices.extend(idx)


        val_df = df_labeled.iloc[val_indices].reset_index(drop=True)
        test_df = df_labeled.iloc[test_indices].reset_index(drop=True)
        train_df = df_labeled.iloc[train_indices].reset_index(drop=True)
    elif fold_idx == n_folds: # Ab/Ag-based Split
        """
        Remeber to change the save_dir_path accordingly:
        - Ag-based Split: dataset/runs_Agbasedsplit_Cluster0
        - Ab-based Split: dataset/runs_Abbasedsplit_Cluster1 or dataset/runs_Abbasedsplit_Cluster5
        """
        save_dir_path = "../dataset/runs_Agbasedsplit_Cluster0"
        print(f"Using Hard Ag/Ab Split from {save_dir_path}...")

        val_indices = np.load(os.path.join(save_dir_path, f"val_indices.npy"))
        test_indices = np.load(os.path.join(save_dir_path, f"test_indices.npy"))
        train_indices = np.load(os.path.join(save_dir_path, f"train_indices.npy"))

        val_df = df_labeled.iloc[val_indices].reset_index(drop=True)
        test_df = df_labeled.iloc[test_indices].reset_index(drop=True)
        train_df = df_labeled.iloc[train_indices].reset_index(drop=True)



    return train_df, val_df, test_df

def build_unknown_pool(ab_list: list[str],
                       ag_list: list[str],
                       labeled_pairs: set[tuple[str, str]],
                       max_pool_size: int,
                       seed: int,
                       ab_col: str,
                       ag_col: str) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ab_arr = np.array(ab_list)
    ag_arr = np.array(ag_list)
    n_ab = len(ab_arr)
    n_ag = len(ag_arr)

    # sampling max_pool_size pairs not in labeled_pairs
    pairs = []
    seen = set()
    trials = 0
    max_trials = max_pool_size * 20

    while len(pairs) < max_pool_size and trials < max_trials:
        trials += 1
        ai = int(rng.integers(0, n_ab))
        gi = int(rng.integers(0, n_ag))
        ab = str(ab_arr[ai])
        ag = str(ag_arr[gi])
        key = (ai, gi)
        if key in labeled_pairs: 
            continue
        if key in seen:
            continue
        seen.add(key)
        pairs.append(key)

    unk_df = pd.DataFrame(pairs, columns=[ab_col, ag_col])
    return unk_df



    
def load_dataset(args):
    Ag_seq_set, Ab_seq_set, _, paper_Ab_Ag_Aff_data_filtered = load_all_data()
    graph_root = args.graph_root
    ab_col = getattr(args, "ab_col", "Ab_seq_idx")
    ag_col = getattr(args, "ag_col", "Ag_seq_idx")
    label_col = getattr(args, "label_col", "label")
    type_col = getattr(args, "type_col", "type")


    # 1) discretize labels
    df = paper_Ab_Ag_Aff_data_filtered.copy()
    df["label"] = ternarize_labels(df["log_Aff_new"].values, lo=-0.5, hi=0.5)


    # 2) split
    train_df, val_df, test_df = make_splits(df, fold_idx=args.fold_idx, n_folds=args.n_folds, seed=args.seed, val_test_ratio=0.2)
    print("Training Data with labels: ", count_classes(train_df["label"].values))
    print("Val Data with labels: ", count_classes(val_df["label"].values))
    print("Test Data with labels: ", count_classes(test_df["label"].values))


    # 3) Unknown pool construction：sampling from the dbID matrix with the size of |labeled Ab-Ag pairs| or N*(|labeled Ab-Ag pairs|)
    ab_all = list(Ab_seq_set.keys()) if isinstance(Ab_seq_set, dict) else list(Ab_seq_set)
    ag_all = list(Ag_seq_set.keys()) if isinstance(Ag_seq_set, dict) else list(Ag_seq_set)

    labeled_pairs = set(zip(df[ab_col].astype(int).tolist(), df[ag_col].astype(int).tolist()))
    target_pool_size = int(len(train_df))
    unk_df = build_unknown_pool(ab_all, ag_all, labeled_pairs, max_pool_size=target_pool_size, seed=args.seed, ab_col=ab_col, ag_col=ag_col)
    unk_df[label_col] = 1



    # 4) Dataset/Dataloader（labeled）
    train_df_ = train_df[[ab_col, ag_col, label_col]].reset_index(drop=True)
    train_df_[type_col] = "train"
    unk_df_ = unk_df[[ab_col, ag_col, label_col]].reset_index(drop=True)
    unk_df_[type_col] = "unk"
    train_unk_df = pd.concat([train_df_, unk_df_], axis=0).reset_index(drop=True)

    val_df_ = val_df[[ab_col, ag_col, label_col]].reset_index(drop=True)
    val_df_[type_col] = "val"
    test_df_ = test_df[[ab_col, ag_col, label_col]].reset_index(drop=True)
    test_df_[type_col] = "test"

    train_ds = LabeledPairDataset(train_unk_df, graph_root, ab_col=ab_col, ag_col=ag_col, label_col=label_col, type_col=type_col, augment=True)
    val_ds   = LabeledPairDataset(val_df_,   graph_root, ab_col=ab_col, ag_col=ag_col, label_col=label_col, type_col=type_col, augment=False)
    test_ds  = LabeledPairDataset(test_df_,  graph_root, ab_col=ab_col, ag_col=ag_col, label_col=label_col, type_col=type_col, augment=False)
    eval_ds = LabeledPairDataset(train_unk_df, graph_root, ab_col=ab_col, ag_col=ag_col, label_col=label_col, type_col=type_col, augment=False)

    train_loader = make_dataloader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=args.num_workers)
    val_loader   = make_dataloader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader  = make_dataloader(test_ds,  batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    eval_loader  = make_dataloader(eval_ds,  batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)



    meta = {
        "train_df": train_df,
        "val_df": val_df,
        "test_df": test_df,
        "ab_col": ab_col,
        "ag_col": ag_col,
        "label_col": label_col,
        "type_col": type_col,
    }
    return train_loader, val_loader, test_loader, eval_loader, meta