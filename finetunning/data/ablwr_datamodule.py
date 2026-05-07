import os
import os.path as osp
from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import torch
from loguru import logger
from rich.logging import RichHandler
from torch import Tensor
from torch.utils.data import Dataset as TorchDataset
from torch_geometric.data import Batch as PyGBatch
from torch_geometric.loader import DataLoader as PyGDataLoader
from sklearn.model_selection import KFold
import re
import json
from finetunning.data.components.pair_data import PairData
logger.configure(handlers=[{"sink": RichHandler(rich_tracebacks=True), "format": "{message}"}])


# ==================== Function ====================
valid_pattern = re.compile(r'^[ACDEFGHIKLMNPQRSTVWYX]+$')
def is_valid_sequence(seq):
    return bool(valid_pattern.match(seq))

class AbLWRDataCollator:
    """
    Collator for ranking list data
    """

    def __init__(
        self,
        follow_batch: List[str] = ["x_b", "x_g"],
        exclude_keys: List[str] = ["metadata", "edge_index_bg"],
    ):
        self.follow_batch = follow_batch
        self.exclude_keys = exclude_keys

    def __call__(self, batch: List[Tuple[PairData, PairData, PairData, PairData, PairData, Tensor]]) -> Tuple[PyGBatch, PyGBatch, Tensor]:
        # Unzip the batch into separate lists
        g1s, g2s, g3s, g4s, g5s, labels = zip(*batch)

        # Create PyG batch objects for g1 and g2 graphs
        g1_batch = PyGBatch.from_data_list(g1s, follow_batch=self.follow_batch, exclude_keys=self.exclude_keys)
        g2_batch = PyGBatch.from_data_list(g2s, follow_batch=self.follow_batch, exclude_keys=self.exclude_keys)
        g3_batch = PyGBatch.from_data_list(g3s, follow_batch=self.follow_batch, exclude_keys=self.exclude_keys)
        g4_batch = PyGBatch.from_data_list(g4s, follow_batch=self.follow_batch, exclude_keys=self.exclude_keys)
        g5_batch = PyGBatch.from_data_list(g5s, follow_batch=self.follow_batch, exclude_keys=self.exclude_keys)

        # Stack the labels into a single tensor
        labels_batch = torch.stack(labels)

        return g1_batch, g2_batch, g3_batch, g4_batch, g5_batch, labels_batch
    
class AbLWRBase(TorchDataset):
    def __init__(self, seed: int = 42):
        super().__init__()
        self.seed = seed

    @abstractmethod
    def __getitem__(self, idx: int) -> Tuple[PairData, PairData]:
        pass

    @abstractmethod
    def get_pair(self, idx1: int, idx2: int) -> Tuple[PairData, PairData, Tensor]:
        pass


class AbLWRSampler(AbLWRBase):
    """
    This class is used to sample a random pair of antibody-antigen pairs
    """

    def __init__(
        self,
        combinations: Tensor,
        seed: int = 42,
        mode: str = "train",
    ):
        super().__init__(seed)
        self.combinations = combinations
        self.mode = mode

        self.pairdata_dir = "/path/to/pairdata"
        data_registry_path = "/path/to/dataset/pairgraph/AbRank-all.csv"
        df = pd.read_csv(data_registry_path)
        df["dbID"] = df["dbID"].astype(int)
        df = df.sort_values(by="dbID", ascending=True)
        self.data_registry = df.set_index("dbID")["fileName"].to_dict()


    def __len__(self) -> int:
        return len(self.combinations)


    def get_pair(self, idx: int) -> Tuple[PairData, PairData, PairData, PairData, PairData, Tensor]:
        """
        Get a pair of antibody-antigen pairs
        Remember to shuffle the order of the pairs to avoid positional bias
        """
        row = self.combinations[idx]
        ids_ = row[:5]
        logKDs_ = row[5:]
        perm = torch.randperm(5).tolist()
        ids_shuffled = tuple(ids_[i] for i in perm)
        logKDs_shuffled = tuple(logKDs_[i] for i in perm)
        dbID1, dbID2, dbID3, dbID4, dbID5 = ids_shuffled
        logKD1, logKD2, logKD3, logKD4, logKD5 = logKDs_shuffled

        g1 = self.get(dbID1)
        g2  = self.get(dbID2)
        g3  = self.get(dbID3)
        g4  = self.get(dbID4)
        g5  = self.get(dbID5)
        logKDs = torch.tensor([logKD1, logKD2, logKD3, logKD4, logKD5]).type(torch.float32)

        return g1, g2, g3, g4, g5, logKDs


    def get(self, idx:int):
        filename = self.data_registry[idx]
        g = torch.load(osp.join(self.pairdata_dir, filename))
        g.abdbid = idx
        return g
    

    def __getitem__(self, idx: int) -> Tuple[PairData, PairData, Tensor]:
        """
        Get a pair of antibody-antigen pairs
        """
        return self.get_pair(idx)





class AbLWRDataModule:
    def __init__(
        self,
        # dataset config
        save_dir_path:str,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
        n_folds: int = 5, 
        seed: int = 42,
        num_workers: int = 4,
        batch_size: int = 32,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        shuffle: bool = True,

        # PyG DataLoader config
        follow_batch: List[str] = ["x_b", "x_g"],
        exclude_keys: List[str] = ["metadata", "edge_index_bg", "y_b", "y_g", "y"],
    ):

        super().__init__()
        # --- dataset config ---
        self.save_dir = save_dir_path
        self.n_folds = n_folds
        self.transform = transform
        self.pre_transform = pre_transform
        self.pre_filter = pre_filter
        self.seed = seed
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        if self.num_workers == 0:
            self.pin_memory = False
            self.persistent_workers = False
        self.follow_batch = follow_batch
        self.exclude_keys = exclude_keys
        self.collator = AbLWRDataCollator(follow_batch=self.follow_batch, exclude_keys=self.exclude_keys)

        self._prepared = False  

    def prepare_data(self):
        if self._prepared:
            print("prepare_data already called, skipping.")
            return
    
        if not os.path.exists(os.path.join(self.save_dir, f"fold_0_indices.npy")) and not os.path.exists(os.path.join(self.save_dir, f"test_indices.npy")):
            print(f"\033[31mSpliting data...\033[0m")
            kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.seed)
            self.all_folds = list(kf.split(self.all_data))
            for i, (_, indices) in enumerate(self.all_folds):
                np.save(os.path.join(self.save_dir, f"fold_{i}_indices.npy"), indices)
        else:
            if os.path.exists(os.path.join(self.save_dir, f"fold_0_indices.npy")):
                print(f"\033[31mLoading the splited data...\033[0m")
                self.all_folds = []
                for i in range(self.n_folds):
                    indices = np.load(os.path.join(self.save_dir, f"fold_{i}_indices.npy"))
                    self.all_folds.append((None, indices))
            elif os.path.exists(os.path.join(self.save_dir, f"test_indices.npy")):
                print(f"\033[31mLoading the splited data...\033[0m")
                self.all_folds = []
                test_indices = np.load(os.path.join(self.save_dir, f"test_indices.npy"))
                self.all_folds.append((None, test_indices))

                val_indices = np.load(os.path.join(self.save_dir, f"val_indices.npy"))
                self.all_folds.append((None, val_indices))

                train_indices = np.load(os.path.join(self.save_dir, f"train_indices.npy"))
                self.all_folds.append((None, train_indices))
                
        self._prepared = True

    def setup(self, fold_idx):
        print(f"\033[31mLoading the sampled data...\033[0m")
        data_sampling_path = os.path.join(self.save_dir, f"fold_{fold_idx}_data_sampling_split.json")
        with open(data_sampling_path, "r") as f:
            data_sampling_dict = json.load(f)

        train_sampling = data_sampling_dict.get("trainset")
        val_sampling = data_sampling_dict.get("valset")
        test_sampling = data_sampling_dict.get("testset")
        print(f"Train samples: {len(train_sampling)}, Val samples: {len(val_sampling)}, Test samples: {len(test_sampling)}")


        self.train_dataset = AbLWRSampler(
            seed=self.seed,
            combinations=train_sampling,
            mode="train")
        
        self.val_dataset = AbLWRSampler(
            seed=self.seed,
            combinations=val_sampling,
            mode="val")

        self.test_dataset = AbLWRSampler(
            seed=self.seed,
            combinations=test_sampling,
            mode="test")  


    def train_dataloader(self):
        return PyGDataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            follow_batch=self.follow_batch,
            exclude_keys=self.exclude_keys,
            collate_fn=self.collator,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )

    def val_dataloader(self):
        return PyGDataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            follow_batch=self.follow_batch,
            exclude_keys=self.exclude_keys,
            collate_fn=self.collator,
            pin_memory= self.pin_memory,
            persistent_workers=self.persistent_workers,
            )
    
    def test_dataloader(self):
        return PyGDataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            follow_batch=self.follow_batch,
            exclude_keys=self.exclude_keys,
            collate_fn=self.collator,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )


# ==================== Main ====================
if __name__ == "__main__":
    pass