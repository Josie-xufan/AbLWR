import torch
import esm
import os
from tqdm import tqdm
import numpy as np
import pandas as pd
import re
import gc
import time

device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")
antigen_save_path = "dataset/antigen_structure"



"""
Step1: prepare sequence
"""
Ab_Ag_Aff_data = pd.read_csv('dataset/Ab_Ag_Pairs.csv')
valid_pattern = re.compile(r'^[ACDEFGHIKLMNPQRSTVWYX]+$')
def is_valid_sequence(seq):
    return bool(valid_pattern.match(seq))
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
Ag_seq_df = pd.DataFrame(Ag_seq_set, columns=["Ag_seq"])
Ag_seq_df = Ag_seq_df.reset_index().rename(columns={"index": "loc"})




"""
Step2: predict antigen structure
"""
print("Loading ESMFold model ...")
model = esm.pretrained.esmfold_v1()
model = model.eval().to(device)
model.set_chunk_size(32)
for row in tqdm(Ag_seq_df.itertuples(), total=len(Ag_seq_df), desc="Predicting Antigen structures"):
    seqid = row.loc
    seq = row.Ag_seq
    output_path = os.path.join(antigen_save_path, f"Ag_{seqid:05d}.pdb")
    if os.path.exists(output_path): continue

    try:
        # print(f"Predicting antigen {seqid}, length {len(seq)} ...")
        with torch.no_grad():
            output = model.infer_pdb(seq)
        
        with open(output_path, "w") as f:
            f.write(output)
    except RuntimeError as e:
        print(f"[Warning] Antigen {seqid} too long for GPU, skipping! Length: {len(seq)}")
        continue      

    torch.cuda.empty_cache()
    gc.collect()
    time.sleep(2)
