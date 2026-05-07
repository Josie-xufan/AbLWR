from igfold import IgFoldRunner
from igfold.refine.pyrosetta_ref import init_pyrosetta
import os
import re
import pandas as pd
import numpy as np
import torch
import gc
import time
from tqdm import tqdm
from anarci import anarci
import sys
from abnumber.exceptions import ChainParseError
init_pyrosetta()

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
antibody_save_path = "dataset/antibody_structure"

"""
Functions for use
"""
def extract_chothia_variable_region(seq, chain_type):
    results = anarci([(chain_type, seq)], scheme='chothia')
    numbered = results[0][0][0][0] # Residue number for chothia scheme
    variable_seq = ''.join([res[1] for res in numbered if res[1] != '-' and res[1] is not None])
    
    if len(variable_seq) < 50:  # len of variable length usually is smaller than 50
        print(f"Warning: {chain_type} chain variable region extraction failed, length={len(variable_seq)}")
    return variable_seq


def pick_unique(series, key_name=""):
    vals = pd.unique(series.dropna())
    if len(vals) == 0:
        return np.nan
    if len(vals) > 1:
        print(f"[WARN] Multiple distinct values found for the same antibody in column {key_name}. Using the first one. Candidates={len(vals)}", file=sys.stderr)
    return vals[0]


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
Ab_seq_df = Ab_Ag_Aff_data.groupby("Ab_seq_idx", as_index=False).agg({
    "Ab_heavy_chain_seq": lambda s: pick_unique(s, "Ab_heavy_chain_seq"),
    "Ab_light_chain_seq": lambda s: pick_unique(s, "Ab_light_chain_seq"),})




"""
Step2: predict antibody structure
"""
igfold = IgFoldRunner()
for row in tqdm(Ab_seq_df.itertuples(), total=len(Ab_seq_df), desc="Predicting Antibody structures"):
    ab_idx = row.Ab_seq_idx
    heavy_seq = row.Ab_heavy_chain_seq
    light_seq = row.Ab_light_chain_seq
    heavy_var = extract_chothia_variable_region(heavy_seq, "H")
    light_var = extract_chothia_variable_region(light_seq, "L")
    sequences = {"H": heavy_var, "L": light_var}

    outp = os.path.join(antibody_save_path, f"Ab_{ab_idx:05d}.pdb")
    if os.path.exists(outp): continue

    try:
        igfold.fold(
            outp,
            sequences=sequences, 
            do_refine=True, # Refine the antibody structure with PyRosetta
            do_renum=True, # Renumber predicted antibody structure (Chothia)
        )
    except RuntimeError as e:
        print(f"[ERROR] Antibody {ab_idx} skipping!")
        print(e)
        continue


    torch.cuda.empty_cache()
    gc.collect()
    time.sleep(2)