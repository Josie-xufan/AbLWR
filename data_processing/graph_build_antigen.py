import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import gc
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import torch
import json
import subprocess
import re
from Bio.PDB import PDBIO, PDBParser, Select
from graphein.protein.config import DSSPConfig, ProteinGraphConfig
from Bio.SeqUtils import seq1
from graphein.protein.features.nodes import rsa
from graphein.protein.graphs import construct_graph
from networkx import Graph
from pandas import DataFrame
from scipy.spatial.distance import pdist, squareform
from torch_geometric.utils import dense_to_sparse
from torch_geometric.data import Data as PygData
sys.path.insert(0, str(Path(__file__).parent.parent))

# ==============================================================================
# REFERENCE & ATTRIBUTION
# ==============================================================================
# This code is adapted/referenced from the following research works:
#
# Papers:
# 1. Liu, Chunan, et al. "Abrank: A benchmark dataset and metric-learning framework 
#    for antibody-antigen affinity ranking." arXiv preprint arXiv:2506.17857 (2025).
# 2. Liu, ChuNan, et al. "AsEP: Benchmarking deep learning methods for antibody-specific 
#    epitope prediction." Advances in Neural Information Processing Systems 37 (2024).
#
# Source Repositories:
# - https://github.com/biochunan/AbRank-WALLE-Affinity
# - https://github.com/biochunan/AsEP-dataset/tree/main
#
# License: 
# Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)
# (See: https://creativecommons.org/licenses/by-nc-sa/4.0/)
#
# Description: 
# This function is used to build the graph for antigens based on the methods described above.
# ==============================================================================




"""
Configuration
"""
# Types
PathLike = Union[str, Path]
PDBDataFrame = DataFrame
AllAtomDataFrame = DataFrame
AdjMatrix = np.ndarray
BinaryMask = np.ndarray

# Paths
BASE = Path(__file__).resolve().parent  # => walle

# structure processing
GRAPHEIN_CFG = ProteinGraphConfig(
    **{
        "granularity": "centroids",  # "atom", "CA", "centroids"
        "insertions": True,
        "edge_construction_functions": [],
        "dssp_config": DSSPConfig(executable=shutil.which("mkdssp")),
        "graph_metadata_functions": [rsa],
    }
)
RSA_THR = 0.0  # surface residue Relative solvent-accessible Surface Area threshold
CDR_DEF = "ABM"  # CDR definition
DIST_THR = 4.5  # distance threshold for contacting residues





"""
Functions
"""
class PairData(PygData):
    # define how to increment the edge_index_b and edge_index_g
    # when concatenating multiple PairData objects
    def __inc__(self, key: str, value: Any, *args, **kwargs) -> Any:
        if key == "edge_index_b":
            # return the number of the Ab nodes
            return self.x_b.size(0)
        if key == "edge_index_g":
            # return the number of the Ag nodes
            return self.x_g.size(0)
        if key == "edge_index_bg":
            # return the number of the Ab and Ag nodes
            return torch.tensor([[self.x_b.size(0)], [self.x_g.size(0)]])
        return super().__inc__(key, value, *args, **kwargs)

# helper: insert CRYST1 line to pdb file (required by DSSP)
def insert_cryst1_line_to_pdb(pdb_file: str) -> Optional[str]:
    """Insert a CRYST1 line to the pdb file if it doesn't exist, and return the new pdb file path."""
    pdb_file = Path(pdb_file)
    # check if CRYST1 line exists
    with open(pdb_file, "r") as f:
        lines = f.readlines()
    if cryst1_line := [line for line in lines if line.startswith("CRYST1")]:
        print(f"CRYST1 line already exists in {pdb_file}")
        return pdb_file.as_posix()
    # add a CRYST1 line to the pdb file
    cryst1_line = "CRYST1    1.000    1.000    1.000  90.00  90.00  90.00 P 1           1          \n"
    with open(pdb_file, "r") as f:
        lines = f.readlines()
    # add the CRYST1 line before the first line startswith ATOM
    for i, line in enumerate(lines):
        if line.startswith("ATOM"):
            lines.insert(i, cryst1_line)
            break
    # new file
    new_pdb_file = pdb_file.parent / f"{pdb_file.stem}_cryst1.pdb"
    with open(new_pdb_file, "w") as f:
        f.writelines(lines)  # write the new pdb file
    return new_pdb_file

# wrapper: call seqres2atmseq
def run_seqres2atmseq(seqres: str, atmseq: str) -> Dict[str, Any]:
    """
    Run seqres2atmseq to generate a mask file.
    Requires seqres2atmseq to be installed.
    ```
    $ pip install git+https://github.com/biochunan/seqres2atmseq.git
    ```

    Args:
        seqres (str): seqres sequence
        atmseq (str): atmseq sequence

    Raises:
        subprocess.CalledProcessError: if seqres2atmseq returns non-zero exit code

    Returns:
        Dict[str, Any]: a dictionary of the mask file
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        process = subprocess.Popen(
            [
                "seqres2atmseq",
                "-s",
                seqres,
                "-a",
                atmseq,
                "-o",
                Path(tmpdir) / "mask.json",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        stdout, stderr = process.communicate()
        retcode = process.wait()
        if retcode != 0:
            raise subprocess.CalledProcessError(retcode, "seqres2atmseq", stderr)
        with open(Path(tmpdir) / "mask.json") as f:
            seqres2atmseq_mask = json.load(f)
    return seqres2atmseq_mask



# construct graph (Graphein)
def read_pdb_as_graph(pdb_path: Path, graphein_cfg: ProteinGraphConfig, chains: Optional[List[str]] = None) -> Graph:
    """
    Copy raw pdb file to a temporary directory,
    insert "CRYST1" line to the pdb file,
    then construct graph from the new pdb file.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        pdb_fp = tmpdir / f"{pdb_path.stem}_cryst1.pdb"
        shutil.copy(pdb_path, pdb_fp)
        if chains:
            structure = PDBParser(QUIET=True).get_structure(id="pdb", file=pdb_fp)

            class ExtractChains(Select):
                def __init__(self, chain_id_to_extract):
                    self.chain_id_to_extract = chain_id_to_extract

                def accept_chain(self, chain):
                    return chain.get_id() in self.chain_id_to_extract

            io = PDBIO()
            io.set_structure(structure)
            io.save(pdb_fp.as_posix(), select=ExtractChains(chains))

        pdb_fp = insert_cryst1_line_to_pdb(pdb_fp)
        g = construct_graph(config=graphein_cfg, path=pdb_fp, verbose=False)

    return g


# decorator for process_graph
def dec_process_ag_graph(func: Callable, rsa_thr: float = 0.2) -> Callable:
    """
    A decorator that adds `is_surface` column to the output of `func`.
    `func` must return a DataFrame with `node_id` column.

    Args:
        func (Callable): a function (func: `process_graph`) that takes a Graph as input and returns a DataFrame with `node_id` column.

        rsa_thr (float, optional): Relative solvent-accessible Surface Area threshold. Defaults to 0.2.

    Returns:
        Callable: the decorated function
    """

    def wrapper(G: Graph) -> Tuple[DataFrame, DataFrame]:
        atom_df = func(G)
        # [x] TODO: add `is_surface` column
        dssp_df = G.graph["dssp_df"]
        atom_df["is_surface"] = atom_df.node_id.apply(
            lambda x: x in dssp_df.query(f"rsa > {rsa_thr}").index.values
        )
        surf_df = atom_df[atom_df["is_surface"]].reset_index(drop=True).copy()
        return atom_df, surf_df

    return wrapper


# process graph read from Graphein
def process_graph(G: Graph) -> DataFrame:
    # G, chains = G_ag.copy(), ['C']  # debug only
    # G, chains = G_ab.copy(), ['H', 'L']  # debug only
    atom_df = G.graph["raw_pdb_df"]
    # remove HETATM
    atom_df = atom_df[atom_df["record_name"] == "ATOM"].reset_index(drop=True).copy()
    return atom_df



# generate intra-graph adjacency matrix
def generate_intra_graph_adj(
    df: AllAtomDataFrame, coord_columns: Optional[List[str]]
) -> AdjMatrix:
    """
    Generate intra-graph adjacency matrix from an ALL-ATOM structure DataFrame.
    Assume column `node_id` (Returned by Graphein) exists in the DataFrame.

    Args:
        df (AllAtomDataFrame): all-atom structure DataFrame
        coord_columns (Optional[List[str]]): coordinates columns. Defaults to ["x_coord", "y_coord", "z_coord"].

    Returns:
        AdjMatrix: intra-graph adjacency matrix, shape (N, N)
            where N is the number of residues in the structure
    """
    coord_columns = coord_columns or ["x_coord", "y_coord", "z_coord"]
    x = squareform(pdist(df[coord_columns].values))

    df_nr = df.drop_duplicates("node_id").reset_index(drop=True)
    df_nr_node_id_to_idx: Dict[str, int] = {
        node_id: idx for idx, node_id in enumerate(df_nr.node_id)
    }
    n_residue = len(df_nr)
    adj = np.zeros((n_residue, n_residue)).astype(np.int8)

    rows, cols = np.where((x < DIST_THR) & (x > 0))  # >0 => exclude self-loop
    for r, c in zip(df.iloc[rows, :].node_id, df.iloc[cols, :].node_id):
        n1, n2 = df_nr_node_id_to_idx[r], df_nr_node_id_to_idx[c]
        adj[n1, n2] = 1

    return adj


# map seqres -> atmseq -> surf_mask
def map_seqres_to_atmseq_with_downstream_mask(
    seqres2atmseq_mask: Dict[str, Any], target_mask: np.ndarray
) -> AdjMatrix:
    """
    Map SEQRES to ATMSEQ then to Surface residue mask.

    Args:
        seqres2atmseq_mask (Dict[str, Any]): a dictionary with keys: 'seqres', 'atmseq', 'mask'. This is mapping from SEQRES to ATMSEQ.
            All three str values have the same length L.
        target_mask (np.ndarray): e.g. surface residue mask, this is mapping from ATMSEQ to a downstream residue mask.
            The length of this array must equal to the length of 'atmseq' (exclude '-') in `seqres2atmseq_mask`.

    Returns:
        np.ndarray: a binary mask mapping SEQRES to Surface residues, shape (L, )
    """
    assert len(target_mask) == len(seqres2atmseq_mask["seq"]["atmseq"].replace("-", ""))
    seqres2target_mask, i = [], 0
    for c in seqres2atmseq_mask["seq"]["mask"]:
        if (
            c == "1" or c == 1 or c is True
        ):  # residue exists in the structure with a surface mask
            seqres2target_mask.append(target_mask[i])
            i += 1
        elif c == "0" or c == 0 or c is False:  # missing residue
            seqres2target_mask.append(0)
    return np.array(seqres2target_mask)

def ag_emb_esm2(ag_sequence):
    """
    Returns an instance of ESM2 for antigen embedding.
    """
    # print("Step2: prepare data...")
    all_data = []
    all_data.append(("ag_seq", ag_sequence))
    print("\n")


    # print("Step3: extract per-residue representations and save as pkl...")
    sequence_representations = []
    batch_labels, batch_strs, batch_tokens = batch_converter(all_data)
    batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)
    batch_tokens = batch_tokens.to(device)
    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[33], return_contacts=True)
    token_representations = results["representations"][33]

    for i, tokens_len in enumerate(batch_lens):
        sequence_representations.append((batch_labels[i], token_representations[i, 1 : tokens_len - 1].cpu()))

    del batch_tokens, results, token_representations
    torch.cuda.empty_cache()
    gc.collect()

    return sequence_representations[0][1]



def main(job_id, ag_structure, ag_seqres, ag_chain_id) -> PairData:
    G_ag = read_pdb_as_graph(Path(ag_structure), GRAPHEIN_CFG, chains=ag_chain_id)
    ag_atom_df, surf_atom_df = dec_process_ag_graph(process_graph, rsa_thr=RSA_THR)(G_ag)

    # edge_index_g
    adj_g: AdjMatrix = generate_intra_graph_adj(df=surf_atom_df, coord_columns=None)

    # SEQRES
    ag_emb = ag_emb_esm2(ag_seqres)
    
    # SEQRES2NODES
    ag_atmseq = "".join(ag_atom_df.drop_duplicates("node_id").residue_name.apply(lambda x: seq1(x)))
    seqres2atmseq_mask = run_seqres2atmseq(seqres=ag_seqres, atmseq=ag_atmseq)
    surf_mask = ag_atom_df.drop_duplicates("node_id").is_surface.values.astype(int)
    seqres2surf_mask = map_seqres_to_atmseq_with_downstream_mask(seqres2atmseq_mask=seqres2atmseq_mask, target_mask=surf_mask)
    x_g = ag_emb[seqres2surf_mask == 1, :]

    # BUILD GRAPH REPRESENTATION
    edge_index_g, _ = dense_to_sparse(torch.from_numpy(adj_g))
    pair_data = PairData(abdbid=job_id, x_g=x_g, edge_index_g=edge_index_g)
    return pair_data




# ==================== Main ====================
if __name__ == "__main__":

    """
    Step0: Load ESM2 Model
    """
    device = torch.device("cpu") #"cuda:0" if torch.cuda.is_available() else 
    model, alphabet = torch.hub.load("facebookresearch/esm:main", "esm2_t33_650M_UR50D")
    model = model.to(device)
    batch_converter = alphabet.get_batch_converter()
    model.eval()


    """
    Step1: Prepare Sequence
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


    """
    Step2: Graph Construction
    """
    antigen_save_path = "dataset/antigen_structure"
    pt_save_path = "dataset/pairgraph/antigen"
    Ag_seq_df = pd.DataFrame(Ag_seq_set, columns=["Ag_seq"])
    Ag_seq_df = Ag_seq_df.reset_index().rename(columns={"index": "loc"})

    for row in tqdm(Ag_seq_df.itertuples(), total=len(Ag_seq_df), desc="Build Antigen graph"):
        seqid, seq = row.loc, row.Ag_seq

        pdb_path = os.path.join(antigen_save_path, f"Ag_{seqid:05d}.pdb")
        pdb_name = f"Ag_{seqid:05d}"
        
        if os.path.exists(os.path.join(pt_save_path, f"{pdb_name}.pt")):continue
        if not os.path.exists(pdb_path):continue

        pair_data = main(job_id=pdb_name, ag_structure=pdb_path, ag_seqres=seq, ag_chain_id=['A'])
        torch.save(pair_data, os.path.join(pt_save_path, f"{pdb_name}.pt"))
