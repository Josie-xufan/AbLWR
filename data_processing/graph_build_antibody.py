import re
import shutil
import sys
import tempfile
from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import torch
import json
import subprocess
from Bio.PDB import PDBIO, PDBParser, Select
from graphein.protein.config import DSSPConfig, ProteinGraphConfig
from Bio.SeqUtils import seq1
from graphein.protein.features.nodes import rsa
from graphein.protein.graphs import construct_graph
from networkx import Graph
from pandas import DataFrame
from scipy.spatial.distance import pdist, squareform
from torch_geometric.utils import dense_to_sparse
from igfold import IgFoldRunner
from anarci import anarci
from torch_geometric.data import Data as PygData
sys.path.insert(0, str(Path(__file__).parent.parent))
os.environ["CUDA_VISIBLE_DEVICES"] = "6"

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
CDR_DEF = "CHOTHIA"  # CDR definition
DIST_THR = 4.5  # distance threshold for contacting residues









"""
Functions
"""
# ref: http://www.bioinf.org.uk/abs/info.html#cdrdef
CDR = {
    "ABM": {
        "H1": [26, 35],
        "H2": [50, 58],
        "H3": [95, 102],
        "L1": [24, 34],
        "L2": [50, 56],
        "L3": [89, 97],
    },
    "IMGT": {
        "H1": [26, 33],
        "H2": [51, 56],
        "H3": [93, 102],
        "L1": [27, 32],
        "L2": [50, 51],
        "L3": [89, 97],
    },
    "KABAT": {
        "H1": [31, 35],
        "H2": [50, 65],
        "H3": [95, 102],
        "L1": [24, 34],
        "L2": [50, 56],
        "L3": [89, 97],
    },
    "CHOTHIA": {
        "H1": [26, 32],
        "H2": [52, 56],
        "H3": [96, 101],
        "L1": [26, 32],
        "L2": [50, 52],
        "L3": [91, 96],
    },
}

_to_list = lambda cdr, a, b: [f"{cdr[0]}{i}" for i in range(a, b + 1)]
# 'H1' -> [H26, H26, ..., H35]
CDR2Resi = {
    name: {
        cdr: _to_list(
            cdr,
            a,
            b,
        )
        for cdr, (a, b) in d.items()
    }
    for name, d in CDR.items()
}
# 'H26' -> H1
Resi2CDR = {name: {resi: cdr for cdr, resis in d.items() for resi in resis}for name, d in CDR2Resi.items()}
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
def dec_process_ab_graph(func: Callable, cdr_def: Optional[str] = None) -> Callable:
    """
    A decorator that adds `cdr` (str, H1, H2, ...) and `is_cdr` (bool) columns to the output of `func`.
    `func` must return a DataFrame with `node_id` column.

    Args:
        func (Callable): a function (func: `process_graph`) that takes a Graph as input and returns a DataFrame with `node_id` column.
        cdr_def (Optional[str], optional): CDR definition. Defaults to 'ABM'.

    Returns:
        Returns:
        Callable: the decorated function
    """
    cdr_def = cdr_def or "ABM"

    def wrapper(G: Graph, *args, **kwargs) -> Tuple[DataFrame, DataFrame]:
        atom_df = func(G, *args, **kwargs)
        # [x]TODO: add `cdr`, `is_cdr` column
        atom_df["cdr"] = atom_df.apply(lambda x: Resi2CDR[cdr_def].get(f"{x.chain_id}{x.residue_number}", ""),axis=1,)
        atom_df["is_cdr"] = atom_df.cdr.apply(lambda x: x != "")
        # [x] TODO: Add a step to force chain order of H, L
        atom_df = pd.concat(
            [
                atom_df.query(f'chain_id == "{c}"')
                for c in ["H", "L"]
                if c in atom_df.chain_id.unique()
            ]
        ).reset_index(drop=True)
        cdr_df = atom_df[atom_df["is_cdr"]].reset_index(drop=True).copy()
        return atom_df, cdr_df

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

def ab_emb_igfold(ab_sequence):
    """
    Returns an instance of IgFoldRunner for antibody embedding.
    """
    emb = igfold.embed(sequences=ab_sequence)
    final_emb = torch.squeeze(emb.bert_embs)
    return final_emb

def pick_unique(series, key_name=""):
    vals = pd.unique(series.dropna())
    if len(vals) == 0:
        return np.nan
    if len(vals) > 1:
        print(f"[WARN] Multiple distinct values found for the same antibody in column {key_name}. Using the first one. Candidates={len(vals)}", file=sys.stderr)
    return vals[0]


def extract_chothia_variable_region(seq, chain_type):
    results = anarci([(chain_type, seq)], scheme='chothia')
    numbered = results[0][0][0][0]
    variable_seq = ''.join([res[1] for res in numbered if res[1] != '-' and res[1] is not None])
    
    if len(variable_seq) < 50:
        print(f"Warning: {chain_type} chain variable region extraction failed, length={len(variable_seq)}")
    return variable_seq

def main(job_id, ab_structure, ab_seqres, ab_chain_id) -> PairData:
    G_ab = read_pdb_as_graph(Path(ab_structure), GRAPHEIN_CFG, chains=ab_chain_id)
    ab_atom_df, cdr_atom_df = dec_process_ab_graph(process_graph, cdr_def=CDR_DEF)(G_ab)

    # edge_index_b
    adj_b: AdjMatrix = generate_intra_graph_adj(df=cdr_atom_df, coord_columns=None)

    # SEQRES
    ab_emb = ab_emb_igfold(ab_seqres)
    
    # SEQRES2NODES
    seqres = "".join(ab_seqres.values())
    atmseq = "".join(ab_atom_df.drop_duplicates("node_id").residue_name.apply(lambda x: seq1(x)))
    seqres2atmseq_mask = run_seqres2atmseq(seqres=seqres, atmseq=atmseq)
    atmseq2cdr_mask = ab_atom_df.drop_duplicates("node_id").is_cdr.values.astype(int)
    seqres2cdr_mask = map_seqres_to_atmseq_with_downstream_mask(seqres2atmseq_mask=seqres2atmseq_mask, target_mask=atmseq2cdr_mask)
    seqres2cdr_mask = np.array(seqres2cdr_mask).astype(bool)
    x_b = ab_emb[seqres2cdr_mask == 1, :]

    # BUILD GRAPH REPRESENTATION
    edge_index_b, _ = dense_to_sparse(torch.from_numpy(adj_b))
    pair_data = PairData(abdbid=job_id, x_b=x_b, edge_index_b=edge_index_b)
    return pair_data

# ==================== Main ====================
if __name__ == "__main__":

    """
    Step0: Load Igfold Model
    """
    igfold = IgFoldRunner()


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

    Ab_seq_df = Ab_Ag_Aff_data.groupby("Ab_seq_idx", as_index=False).agg({
        "Ab_heavy_chain_seq": lambda s: pick_unique(s, "Ab_heavy_chain_seq"),
        "Ab_light_chain_seq": lambda s: pick_unique(s, "Ab_light_chain_seq"),
    })

    


    """
    Step2: Graph Construction
    """
    antibody_save_path = "dataset/antibody_structure"
    pt_save_path = "dataset/pairgraph/antibody"


    for row in tqdm(Ab_seq_df.itertuples(), total=len(Ab_seq_df), desc="Build Antibody graph"):
        ab_idx = row.Ab_seq_idx
        heavy_seq = row.Ab_heavy_chain_seq
        light_seq = row.Ab_light_chain_seq
        heavy_var = extract_chothia_variable_region(heavy_seq, "H")
        light_var = extract_chothia_variable_region(light_seq, "L")

        pdb_path = os.path.join(antibody_save_path, f"Ab_{ab_idx:05d}.pdb")
        pdb_name = f"Ab_{ab_idx:05d}"

        if os.path.exists(os.path.join(pt_save_path, f"{pdb_name}.pt")):continue

        if len(heavy_var)>0:
            ab_seqres = {"H": heavy_var, "L": light_var}
        else:
            ab_seqres = {"L": light_var}
        ab_seqres = OrderedDict({x: ab_seqres[x] for x in "HL" if x in ab_seqres})            

        pair_data = main(job_id=pdb_name, ab_structure=pdb_path, ab_seqres=ab_seqres, ab_chain_id=["H", "L"])
        torch.save(pair_data, os.path.join(pt_save_path, f"{pdb_name}.pt"))


