import torch
import torch_geometric.data
from torch_geometric.utils import dropout_edge, degree

def weak_augmentation(graph: torch_geometric.data.Data, edge_name: str = "edge_index_b", edge_drop_p: float = 0.1):
    aug_graph = graph.clone()
    
    if edge_name == "edge_index_b":
        new_edge_index, _ = dropout_edge(aug_graph.edge_index_b, p=edge_drop_p, force_undirected=False)
        aug_graph.edge_index_b = new_edge_index
    elif edge_name == "edge_index_g":
        new_edge_index, _ = dropout_edge(aug_graph.edge_index_g, p=edge_drop_p, force_undirected=False)
        aug_graph.edge_index_g = new_edge_index        
    return aug_graph

def strong_augmentation(graph: torch_geometric.data.Data, edge_name: str = "edge_index_b", edge_drop_p: float = 0.3, node_feature_mask_p: float = 0.3):
    aug_graph = graph.clone()
    
    if edge_name == "edge_index_b":
        new_edge_index, _ = dropout_edge(aug_graph.edge_index_b, p=edge_drop_p, force_undirected=False)
        aug_graph.edge_index_b = new_edge_index

        num_nodes = aug_graph.x_b.size(0)
        mask = torch.rand(num_nodes) < node_feature_mask_p
        aug_graph.x_b[mask] = 0.0

    elif edge_name == "edge_index_g":
        new_edge_index, _ = dropout_edge(aug_graph.edge_index_g, p=edge_drop_p, force_undirected=False)
        aug_graph.edge_index_g = new_edge_index

        num_nodes = aug_graph.x_g.size(0)
        mask = torch.rand(num_nodes) < node_feature_mask_p
        aug_graph.x_g[mask] = 0.0
    
    return aug_graph
