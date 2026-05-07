from typing import Any
import torch
from torch_geometric.data import Data as PygData
class PairData(PygData):
    def __inc__(self, key: str, value: Any, *args, **kwargs) -> Any:
        if key == "edge_index_b":
            return self.x_b.size(0)
        if key == "edge_index_g":
            return self.x_g.size(0)
        if key == "edge_index_bg":
            return torch.tensor([[self.x_b.size(0)], [self.x_g.size(0)]])
        return super().__inc__(key, value, *args, **kwargs)
