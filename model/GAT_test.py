import torch

from torch_geometric.nn import GATConv
from torch_geometric.data import Data


# data1 = Data(x=torch.tensor([[1,2,3],[4,5,6],[7,8,9]], dtype=torch.float), edge_index=torch.tensor([[0,1,2], [1,2,0]]))
#
# gat = GATConv(3, 3, heads=2)
# out = gat(data1.x, data1.edge_index)
# print(out)


weight = torch.load('/sdd/yujunshuai/model/en_glove_vector/en_weights.pt')
print(weight)