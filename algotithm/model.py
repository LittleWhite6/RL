import torch
import torch.nn as nn
import torch.nn.functional as F

from cvrp import *


class GAT_net(nn.Module):
    #定义嵌入网络的结构
    def __init__(self, feature, adjacency_M, input_dim, out_dim):
        super(embed_net, self).__init__()
        

    
    def forward(self, feature):
        pass