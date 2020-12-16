import torch
import torch.nn as nn
import torch.nn.functional as F

from LocalSearch import *


#生成每一次rollout的state
def generate_state(state=None, action=0, reward=0, delta_min=0, delta=0):
    if state:
        state = [action, reward, delta_min, delta]
    else:
        state = [action, reward, delta_min, delta]
    return state


class embed_net(nn.Module):
    #定义嵌入网络结构
    def __init__(self):
        super(embed_net, self).__init__()
        
    
    def forward():