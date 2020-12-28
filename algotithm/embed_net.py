import torch
import torch.nn as nn
import torch.nn.functional as F


'''
class SpecialspmmFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, indices, values, shape, b):
        assert indices.requires_grad == False
        a = torch.sparse_coo_tensor(indices, values, shape)
        ctx.save_for_backward(a, b)
        ctx.N = shape[0]
        return torch.matmul(a, b)
    
    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        grad_values = grad_b = None
        if ctx.needs_input_grad[1]:
            grad_a_dense = grad_output.matmul(b.t())
            edge_idx = a._indices()[0, :] * ctx.N + a._indices()[1, :]
            grad_values = grad_a_dense.view(-1)[edge_idx]
        if ctx.needs_input_grad[3]:
            grad_b = a.t().matmul(grad_output)
        return None, grad_values, None, grad_b

class Specialspmm(nn.Module):
    def forward(self, indices, values, shape, b):
        return SpecialspmmFunction.apply(indices, values, shape, b)
'''


class GraphAttentionLayer(nn.Module):
    #定义嵌入网络的结构
    def __init__(self, input_dim, output_dim, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = input_dim    #输入特征数
        self.out_features = output_dim  #输出特征数
        self.dropout = dropout          #防止过拟合
        self.W = nn.Parameter(torch.empty((self.in_features, self.out_features)))
        nn.init.xavier_uniform_(self.W, gain = 1.414)
        #self.W.gain = nn.init.calculate_gain(nonlinearity=Relu, param=None)
        self.a = nn.Parameter(torch.empty(size=(2*output_dim, 1)))
        nn.init.xavier_uniform_(self.a, gain= 1.414)
        self.LeakyReLU = nn.LeakyReLU(alpha)
        self.concat = concat

    def forward(self, feature, adjacency_M):
        Wh = torch.mm(feature, self.W)
        # shape = (num_train_points + 1, out_features)
        a_input = self._prepare_attentional_mechanism_input(Wh)
        # shape = (num_train_points + 1, num_train_points + 1, out_features)
        e = self.LeakyReLU(torch.matmul(a_input, self.a).squeeze(2))
        # shape = (num_train_points + 1, num_train_points + 1)
        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adjacency_M > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)
        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime
        # h-> shape = (num_train_points + 1, out_features)

    def _prepare_attentional_mechanism_input(self, Wh):
        N = Wh.size()[0]
        # N是节点个数
        Wh_repeated_in_chunks = Wh.repeat_interleave(N, dim=0)
        # 将Wh每行复制N个. shape = (N * N, out_features)
        # h1 ...  h1    h2  ...   h2           h3   ...  h3     hN   ...  hN
        Wh_repeated_alternating = Wh.repeat(N, 1)
        # 将Wh每行复制N倍，每列复制1倍（不变），按序复制. shape = (N * N, out_features)
        # h1 h2...hN    h1 h2 ... hN           h1 h2 ... hN     h1 h2 ... hN
        combinations_matrix = torch.cat([Wh_repeated_in_chunks, Wh_repeated_alternating], dim=1)
        #shape = (N * N, 2 * out_features)
        return combinations_matrix.view(N, N, 2 * self.out_features)
        # [N * [N * [out_features * 2]]]
        # h1h1 h1h2 h1h3...h1hN ... hNh1 hNh2 hNh3...hNhN


class EMBED_NET(nn.Module):
    def __init__(self, features_dim, output_dim, nclass, dropout, alpha, n_heads):
        super(EMBED_NET, self).__init__()
        self.output_dim = output_dim
        self.dropout = dropout
        self.attentions = [GraphAttentionLayer(features_dim, output_dim, dropout=dropout, alpha=alpha) for _ in range(n_heads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)
        #self.out_att = GraphAttentionLayer(output_dim * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, feature, adjacency_M):
        feature = F.dropout(feature, self.dropout, training=self.training)
        new_feature = torch.zeros(size=(feature.size()[0], self.output_dim))
        multi_attention = [att(feature, adjacency_M) for att in self.attentions]
        #多头注意力机制，将每个自注意力特征都存储在list当中
        for h in multi_attention:
            h = F.dropout(h, self.dropout, training=self.training)
            new_feature += h
        new_feature /= len(multi_attention)
        new_feature = F.log_softmax(new_feature, dim=1)
        # 非线性函数, shape = (num_train_points + 1, output_dim)
        return torch.mean(new_feature, dim=1)
        # 对每个节点特征求均值(按列求均值)得solution feature, shape = (output_dim)