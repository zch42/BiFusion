import torch
import torch.nn.functional as F
from torch.nn import Parameter, Linear
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import softmax

from utils.weight_inits import glorot, zeros


class BipartiteGAT(MessagePassing):

    def __init__(self, in_channels_i, in_channels_j, out_channels, heads=1,
                 attention_concat=False, multi_head_concat=False,
                 negative_slope=0.2, dropout=0.0, bias=True, **kwargs):
        """
        :param in_channels: Size of each input sample.
        :param out_channels: Size of each output sample.
        :param heads: Number of multi-head-attentions.
        :param attention_concat: If set to False, the attentions are only based on one side.
        :param multi_head_concat: If set to False, the multi-head attentions are averaged instead of concatenated.
        :param negative_slope: LeakyReLU angle of the negative.
        :param dropout: Dropout probability of the normalized attention coefficients which exposes each node to a
                        stochastically sampled neighborhood during training.
        :param bias: If set to False, the layer will not learn an additive bias.
        :param kwargs: Additional arguments of `torch_geometric.nn.conv.MessagePassing.
        """
        super(BipartiteGAT, self).__init__(aggr='add', **kwargs)

        self.in_channels_i = in_channels_i
        self.in_channels_j = in_channels_j
        self.out_channels = out_channels
        self.heads = heads
        self.attention_concat = attention_concat
        self.multi_head_concat = multi_head_concat
        self.negative_slope = negative_slope
        self.dropout = dropout

        if attention_concat:
            self.att = Parameter(torch.Tensor(1, heads, 2 * out_channels))
        else:
            self.att = Parameter(torch.Tensor(1, heads, out_channels))

        self.mlp_i = Linear(in_channels_i, heads * out_channels)
        self.mlp_j = Linear(in_channels_j, heads * out_channels)

        if bias and multi_head_concat:
            self.bias = Parameter(torch.Tensor(heads * out_channels))
        elif bias and not multi_head_concat:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.att)
        zeros(self.bias)

    def forward(self, x, edge_index, size=None):
        """"""
        # feature dimension alignment
        x = (self.mlp_i(x[0]), self.mlp_j(x[1]))

        propagate_result = self.propagate(edge_index, size=size, x=x)
        index = 1 if self.flow == "source_to_target" else 0
        final_result = propagate_result + x[index]

        return final_result

    def message(self, edge_index_i, x_i, x_j, size_i):
        # Compute attention coefficients.
        x_j = x_j.view(-1, self.heads, self.out_channels)
        if self.attention_concat:
            x_i = x_i.view(-1, self.heads, self.out_channels)
            alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1)
        else:
            alpha = (x_j * self.att).sum(dim=-1)

        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, edge_index_i, size_i)

        # Sample attention coefficients stochastically.
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        return x_j * alpha.view(-1, self.heads, 1)

    def update(self, aggr_out):
        if self.multi_head_concat is True:
            aggr_out = aggr_out.view(-1, self.heads * self.out_channels)
        else:
            aggr_out = aggr_out.mean(dim=1)

        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)
