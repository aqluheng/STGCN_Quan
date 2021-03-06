import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from time import time
from torch.nn.quantized.modules import DeQuantize

from torch.tensor import Tensor

from mmskeleton.ops.st_gcn import ConvTemporalGraphical, Graph
from torch.quantization import QuantStub, DeQuantStub


def zero(x):
    return 0


def zeroTensor(x):
    x[:] = 0
    return x


def iden(x):
    return x


CountNum = 0
timeGCN = 0
timeTCN = 0
timeAGG = 0
timeCOM = 0
TotalGCNTime = 0
# class myConv(nn.Conv2d):
#     def __init__(self, in_channels: int, out_channels: int, kernel_size, stride = 1, padding = 0, dilation = 1, groups: int = 1, bias: bool = True, padding_mode: str = 'zeros'):
#         super().__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias, padding_mode=padding_mode)
#         self.quant = QuantStub()
#         self.dequant = DeQuantStub()
#     def forward(self, input: Tensor) -> Tensor:
#         return self.dequant(super().forward(self.quant(input)))


class ST_GCN_18(nn.Module):
    r"""Spatial temporal graph convolutional networks.

    Args:
        in_channels (int): Number of channels in the input data
        num_class (int): Number of classes for the classification task
        graph_cfg (dict): The arguments for building the graph
        edge_importance_weighting (bool): If ``True``, adds a learnable
            importance weighting to the edges of the graph
        **kwargs (optional): Other parameters for graph convolution units

    Shape:
        - Input: :math:`(N, in_channels, T_{in}, V_{in}, M_{in})`
        - Output: :math:`(N, num_class)` where
            :math:`N` is a batch size,
            :math:`T_{in}` is a length of input sequence,
            :math:`V_{in}` is the number of graph nodes,
            :math:`M_{in}` is the number of instance in a frame.
    """

    def __init__(self,
                 in_channels,
                 num_class,
                 graph_cfg,
                 edge_importance_weighting=True,
                 data_bn=True,
                 **kwargs):
        super().__init__()
        # load graph
        self.graph = Graph(**graph_cfg)
        A = torch.tensor(self.graph.A,
                         dtype=torch.float32,
                         requires_grad=False)
        self.register_buffer('A', A)

        # build networks
        spatial_kernel_size = A.size(0)
        temporal_kernel_size = 9
        kernel_size = (temporal_kernel_size, spatial_kernel_size)
        # self.data_bn = nn.BatchNorm1d(in_channels * A.size(1)) if data_bn else iden
        self.data_bn = nn.BatchNorm2d(in_channels * A.size(1)) if data_bn else iden
        kwargs0 = {k: v for k, v in kwargs.items() if k != 'dropout'}
        self.st_gcn_networks = nn.ModuleList((
            st_gcn_block(in_channels,
                         64,
                         kernel_size,
                         1,
                         residual=False,
                         **kwargs0),
            st_gcn_block(64, 64, kernel_size, 1, **kwargs),
            st_gcn_block(64, 64, kernel_size, 1, **kwargs),
            st_gcn_block(64, 64, kernel_size, 1, **kwargs),
            st_gcn_block(64, 128, kernel_size, 2, **kwargs),
            st_gcn_block(128, 128, kernel_size, 1, **kwargs),
            st_gcn_block(128, 128, kernel_size, 1, **kwargs),
            st_gcn_block(128, 256, kernel_size, 2, **kwargs),
            st_gcn_block(256, 256, kernel_size, 1, **kwargs),
            st_gcn_block(256, 256, kernel_size, 1, **kwargs),
        ))

        # initialize parameters for edge importance weighting
        if edge_importance_weighting:
            self.edge_importance = nn.ParameterList([
                nn.Parameter(torch.ones(self.A.size()))
                for i in self.st_gcn_networks
            ])
        else:
            self.edge_importance = [1] * len(self.st_gcn_networks)

        # fcn for prediction
        self.quant = QuantStub()
        self.dequant = DeQuantStub()
        self.fcn = nn.Conv2d(256, num_class, kernel_size=1)

    def forward(self, x):
        # data normalization
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous()
        x = x.view(N * M, V * C, T)
        # x = self.data_bn(x)
        x = self.data_bn(x.unsqueeze(-1)).squeeze(-1)
        x = x.view(N, M, V, C, T)
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(N * M, C, T, V)

        # global TotalGCNTime
        # TotalGCNTime -= time()
        # forward
        for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
            x, _ = gcn(x, self.A * importance)
        # TotalGCNTime += time()

        # global pooling
        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(N, M, -1, 1, 1).mean(dim=1)

        # prediction
        x = self.quant(x)
        x = self.fcn(x)
        x = self.dequant(x)
        x = x.view(x.size(0), -1)
        exit()
        # global CountNum
        # CountNum += 1
        # if CountNum == 620:
        #     global timeGCN
        #     global timeTCN
        #     global timeAGG
        #     global timeCOM
        #     print("GCNTime",timeGCN)
        #     print("TCNTime",timeTCN)
        #     print("AGGTime",timeAGG)
        #     print("CombineTime",timeCOM)
        return x

    def extract_feature(self, x):

        # data normalization
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous()
        x = x.view(N * M, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T)
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(N * M, C, T, V)

        # forward
        for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
            x, _ = gcn(x, self.A * importance)

        _, c, t, v = x.size()
        feature = x.view(N, M, c, t, v).permute(0, 2, 3, 4, 1)

        # prediction
        x = self.fcn(x)
        output = x.view(N, M, -1, t, v).permute(0, 2, 3, 4, 1)

        return output, feature


class st_gcn_block(nn.Module):
    r"""Applies a spatial temporal graph convolution over an input graph sequence.

    Args:
        in_channels (int): Number of channels in the input sequence data
        out_channels (int): Number of channels produced by the convolution
        kernel_size (tuple): Size of the temporal convolving kernel and graph convolving kernel
        stride (int, optional): Stride of the temporal convolution. Default: 1
        dropout (int, optional): Dropout rate of the final output. Default: 0
        residual (bool, optional): If ``True``, applies a residual mechanism. Default: ``True``

    Shape:
        - Input[0]: Input graph sequence in :math:`(N, in_channels, T_{in}, V)` format
        - Input[1]: Input graph adjacency matrix in :math:`(K, V, V)` format
        - Output[0]: Outpu graph sequence in :math:`(N, out_channels, T_{out}, V)` format
        - Output[1]: Graph adjacency matrix for output data in :math:`(K, V, V)` format

        where
            :math:`N` is a batch size,
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]`,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes.

    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dropout=0,
                 residual=True):
        super().__init__()

        assert len(kernel_size) == 2
        assert kernel_size[0] % 2 == 1
        padding = ((kernel_size[0] - 1) // 2, 0)

        self.gcn = ConvTemporalGraphical(in_channels, out_channels,
                                         kernel_size[1])

        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            # QuantStub(),
            nn.Conv2d(
                out_channels,
                out_channels,
                (kernel_size[0], 1),
                (stride, 1),
                padding,
            ),
            # DeQuantStub(),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True),
        )

        if not residual:
            self.residual = zero

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = iden

        else:
            self.residual = nn.Sequential(
                # QuantStub(),
                nn.Conv2d(in_channels,
                          out_channels,
                          kernel_size=1,
                          stride=(stride, 1)),
                # DeQuantStub(),
                nn.BatchNorm2d(out_channels),
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, A):
        def computeDiff(x1, x2):
            diff = x1 - x2
            print(abs(diff.max()),(diff**2).sum())
        res = self.residual(x)
        x1 = torch.rand(x.shape) * x.max() / 2
        x2 = x - x1
        computeDiff(x, x1+x2)
        x, A = self.gcn(x, A)
        x1, A = self.gcn(x1, A)
        x2, A = self.gcn(x2, A, diff = True)
        computeDiff(x, x1+x2)
        
        


        x = self.tcn(x) + res
        return self.relu(x), A
    '''
    def forward(self, x, A):
        res = self.residual(x)
        x, A = self.gcn(x, A)
        x = self.tcn(x) + res
        return self.relu(x), A
    '''
    
    '''
    def forward(self, x, A):
        # global timeGCN
        # global timeTCN
        # global timeAGG
        # global timeCOM
        res = self.residual(x)

        # torch.cuda.synchronize()
        # timeGCN -= time()
        # x, A, tmpComTime, tmpAggTime = self.gcn(x, A)
        x, A = self.gcn(x, A)
        # torch.cuda.synchronize()
        # timeGCN += time()
        # timeAGG += tmpAggTime
        # timeCOM += tmpComTime
        # torch.cuda.synchronize()
        # timeTCN -= time()
        x = self.tcn(x)
        x = x + res
        # torch.cuda.synchronize()
        # timeTCN += time()

        return self.relu(x), A
    '''
