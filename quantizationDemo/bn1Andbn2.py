import torch
import torch.nn as nn

bn1d = nn.BatchNorm1d(1)
bn2d = nn.BatchNorm2d(1)

data = torch.randn(4, 1, 16)
res1d = bn1d(data)

res2d = bn2d(data.unsqueeze(-1)).squeeze(-1)
print(torch.allclose(res1d, res2d))