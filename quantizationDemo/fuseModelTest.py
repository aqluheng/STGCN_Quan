import torch
from torch.functional import Tensor
import torch.nn as nn
from torch.quantization import QuantStub,DeQuantStub
class CivilNet(nn.Module):
    def __init__(self):
        super(CivilNet, self).__init__()
        self.conv = nn.Conv2d(1, 1, kernel_size=1, stride=1, padding=0, groups=1, bias=False)
        self.quant = QuantStub()
        self.dequant = DeQuantStub()

    def forward(self, x0, printVal = False):
        x1 = self.quant(x0)
        x2 = self.conv(x1)
        x3 = self.dequant(x2)
        if printVal:
            print("X0:", x0)
            print("X1 = quant(X0):", x1, int(x1.int_repr()))
            print("X2 = conv(X1):", x2, int(x2.int_repr()))
            print("X3 = dequant(X2):", x3)

        return x3

net = CivilNet()
inputTensor = torch.ones(1,1,1,1)
net.conv.load_state_dict({"weight":inputTensor*(-8)})
net.qconfig = torch.quantization.get_default_qconfig('fbgemm')
torch.quantization.prepare(net,inplace = True)
for i in range(10):
    net(i*inputTensor)
torch.quantization.convert(net,inplace = True)
print("\n\n\n")
print("Quant参数",net.quant)
print("卷积参数",net.conv)
print("卷积核量化",net.conv.weight(),net.conv.weight().int_repr())

net(inputTensor*0.0709,True)