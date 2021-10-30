import torch
import torchvision.transforms as transforms
import torchvision
from tqdm import tqdm
from torch.quantization import QuantStub, DeQuantStub
import torch.nn as nn
import torchvision.models as models


transform_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

dataset_test = torchvision.datasets.CIFAR100("~/datasets/", train=False, transform=transform_test)
test_sampler = torch.utils.data.RandomSampler(dataset_test)
data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=500, sampler=test_sampler)

# vggQuant = torch.jit.load("quantVGG.pth")
# vggQuant.eval()


class vgg16Quan(nn.Module):
    def __init__(self):
        super(vgg16Quan, self).__init__()
        self.vgg16 = models.vgg16()
        self.quant = QuantStub()
        self.dequant = DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self.vgg16(x)
        x = self.dequant(x)
        return x

vggQuant = vgg16Quan()
vggQuant.eval()
vggQuant.qconfig = torch.quantization.default_qconfig
torch.quantization.prepare(vggQuant, inplace=True)
torch.quantization.convert(vggQuant, inplace=True)
vggQuant.load_state_dict(torch.load("vgg16Quant.state_dict")["state_dict"])

total = 0
correct = 0
with torch.no_grad():
    for X, target in tqdm(data_loader_test):
        output = vggQuant(X)

        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()
print(correct/total*100.0)