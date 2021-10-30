import torch.optim as optim
import torchvision
from torchvision.models import vgg
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
from torch.quantization import QuantStub, DeQuantStub
import torch
from torchvision.transforms.transforms import Normalize
from tqdm import tqdm
import torch.quantization

vgg16 = models.vgg16()


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


transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomGrayscale(),
    transforms.ToTensor(),
    Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

transform_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# test = vgg16Quan()
dataset_train = torchvision.datasets.CIFAR100("~/datasets/", train=True, transform=transform_train)
train_sampler = torch.utils.data.RandomSampler(dataset_train)
data_loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=500, sampler=train_sampler)

dataset_test = torchvision.datasets.CIFAR100("~/datasets/", train=False, transform=transform_test)
test_sampler = torch.utils.data.RandomSampler(dataset_test)
data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=500, sampler=test_sampler)

optimizer = optim.Adam(vgg16.parameters(), lr=0.0001)
criteria = nn.CrossEntropyLoss()

vgg16 = vgg16.cuda()


def train():
    path = "vgg16_"
    for epoch in range(100):
        running_loss = 0.0
        i = 0
        for X, target in tqdm(data_loader_train):
            X, target = X.cuda(), target.cuda()
            optimizer.zero_grad()
            output = vgg16(X)
            loss = criteria(output, target.long())
            loss.backward()
            optimizer.step()

            i += 1
            if i % 100 == 99:  # print every 500 mini-batches
                print('[%d, %5d] loss: %.4f' %
                      (epoch, i, running_loss / 100))
                running_loss = 0.0
                _, predicted = torch.max(output.data, 1)
                total = target.size(0)
                correct = (predicted == target).sum().item()
                print('Accuracy of the network on the %d train images: %.3f %%' % (total,
                                                                                   100.0 * correct / total))
                if epoch % 10 == 9:
                    torch.save({'epoch': epoch,
                                'model_state_dict': vgg16.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'loss': loss
                                }, path+str(epoch)+".pth")

            running_loss += loss.item()


def test(path):
    train_dict = torch.load(path)
    vgg16.load_state_dict(train_dict["model_state_dict"])
    vgg16.eval()
    vgg16.cuda()

    total = 0
    correct = 0
    with torch.no_grad():
        for X, target in tqdm(data_loader_test):
            X, target = X.cuda(), target.cuda()
            output = vgg16(X)

            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    print(correct/total*100.0)


def test_quant(path):
    vggQuant = vgg16Quan().cpu()
    train_dict = torch.load(path)
    vggQuant.vgg16.load_state_dict(train_dict["model_state_dict"])
    vggQuant.eval()
    vggQuant.cpu()

    vggQuant.qconfig = torch.quantization.default_qconfig
    print(vggQuant.qconfig)
    # input("Look at once")
    torch.quantization.prepare(vggQuant, inplace=True)
    with torch.no_grad():
        cnt = 0
        for X, target in tqdm(data_loader_train):
            output = vggQuant(X)
            cnt += 1
            if cnt >= 20:
                break
    torch.quantization.convert(vggQuant, inplace=True)
    torch.save({"state_dict":vggQuant.state_dict()},"vgg16Quant.state_dict")
    total = 0
    correct = 0
    with torch.no_grad():
        for X, target in tqdm(data_loader_test):
            output = vggQuant(X)

            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    print(correct/total*100.0)
    # torch.jit.save(torch.jit.script(vggQuant),"vggQuant.pth")
    return vggQuant



test("vgg16_99.pth")
quantModel = test_quant("vgg16_99.pth")
