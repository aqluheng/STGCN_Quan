import torch
import torchvision.transforms as transforms
import torchvision
from tqdm import tqdm


transform_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

dataset_test = torchvision.datasets.CIFAR100(".", train=False, transform=transform_test)
test_sampler = torch.utils.data.RandomSampler(dataset_test)
data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=500, sampler=test_sampler)

vggQuant = torch.jit.load("quantVGG.pth")
vggQuant.eval()


total = 0
correct = 0
with torch.no_grad():
    for X, target in tqdm(data_loader_test):
        output = vggQuant(X)

        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()
print(correct/total*100.0)