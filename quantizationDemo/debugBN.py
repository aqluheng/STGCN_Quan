import os
from os.path import abspath, dirname, join
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import FakeData
import torchvision.transforms as transforms
from torch.quantization import fuse_modules

use_relu = False
disable_single_bns = False

class PReLU_Quantized(nn.Module):
    def __init__(self, prelu_object):
        super().__init__()
        self.prelu_weight = prelu_object.weight
        self.weight = self.prelu_weight
        self.quantized_op = nn.quantized.FloatFunctional()
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, inputs):
        # inputs = max(0, inputs) + alpha * min(0, inputs) 
        # this is how we do it 
        # pos = torch.relu(inputs)
        # neg = -alpha * torch.relu(-inputs)
        # res3 = pos + neg
        self.weight = self.quant(self.weight)
        weight_min_res = self.quantized_op.mul(-self.weight, torch.relu(-inputs))
        inputs = self.quantized_op.add(torch.relu(inputs), weight_min_res)
        inputs = self.dequant(inputs)
        self.weight = self.dequant(self.weight)
        return inputs

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.add_relu = torch.nn.quantized.FloatFunctional()

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        # out += residual
        # out = self.relu(out)
        out = self.add_relu.add_relu(out, residual)
        return out

    def fuse_model(self):
        torch.quantization.fuse_modules(self, [['conv1', 'bn1', 'relu'],
                                               ['conv2', 'bn2']], inplace=True)
        if self.downsample:
            torch.quantization.fuse_modules(self.downsample, ['0', '1'], inplace=True)

class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu1 = nn.ReLU(inplace=False)
        self.relu2 = nn.ReLU(inplace=False)
        self.downsample = downsample
        self.stride = stride
        self.skip_add_relu = nn.quantized.FloatFunctional()

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        # out += residual
        # out = self.relu(out)
        out = self.skip_add_relu.add_relu(out, residual)
        return out

    def fuse_model(self):
        fuse_modules(self, [['conv1', 'bn1', 'relu1'],
                            ['conv2', 'bn2', 'relu2'],
                            ['conv3', 'bn3']], inplace=True)
        if self.downsample:
            torch.quantization.fuse_modules(self.downsample, ['0', '1'], inplace=True)

class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.mult_xy = nn.quantized.FloatFunctional()
        self.fc = nn.Sequential(nn.Linear(channel, channel // reduction),
                                nn.PReLU(),
                                nn.Linear(channel // reduction, channel),
                                nn.Sigmoid())
        self.fc1 = self.fc[0]
        self.prelu = self.fc[1]
        self.fc2 = self.fc[2]
        self.sigmoid = self.fc[3]
        self.prelu_q = PReLU_Quantized(self.prelu)
        if use_relu:
            self.prelu_q_or_relu = torch.relu
        else:
            self.prelu_q_or_relu = self.prelu_q

    def forward(self, x):
        # print(f'<inside se forward:>')
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        # y = self.fc(y).view(b, c, 1, 1)
        y = self.fc1(y)
        y = self.prelu_q_or_relu(y)
        y = self.fc2(y)
        y = self.sigmoid(y).view(b, c, 1, 1)
        # print('--------------------------')
        # out = x*y 
        out = self.mult_xy.mul(x, y)
        return out

class IRBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None, use_se=True):
        super().__init__()
        self.bn0 = nn.BatchNorm2d(inplanes)
        if disable_single_bns:
            self.bn0_or_identity = torch.nn.Identity()
        else:
            self.bn0_or_identity = self.bn0

        self.conv1 = conv3x3(inplanes, inplanes)
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.prelu = nn.PReLU()
        self.prelu_q = PReLU_Quantized(self.prelu)
        
        if use_relu:
            self.prelu_q_or_relu = torch.relu
        else:
            self.prelu_q_or_relu = self.prelu_q

        self.conv2 = conv3x3(inplanes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.use_se = use_se
        # if self.use_se:
        self.se = SEBlock(planes)
        self.add_residual = nn.quantized.FloatFunctional()

    def forward(self, x):
        residual = x
        # TODO:
        # this needs to be quantized as well!
        out = self.bn0_or_identity(x)

        out = self.conv1(out)
        out = self.bn1(out)
        # out = self.prelu(out)
        out = self.prelu_q_or_relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        if self.use_se:
            out = self.se(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        # out += residual
        # out = self.prelu(out)
        out = self.prelu_q_or_relu(out)
        # we may need to change prelu into relu and instead of add, use add_relu here
        out = self.add_residual.add(out, residual)
        return out

    def fuse_model(self):
        fuse_modules(self, [# ['bn0'],
                            ['conv1', 'bn1'],
                            ['conv2', 'bn2']], inplace=True)
        if self.downsample:
            torch.quantization.fuse_modules(self.downsample, ['0', '1'], inplace=True)

class ResNet(nn.Module):

    def __init__(self, block, layers, use_se=True):
        self.inplanes = 64
        self.use_se = use_se
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.prelu = nn.PReLU()
        self.prelu_q = PReLU_Quantized(self.prelu)
        # This is to only get rid of the unimplemented CPUQuantization type error
        # when we use PReLU_Quantized during test time
        if use_relu:
            self.prelu_q_or_relu = torch.relu
        else:
             self.prelu_q_or_relu = self.prelu_q

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.bn2 = nn.BatchNorm2d(512)
        # This is to get around the single BatchNorms not getting fused and thus causing 
        # a RuntimeError: Could not run 'aten::native_batch_norm' with arguments from the 'QuantizedCPU' backend.
        # 'aten::native_batch_norm' is only available for these backends: [CPU, MkldnnCPU, BackendSelect, Named, Autograd, Profiler, Tracer, Autocast, Batched].
        # during test time
        if disable_single_bns:
            self.bn2_or_identity = torch.nn.Identity()
        else:
            self.bn2_or_identity = self.bn2

        self.dropout = nn.Dropout()
        self.fc = nn.Linear(512 * 7 * 7, 512)
        self.bn3 = nn.BatchNorm1d(512)
        if disable_single_bns:
            self.bn3_or_identity = torch.nn.Identity()
        else:
            self.bn3_or_identity = self.bn3
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, use_se=self.use_se))
        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, use_se=self.use_se))

        return nn.Sequential(*layers)

    def forward(self, x):
        
        x = self.quant(x)
        x = self.conv1(x)
        # TODO: single bn needs to be fused
        x = self.bn1(x)

        # x = self.prelu(x)
        x = self.prelu_q_or_relu(x)

        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.bn2_or_identity(x)
        x = self.dropout(x)
        # x = x.view(x.size(0), -1)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)
        # TODO: single bn needs to be fused
        x = self.bn3_or_identity(x)
        x = self.dequant(x)
        return x

    def fuse_model(self):
        r"""Fuse conv/bn/relu modules in resnet models
        Fuse conv+bn+relu/ Conv+relu/conv+Bn modules to prepare for quantization.
        Model is modified in place.  Note that this operation does not change numerics
        and the model after modification is in floating point
        """
        fuse_modules(self, ['conv1', 'bn1'], inplace=True)
        for m in self.modules():
            if type(m) == Bottleneck or type(m) == BasicBlock or type(m) == IRBlock:
                m.fuse_model()

class SimpleNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=10, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(10)
        self.relu1 = nn.ReLU()

        self.prelu_q = PReLU_Quantized(nn.PReLU())
        self.bn = nn.BatchNorm2d(10)

        self.prelu_q_or_relu = torch.relu if use_relu else self.prelu_q
        self.bn_or_identity = nn.Identity() if disable_single_bns else self.bn    

        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()
    
    def forward(self, x):
        x = self.quant(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.prelu_q_or_relu(x)
        x = self.bn_or_identity(x)

        x = self.dequant(x)
        return x

def resnet18(use_se=True, **kwargs):
    return ResNet(IRBlock, [2, 2, 2, 2], use_se=use_se, **kwargs)

def print_size_of_model(model):
    torch.save(model.state_dict(), "temp.p")
    print('Size (MB):', os.path.getsize("temp.p")/1e6)
    os.remove('temp.p')

def evaluate(model, data_loader, eval_batches):
    model.eval()
    with torch.no_grad():
        for i, (image, target) in enumerate(data_loader):
            features = model(image)
            print(f'{i})feature dims: {features.shape}')
            if i >= eval_batches:
                return

def load_quantized(model, quantized_checkpoint_file_path):
    model.eval()
    if type(model) == ResNet:
        model.fuse_model()
    # Specify quantization configuration
    model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    torch.quantization.prepare(model, inplace=True)
    # Convert to quantized model
    torch.quantization.convert(model, inplace=True)
    checkpoint = torch.load(quantized_checkpoint_file_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint)
    print_size_of_model(model)
    return model

def test_the_model(model, dtloader):
    current_dir = abspath(dirname(__file__))
    model = load_quantized(model, join(current_dir, 'data', 'model_quantized_jit.pth'))
    model.eval()
    img, _ = next(iter(dtloader))
    embd1 = model(img)

def quantize_model(model, dtloader):
    calibration_batches = 10 
    saved_model_dir = 'data'
    scripted_quantized_model_file = 'model_quantized_jit.pth'
    # model = resnet18()
    model.eval()
    if type(model) == ResNet:
        model.fuse_model()
    print_size_of_model(model)
    model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    print(model.qconfig)
    torch.quantization.prepare(model, inplace=True)

    print(f'Model after fusion(prepared): {model}')

    # Calibrate first
    print('Post Training Quantization Prepare: Inserting Observers')
    print('\n Inverted Residual Block:After observer insertion \n\n', model.conv1)

    # Calibrate with the training set
    evaluate(model, dtloader, eval_batches=calibration_batches)
    print('Post Training Quantization: Calibration done')

    # Convert to quantized model
    torch.quantization.convert(model, inplace=True)
    print('Post Training Quantization: Convert done')
    print('\n Inverted Residual Block: After fusion and quantization, note fused modules: \n\n', model.conv1)

    print("Size of model after quantization")
    print_size_of_model(model)
    script = torch.jit.script(model)
    path_tosave = join(dirname(abspath(__file__)), saved_model_dir, scripted_quantized_model_file)
    print(f'path to save: {path_tosave}')
    with open(path_tosave, 'wb') as f:
        torch.save(model.state_dict(), f)

    print(f'model after quantization (prepared and converted:) {model}')
    # torch.jit.save(script, path_tosave)

dataset = FakeData(1000, image_size=(3, 112, 112), num_classes=5, transform=transforms.ToTensor())
data_loader = DataLoader(dataset, batch_size=1)

# quantize the model 
model = resnet18()
# model = SimpleNetwork()
quantize_model(model, data_loader)

# and load and test the quantized model
model = resnet18()
# model = SimpleNetwork()
test_the_model(model, data_loader)

