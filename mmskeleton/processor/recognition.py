from collections import OrderedDict
import torch
import logging
import numpy as np
from mmskeleton.utils import call_obj, import_obj, load_checkpoint
from mmcv.runner import Runner
from mmcv import Config, ProgressBar
from mmcv.parallel import MMDataParallel
from time import time
import os
from torch.quantization import QuantStub, DeQuantStub
import torch.nn as nn
from torch.quantization import prepare,convert

def print_size_of_model(model, label=""):
    torch.save(model.state_dict(), "temp.p")
    size=os.path.getsize("temp.p")
    print("model: ",label,' \t','Size (KB):', size/1e3)
    os.remove('temp.p')
    return size

class QuantModel(nn.Module):
    def __init__(self,model):
        super().__init__()
        self.model = model
        self.quant = QuantStub()
        self.dequant = DeQuantStub()

    def forward(self,X):
        X = self.quant(X)
        X = self.model(X)
        X = self.dequant(X)

def test(model_cfg,
         dataset_cfg,
         checkpoint,
         batch_size=None,
         gpu_batch_size=None,
         gpus=-1,
         workers=4):

    # calculate batch size
    if gpus < 0:
        gpus = torch.cuda.device_count()
    if (batch_size is None) and (gpu_batch_size is not None):
        batch_size = gpu_batch_size * gpus
    assert batch_size is not None, 'Please appoint batch_size or gpu_batch_size.'

    dataset = call_obj(**dataset_cfg)
    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              num_workers=workers)

    # put model on gpus
    if isinstance(model_cfg, list):
        model = [call_obj(**c) for c in model_cfg]
        model = torch.nn.Sequential(*model)
    else:
        model = call_obj(**model_cfg)

    # modelQuant =  torch.jit.load("quantSTGCN.pth")
    # results = []
    # labels = []
    # evaluateTime = 0
    # prog_bar = ProgressBar(len(dataset))

    # for data, label in data_loader:
    #     data = data.cpu()
    #     evaluateTime -= time()
    #     with torch.no_grad():
    #         output = modelQuant(data).data.cpu().numpy()
    #     evaluateTime += time()

    #     results.append(output)
    #     labels.append(label)
    #     for i in range(len(data)):
    #         prog_bar.update()
    # exit()
    modelQuant = QuantModel(model)
    modelQuant.eval()
    modelQuant.qconfig = torch.quantization.default_qconfig
    load_checkpoint(modelQuant.model, checkpoint, map_location='cpu')

    prepare(modelQuant,inplace=True)
    cnt = 0
    for data,label in data_loader:
        with torch.no_grad():
            output = modelQuant(data)
            cnt += 1
            print(cnt)
            if cnt >= 20:
                break
    convert(modelQuant,inplace=True)
    # torch.jit.save(torch.jit.script(modelQuant),"quantSTGCN.pth")
    # exit()
    modelQuant = MMDataParallel(modelQuant, device_ids=range(gpus)).cpu()

    results = []
    labels = []
    evaluateTime = 0
    prog_bar = ProgressBar(len(dataset))

    for data, label in data_loader:
        evaluateTime -= time()
        with torch.no_grad():
            output = model(data)
            output = model(data).data.cpu().numpy()
        evaluateTime += time()

        results.append(output)
        labels.append(label)
        for i in range(len(data)):
            prog_bar.update()

    results = np.concatenate(results)
    labels = np.concatenate(labels)

    print('Evalute Time:',evaluateTime)
    print('Top 1: {:.2f}%'.format(100 * topk_accuracy(results, labels, 1)))
    print('Top 5: {:.2f}%'.format(100 * topk_accuracy(results, labels, 5)))


def train(
        work_dir,
        model_cfg,
        loss_cfg,
        dataset_cfg,
        optimizer_cfg,
        total_epochs,
        training_hooks,
        batch_size=None,
        gpu_batch_size=None,
        workflow=[('train', 1)],
        gpus=-1,
        log_level=0,
        workers=4,
        resume_from=None,
        load_from=None,
):

    # calculate batch size
    if gpus < 0:
        gpus = torch.cuda.device_count()
    if (batch_size is None) and (gpu_batch_size is not None):
        batch_size = gpu_batch_size * gpus
    assert batch_size is not None, 'Please appoint batch_size or gpu_batch_size.'

    # prepare data loaders
    if isinstance(dataset_cfg, dict):
        dataset_cfg = [dataset_cfg]
    data_loaders = [
        torch.utils.data.DataLoader(dataset=call_obj(**d),
                                    batch_size=batch_size,
                                    shuffle=True,
                                    num_workers=workers,
                                    drop_last=True) for d in dataset_cfg
    ]

    # put model on gpus
    if isinstance(model_cfg, list):
        model = [call_obj(**c) for c in model_cfg]
        model = torch.nn.Sequential(*model)
    else:
        model = call_obj(**model_cfg)
    model.apply(weights_init)

    model = MMDataParallel(model, device_ids=range(gpus)).cuda()
    loss = call_obj(**loss_cfg)

    # build runner
    optimizer = call_obj(params=model.parameters(), **optimizer_cfg)
    runner = Runner(model, batch_processor, optimizer, work_dir, log_level)
    runner.register_training_hooks(**training_hooks)

    if resume_from:
        runner.resume(resume_from)
    elif load_from:
        runner.load_checkpoint(load_from)

    # run
    workflow = [tuple(w) for w in workflow]
    runner.run(data_loaders, workflow, total_epochs, loss=loss)


# process a batch of data
def batch_processor(model, datas, train_mode, loss):

    data, label = datas
    data = data.cuda()
    label = label.cuda()

    # forward
    output = model(data)
    losses = loss(output, label)

    # output
    log_vars = dict(loss=losses.item())
    if not train_mode:
        log_vars['top1'] = topk_accuracy(output, label)
        log_vars['top5'] = topk_accuracy(output, label, 5)

    outputs = dict(loss=losses, log_vars=log_vars, num_samples=len(data.data))
    return outputs


def topk_accuracy(score, label, k=1):
    rank = score.argsort()
    hit_top_k = [l in rank[i, -k:] for i, l in enumerate(label)]
    accuracy = sum(hit_top_k) * 1.0 / len(hit_top_k)
    return accuracy


def weights_init(model):
    classname = model.__class__.__name__
    if classname.find('Conv1d') != -1:
        model.weight.data.normal_(0.0, 0.02)
        if model.bias is not None:
            model.bias.data.fill_(0)
    elif classname.find('Conv2d') != -1:
        model.weight.data.normal_(0.0, 0.02)
        if model.bias is not None:
            model.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        model.weight.data.normal_(1.0, 0.02)
        model.bias.data.fill_(0)
