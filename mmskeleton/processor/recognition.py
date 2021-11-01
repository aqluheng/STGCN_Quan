from collections import OrderedDict
import torch
import logging
import numpy as np
from torch.nn import quantized
from torch.nn.modules import activation
from torch.nn.quantized.modules import DeQuantize, Quantize
from torch.quantization.stubs import DeQuantStub, QuantStub
from mmskeleton.utils import call_obj, import_obj, load_checkpoint
from mmcv.runner import Runner
from mmcv import Config, ProgressBar
from mmcv.parallel import MMDataParallel
from time import time
import os
import torch.nn as nn
from torch.quantization import prepare, convert
import mmskeleton


def print_size_of_model(model, label=""):
    torch.save(model.state_dict(), "temp.p")
    size = os.path.getsize("temp.p")
    print("model: ", label, ' \t', 'Size (KB):', size/1e3)
    os.remove('temp.p')
    return size

def load_myCheckpoint(model, path):
    load_dict = torch.load(path)["state_dict"]
    new_dict = {}
    for k, v in load_dict.items():
        name = k
        if len(k) > 20:
            if k[18:21] == "tcn":  # tcn.conv
                if k[22] == "2":
                    name = k[:22]+'3'+k[23:]
                elif k[22] == "3":  # tcn.bn
                    name = k[:22]+'5'+k[23:]
            if k[18:26] == "residual":
                if k[27] == "0":
                    name = k[:27]+'1'+k[28:]
                elif k[27] == "1":
                    name = k[:27]+'3'+k[28:]
        new_dict[name] = v
    model.load_state_dict(new_dict)
    return model

def quantCalibrateModel(model, dataset_cfg, path):
    dataset_train_cfg = dataset_cfg.copy()
    # dataset_train_cfg["data_path"] = "./data/Kinetics/kinetics-skeleton/train_data.npy"
    # dataset_train_cfg["label_path"] = "./data/Kinetics/kinetics-skeleton/train_label.pkl"
    # dataset_train_cfg["random_choose"] = True
    # dataset_train_cfg["random_move"] = True
    # dataset_train_cfg["window_size"] = 150
    dataset_train = call_obj(**dataset_train_cfg)
    data_loader_train = torch.utils.data.DataLoader(dataset=dataset_train,
                                                    batch_size=64,
                                                    # shuffle=True,
                                                    num_workers=4)
    # model.qconfig = torch.quantization.default_qconfig
    model.eval()
    model.qconfig = torch.quantization.get_default_qconfig("fbgemm")

    prepare(model, allow_list={nn.Conv2d,QuantStub,DeQuantize}, inplace=True)

    nCalibrateBatch = 15
    cnt = 0
    prog_bar = ProgressBar(nCalibrateBatch)
    for data, label in data_loader_train:
        with torch.no_grad():
            if cnt >= nCalibrateBatch:
                break
            data = data.cpu()
            output = model(data).data
            prog_bar.update()
            cnt += 1

    convert(model, mapping={nn.Conv2d: quantized.Conv2d,QuantStub:quantized.Quantize,DeQuantStub:quantized.DeQuantize}, inplace=True)
    # print(model)
    torch.save({"state_dict": model.state_dict()}, path)
    print("量化校准完成")
    return model


def loadQuantModel(model, path):
    model.qconfig = torch.quantization.default_qconfig
    model.eval()
    prepare(model, allow_list={nn.Conv2d,QuantStub,DeQuantize}, inplace=True)
    convert(model, mapping={nn.Conv2d: quantized.Conv2d,QuantStub:quantized.Quantize,DeQuantStub:quantized.DeQuantize}, inplace=True)
    model.load_state_dict(torch.load(path)["state_dict"])
    return model


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
    # 原有读取checkpoint
    # load_checkpoint(model, checkpoint, map_location='cpu')
    
    # 读取checkpoint,并进行量化
    model = load_myCheckpoint(model, "testModel.pth")
    model = quantCalibrateModel(model, dataset_cfg, "quantSTGCN.pth")

    # model = MMDataParallel(model, device_ids=range(gpus)).cuda()
    # model = loadQuantModel(model, "quantSTGCN.pth")

    results = []
    labels = []
    evaluateTime = 0
    evaluteBatch = 15
    prog_bar = ProgressBar(len(dataset) if evaluteBatch == 0 else evaluteBatch)
    with torch.no_grad():

        for data, label in data_loader:
            evaluateTime -= time()
            output = model(data).data.cpu().numpy()
            evaluateTime += time()

            results.append(output)
            labels.append(label)
            if evaluteBatch == 0:
                for i in range(len(data)):
                    prog_bar.update()
            else:
                prog_bar.update()
                if len(results) >= evaluteBatch:
                    break

    results = np.concatenate(results)
    labels = np.concatenate(labels)

    print('Evalute Time:', evaluateTime)
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
