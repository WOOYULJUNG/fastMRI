import shutil
import numpy as np
import torch
import torch.nn as nn
import time
import requests
from tqdm import tqdm
from pathlib import Path
import copy

from collections import defaultdict
from utils.data.load_data import create_data_loaders
from utils.common.utils import save_reconstructions, ssim_loss
from utils.common.loss_function import SSIMLoss
from utils.model.varnet import VarNet
from utils.model.multidomain import MultiDomainNet
from utils.model.crossdomain import CrossDomainNet
from utils.model.ensemble import EnsembleNet

import psutil
import humanize
import os
import GPUtil as GPU

def train_epoch(args, epoch, model, data_loader, optimizer, loss_type):
    model.train()
    start_epoch = start_iter = time.perf_counter()
    len_loader = len(data_loader)
    total_loss = 0.
    max_memory = 0.

    for iter, data in enumerate(data_loader):
        mask, kspace, target, maximum, _, _ = data
        mask = mask.cuda(non_blocking=True)
        kspace = kspace.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        maximum = maximum.cuda(non_blocking=True)

        output = model(kspace, mask)
        loss = loss_type(output, target, maximum)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        GPUs = GPU.getGPUs()
        gpu = GPUs[0]
        process = psutil.Process(os.getpid())
        if(max_memory < gpu.memoryUsed):
            max_memory = gpu.memoryUsed
            print("GPU RAM Free: {0:.0f}MB | Used: {1:.0f}MB | Util {2:3.0f}% | Total     {3:.0f}MB".format(gpu.memoryFree, gpu.memoryUsed, gpu.memoryUtil*100, gpu.memoryTotal))

        if iter % args.report_interval == 0:
            print(
                f'Epoch = [{epoch:3d}/{args.num_epochs:3d}] '
                f'Iter = [{iter:4d}/{len(data_loader):4d}] '
                f'Loss = {loss.item():.4g} '
                f'Time = {time.perf_counter() - start_iter:.4f}s',
            )
            start_iter = time.perf_counter()
    total_loss = total_loss / len_loader
    return total_loss, time.perf_counter() - start_epoch


def validate(args, model, data_loader):
    model.eval()
    reconstructions = defaultdict(dict)
    targets = defaultdict(dict)
    start = time.perf_counter()

    with torch.no_grad():
        for iter, data in enumerate(data_loader):
            mask, kspace, target, _, fnames, slices = data
            kspace = kspace.cuda(non_blocking=True)
            mask = mask.cuda(non_blocking=True)
            output = model(kspace, mask)

            for i in range(output.shape[0]):
                reconstructions[fnames[i]][int(slices[i])] = output[i].cpu().numpy()
                targets[fnames[i]][int(slices[i])] = target[i].numpy()

    for fname in reconstructions:
        reconstructions[fname] = np.stack(
            [out for _, out in sorted(reconstructions[fname].items())]
        )
    for fname in targets:
        targets[fname] = np.stack(
            [out for _, out in sorted(targets[fname].items())]
        )
    metric_loss = sum([ssim_loss(targets[fname], reconstructions[fname]) for fname in reconstructions])
    num_subjects = len(reconstructions)
    return metric_loss, num_subjects, reconstructions, targets, None, time.perf_counter() - start


def save_model(args, exp_dir, epoch, model, optimizer, best_val_loss, is_new_best):
    torch.save(
        {
            'epoch': epoch,
            'args': args,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_val_loss': best_val_loss,
            'exp_dir': exp_dir
        },
        f=exp_dir / 'model.pt'
    )
    if is_new_best:
        shutil.copyfile(exp_dir / 'model.pt', exp_dir / 'best_model.pt')


def download_model(url, fname):
    response = requests.get(url, timeout=10, stream=True)

    chunk_size = 8 * 1024 * 1024  # 8 MB chunks
    total_size_in_bytes = int(response.headers.get("content-length", 0))
    progress_bar = tqdm(
        desc="Downloading state_dict",
        total=total_size_in_bytes,
        unit="iB",
        unit_scale=True,
    )

    with open(fname, "wb") as fh:
        for chunk in response.iter_content(chunk_size):
            progress_bar.update(len(chunk))
            fh.write(chunk)


# VarNet TRAIN
def varnet_train(args):
    device = torch.device(f'cuda:{args.GPU_NUM}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)

    model = VarNet()
    model.to(device=device)

    # using pretrained parameter
    VARNET_FOLDER = "https://dl.fbaipublicfiles.com/fastMRI/trained_models/varnet/"
    MODEL_FNAMES = "brain_leaderboard_state_dict.pt"
    if not Path(MODEL_FNAMES).exists():
        url_root = VARNET_FOLDER
        download_model(url_root + MODEL_FNAMES, MODEL_FNAMES)

    pretrained = torch.load(MODEL_FNAMES)
    pretrained_copy = copy.deepcopy(pretrained)
    for layer in pretrained_copy.keys():
        if layer.split('.',2)[1].isdigit() and (8<= int(layer.split('.',2)[1]) <=11):
            del pretrained[layer]

    model.load_state_dict(pretrained)

    for name, child in model.named_children():
        if(name=="sens_net"):
            for param in child.parameters():
                param.requires_grad = False

    loss_type = SSIMLoss().to(device=device)
    optimizer = torch.optim.NAdam(model.parameters(), 1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.4)

    best_val_loss = 1.
    start_epoch = 0

    train_loader = create_data_loaders(data_path = args.data_path_train, args = args)
    val_loader = create_data_loaders(data_path = args.data_path_val, args = args, isval=True)

    for epoch in range(start_epoch, args.num_epochs):
        print(f'Epoch #{epoch:2d} ............... {args.net_name}_varnet ...............') 
        
        train_loss, train_time = train_epoch(args, epoch, model, train_loader, optimizer, loss_type)
        val_loss, num_subjects, reconstructions, targets, inputs, val_time = validate(args, model, val_loader)

        scheduler.step()

        train_loss = torch.tensor(train_loss).cuda(non_blocking=True)
        val_loss = torch.tensor(val_loss).cuda(non_blocking=True)
        num_subjects = torch.tensor(num_subjects).cuda(non_blocking=True)

        val_loss = val_loss / num_subjects

        is_new_best = val_loss < best_val_loss
        best_val_loss = min(best_val_loss, val_loss)

        save_model(args, args.exp_dir_varnet, epoch + 1, model, optimizer, best_val_loss, is_new_best)
        print(
            f'Epoch = [{epoch:4d}/{args.num_epochs:4d}] TrainLoss = {train_loss:.4g} '
            f'ValLoss = {val_loss:.4g} TrainTime = {train_time:.4f}s ValTime = {val_time:.4f}s',
        )

        if is_new_best:
            print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@NewRecord@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
    print()

# MultiDomain TRAIN
def multidomain_train(args):
    device = torch.device(f'cuda:{args.GPU_NUM}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)

    model = MultiDomainNet()
    model.to(device=device)

    # using pretrained parameter
    VARNET_FOLDER = "https://dl.fbaipublicfiles.com/fastMRI/trained_models/varnet/"
    MODEL_FNAMES = "brain_leaderboard_state_dict.pt"
    if not Path(MODEL_FNAMES).exists():
        url_root = VARNET_FOLDER
        download_model(url_root + MODEL_FNAMES, MODEL_FNAMES)

    pretrained = torch.load(MODEL_FNAMES)
    pretrained_copy = copy.deepcopy(pretrained)
    for layer in pretrained_copy.keys():
        if layer.split('.',2)[1].isdigit() and (0<= int(layer.split('.',2)[1]) <=11):
            del pretrained[layer]

    model_dict = model.state_dict()
    model_dict.update(pretrained)
    model.load_state_dict(model_dict)

    for name, child in model.named_children():
        if(name=="sens_net"):
            for param in child.parameters():
                param.requires_grad = False

    loss_type = SSIMLoss().to(device=device)
    optimizer = torch.optim.NAdam(model.parameters(), 1e-3)

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[15, 25], gamma=0.4)

    best_val_loss = 1.
    start_epoch = 0

    train_loader = create_data_loaders(data_path = args.data_path_train, args = args)
    val_loader = create_data_loaders(data_path = args.data_path_val, args = args, isval=True)

    for epoch in range(start_epoch, args.num_epochs):
        print(f'Epoch #{epoch:2d} ............... {args.net_name}_multidomainnet ...............')
        
        train_loss, train_time = train_epoch(args, epoch, model, train_loader, optimizer, loss_type)
        val_loss, num_subjects, reconstructions, targets, inputs, val_time = validate(args, model, val_loader)

        scheduler.step()

        train_loss = torch.tensor(train_loss).cuda(non_blocking=True)
        val_loss = torch.tensor(val_loss).cuda(non_blocking=True)
        num_subjects = torch.tensor(num_subjects).cuda(non_blocking=True)

        val_loss = val_loss / num_subjects

        is_new_best = val_loss < best_val_loss
        best_val_loss = min(best_val_loss, val_loss)

        save_model(args, args.exp_dir_multidomain, epoch + 1, model, optimizer, best_val_loss, is_new_best)
        print(
            f'Epoch = [{epoch:4d}/{args.num_epochs:4d}] TrainLoss = {train_loss:.4g} '
            f'ValLoss = {val_loss:.4g} TrainTime = {train_time:.4f}s ValTime = {val_time:.4f}s',
        )

        if is_new_best:
            print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@NewRecord@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
    print()

# CrossDomain cascade 7 TRAIN
def crossdomain7_train(args):
    device = torch.device(f'cuda:{args.GPU_NUM}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)

    model = CrossDomainNet(num_cascades=7, chans=16, pools=5)
    model.to(device=device)

    # using pretrained parameter
    VARNET_FOLDER = "https://dl.fbaipublicfiles.com/fastMRI/trained_models/varnet/"
    MODEL_FNAMES = "brain_leaderboard_state_dict.pt"
    if not Path(MODEL_FNAMES).exists():
        url_root = VARNET_FOLDER
        download_model(url_root + MODEL_FNAMES, MODEL_FNAMES)

    pretrained = torch.load(MODEL_FNAMES)
    pretrained_copy = copy.deepcopy(pretrained)
    for layer in pretrained_copy.keys():
        if layer.split('.',2)[1].isdigit() and (0<= int(layer.split('.',2)[1]) <=11):
            del pretrained[layer]

    model_dict = model.state_dict()
    model_dict.update(pretrained)
    model.load_state_dict(model_dict)

    for name, child in model.named_children():
        if(name=="sens_net"):
            for param in child.parameters():
                param.requires_grad = False

    loss_type = SSIMLoss().to(device=device)
    optimizer = torch.optim.NAdam(model.parameters(), 2e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.2)

    best_val_loss = 1.
    start_epoch = 0

    train_loader = create_data_loaders(data_path = args.data_path_train, args = args)
    val_loader = create_data_loaders(data_path = args.data_path_val, args = args, isval=True)

    for epoch in range(start_epoch, args.num_epochs):
        print(f'Epoch #{epoch:2d} ............... {args.net_name}_crossdomainnet7 ...............')
        
        train_loss, train_time = train_epoch(args, epoch, model, train_loader, optimizer, loss_type)
        val_loss, num_subjects, reconstructions, targets, inputs, val_time = validate(args, model, val_loader)

        scheduler.step()

        train_loss = torch.tensor(train_loss).cuda(non_blocking=True)
        val_loss = torch.tensor(val_loss).cuda(non_blocking=True)
        num_subjects = torch.tensor(num_subjects).cuda(non_blocking=True)

        val_loss = val_loss / num_subjects

        is_new_best = val_loss < best_val_loss
        best_val_loss = min(best_val_loss, val_loss)

        save_model(args, args.exp_dir_crossdomain7, epoch + 1, model, optimizer, best_val_loss, is_new_best)
        print(
            f'Epoch = [{epoch:4d}/{args.num_epochs:4d}] TrainLoss = {train_loss:.4g} '
            f'ValLoss = {val_loss:.4g} TrainTime = {train_time:.4f}s ValTime = {val_time:.4f}s',
        )

        if is_new_best:
            print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@NewRecord@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
    print()

# CrossDomain cascade 8 TRAIN
def crossdomain8_train(args):
    device = torch.device(f'cuda:{args.GPU_NUM}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)

    model = CrossDomainNet(num_cascades=8, chans=18, pools=4)
    model.to(device=device)

    # using pretrained parameter
    VARNET_FOLDER = "https://dl.fbaipublicfiles.com/fastMRI/trained_models/varnet/"
    MODEL_FNAMES = "brain_leaderboard_state_dict.pt"
    if not Path(MODEL_FNAMES).exists():
        url_root = VARNET_FOLDER
        download_model(url_root + MODEL_FNAMES, MODEL_FNAMES)

    pretrained = torch.load(MODEL_FNAMES)
    pretrained_copy = copy.deepcopy(pretrained)
    for layer in pretrained_copy.keys():
        if layer.split('.',2)[1].isdigit() and (0<= int(layer.split('.',2)[1]) <=11):
            del pretrained[layer]

    model_dict = model.state_dict()
    model_dict.update(pretrained)
    model.load_state_dict(model_dict)

    for name, child in model.named_children():
        if(name=="sens_net"):
            for param in child.parameters():
                param.requires_grad = False

    loss_type = SSIMLoss().to(device=device)
    optimizer = torch.optim.NAdam(model.parameters(), 1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.4)

    best_val_loss = 1.
    start_epoch = 0

    train_loader = create_data_loaders(data_path = args.data_path_train, args = args)
    val_loader = create_data_loaders(data_path = args.data_path_val, args = args, isval=True)

    for epoch in range(start_epoch, args.num_epochs):
        print(f'Epoch #{epoch:2d} ............... {args.net_name}_crossdomainnet8 ...............')
        
        train_loss, train_time = train_epoch(args, epoch, model, train_loader, optimizer, loss_type)
        val_loss, num_subjects, reconstructions, targets, inputs, val_time = validate(args, model, val_loader)

        scheduler.step()

        train_loss = torch.tensor(train_loss).cuda(non_blocking=True)
        val_loss = torch.tensor(val_loss).cuda(non_blocking=True)
        num_subjects = torch.tensor(num_subjects).cuda(non_blocking=True)

        val_loss = val_loss / num_subjects

        is_new_best = val_loss < best_val_loss
        best_val_loss = min(best_val_loss, val_loss)

        save_model(args, args.exp_dir_crossdomain8, epoch + 1, model, optimizer, best_val_loss, is_new_best)
        print(
            f'Epoch = [{epoch:4d}/{args.num_epochs:4d}] TrainLoss = {train_loss:.4g} '
            f'ValLoss = {val_loss:.4g} TrainTime = {train_time:.4f}s ValTime = {val_time:.4f}s',
        )

        if is_new_best:
            print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@NewRecord@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
    print()

def ensemble_train(args):
    device = torch.device(f'cuda:{args.GPU_NUM}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)

    varnet = VarNet()
    varnet.to(device=device)
    checkpoint = torch.load(args.exp_dir_varnet / 'best_model_gpu.pt', map_location='cpu')
    print("VarNet Best Model: Epoch =", checkpoint['epoch'], ", ValLoss =", "{:.4g}".format(checkpoint['best_val_loss'].item()))
    varnet.load_state_dict(checkpoint['model'])
    for name, child in varnet.named_children():
        for param in child.parameters():
            param.requires_grad = False

    multidomainnet = MultiDomainNet()
    multidomainnet.to(device=device)
    checkpoint = torch.load(args.exp_dir_multidomain / 'best_model_gpu.pt', map_location='cpu')
    print("MultiDomainNet Best Model: Epoch =", checkpoint['epoch'], ", ValLoss =", "{:.4g}".format(checkpoint['best_val_loss'].item()))
    multidomainnet.load_state_dict(checkpoint['model'])
    for name, child in multidomainnet.named_children():
        for param in child.parameters():
            param.requires_grad = False

    crossdomainnet7 = CrossDomainNet(num_cascades=7, chans=16, pools=5)
    crossdomainnet7.to(device=device)
    checkpoint = torch.load(args.exp_dir_crossdomain7 / 'best_model_gpu.pt', map_location='cpu')
    print("CrossDomainNet7 Best Model: Epoch =", checkpoint['epoch'], ", ValLoss =", "{:.4g}".format(checkpoint['best_val_loss'].item()))
    crossdomainnet7.load_state_dict(checkpoint['model'])
    for name, child in crossdomainnet7.named_children():
        for param in child.parameters():
            param.requires_grad = False

    crossdomainnet8 = CrossDomainNet(num_cascades=8, chans=18, pools=4)
    crossdomainnet8.to(device=device)
    checkpoint = torch.load(args.exp_dir_crossdomain8 / 'best_model_gpu.pt', map_location='cpu')
    print("CrossDomainNet8 Best Model: Epoch =", checkpoint['epoch'], ", ValLoss =", "{:.4g}".format(checkpoint['best_val_loss'].item()))
    crossdomainnet8.load_state_dict(checkpoint['model'])
    for name, child in crossdomainnet8.named_children():
        for param in child.parameters():
            param.requires_grad = False

    ensemblenet = EnsembleNet(varnet, multidomainnet, crossdomainnet7, crossdomainnet8)
    ensemblenet.to(device=device)

    loss_type = SSIMLoss().to(device=device)
    optimizer = torch.optim.Adam(ensemblenet.parameters(), 5e-2)

    best_val_loss = 1.
    start_epoch = 0

    train_loader = create_data_loaders(data_path = args.data_path_train, args = args)
    val_loader = create_data_loaders(data_path = args.data_path_val, args = args, isval=True)

    for epoch in range(start_epoch, args.num_epochs):
        print(f'Epoch #{epoch:2d} ............... {args.net_name}_ensemblenet ...............')
        
        train_loss, train_time = train_epoch(args, epoch, ensemblenet, train_loader, optimizer, loss_type)
        val_loss, num_subjects, reconstructions, targets, inputs, val_time = validate(args, ensemblenet, val_loader)

        train_loss = torch.tensor(train_loss).cuda(non_blocking=True)
        val_loss = torch.tensor(val_loss).cuda(non_blocking=True)
        num_subjects = torch.tensor(num_subjects).cuda(non_blocking=True)

        val_loss = val_loss / num_subjects

        is_new_best = val_loss < best_val_loss
        best_val_loss = min(best_val_loss, val_loss)

        save_model(args, args.exp_dir_ensemble, epoch + 1, ensemblenet, optimizer, best_val_loss, is_new_best)
        print(
            f'Epoch = [{epoch:4d}/{args.num_epochs:4d}] TrainLoss = {train_loss:.4g} '
            f'ValLoss = {val_loss:.4g} TrainTime = {train_time:.4f}s ValTime = {val_time:.4f}s',
        )

        if is_new_best:
            print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@NewRecord@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
            start = time.perf_counter()
            save_reconstructions(reconstructions, args.val_dir, targets=targets, inputs=inputs)
            print(
                f'ForwardTime = {time.perf_counter() - start:.4f}s',
            )

def train_all(args):
    device = torch.device(f'cuda:{args.GPU_NUM}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)
    print('Current cuda device: ', torch.cuda.current_device())

#    varnet_train(args)
#    multidomain_train(args)
#    crossdomain7_train(args)
#    crossdomain8_train(args)
    ensemble_train(args)
