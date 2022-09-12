import numpy as np
import torch

from collections import defaultdict
from utils.common.utils import save_reconstructions
from utils.data.load_data import create_data_loaders
from utils.model.varnet import VarNet
from utils.model.multidomain import MultiDomainNet
from utils.model.crossdomain import CrossDomainNet
from utils.model.ensemble_unet import EnsembleNet

import copy


def test(args, model, data_loader):
    model.eval()
    reconstructions = defaultdict(dict)
    
    with torch.no_grad():
        for (mask, kspace, _, _, fnames, slices) in data_loader:
            kspace = kspace.cuda(non_blocking=True)
            mask = mask.cuda(non_blocking=True)
            output = model(kspace, mask)

            for i in range(output.shape[0]):
                reconstructions[fnames[i]][int(slices[i])] = output[i].cpu().numpy()

    for fname in reconstructions:
        reconstructions[fname] = np.stack(
            [out for _, out in sorted(reconstructions[fname].items())]
        )
    return reconstructions, None


def forward(args):

    device = torch.device(f'cuda:{args.GPU_NUM}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)
    print ('Current cuda device ', torch.cuda.current_device())

    varnet = VarNet()
    varnet.to(device=device)
    checkpoint = torch.load(args.exp_dir_varnet / 'best_model_gpu.pt', map_location='cpu')
    print("VarNet Best Model: Epoch =", checkpoint['epoch'], ", ValLoss =", "{:.4g}".format(checkpoint['best_val_loss'].item()))
    varnet.load_state_dict(checkpoint['model'])

    multidomainnet = MultiDomainNet()
    multidomainnet.to(device=device)
    checkpoint = torch.load(args.exp_dir_multidomain / 'best_model_gpu.pt', map_location='cpu')
    print("MultiDomainNet Best Model: Epoch =", checkpoint['epoch'], ", ValLoss =", "{:.4g}".format(checkpoint['best_val_loss'].item()))
    multidomainnet.load_state_dict(checkpoint['model'])

    crossdomainnet7 = CrossDomainNet(num_cascades=7, chans=16, pools=5)
    crossdomainnet7.to(device=device)
    checkpoint = torch.load(args.exp_dir_crossdomain7 / 'best_model_gpu.pt', map_location='cpu')
    print("CrossDomainNet7 Best Model: Epoch =", checkpoint['epoch'], ", ValLoss =", "{:.4g}".format(checkpoint['best_val_loss'].item()))
    crossdomainnet7.load_state_dict(checkpoint['model'])

    crossdomainnet8 = CrossDomainNet(num_cascades=8, chans=18, pools=4)
    crossdomainnet8.to(device=device)
    checkpoint = torch.load(args.exp_dir_crossdomain8 / 'best_model_gpu.pt', map_location='cpu')
    print("CrossDomainNet8 Best Model: Epoch =", checkpoint['epoch'], ", ValLoss =", "{:.4g}".format(checkpoint['best_val_loss'].item()))
    crossdomainnet8.load_state_dict(checkpoint['model'])

    ensemblenet = EnsembleNet(varnet, multidomainnet, crossdomainnet7, crossdomainnet8)
    ensemblenet.to(device=device)
    checkpoint = torch.load(args.exp_dir_ensemble / 'best_model.pt', map_location='cpu')
    print("ensemblenet Best Model: Epoch =", checkpoint['epoch'], ", ValLoss =", "{:.4g}".format(checkpoint['best_val_loss'].item()))
    ensemblenet.load_state_dict(checkpoint["model"])
    
    forward_loader = create_data_loaders(data_path = args.data_path, args = args, isforward = True)
    reconstructions, inputs = test(args, crossdomainnet7, forward_loader)
    save_reconstructions(reconstructions, args.forward_dir, inputs=inputs)