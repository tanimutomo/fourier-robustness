import random

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from torch.utils.data import Subset


def extract_subset(dataset, num_subset :int, random_subset :bool):
    if random_subset:
        random.seed(0)
        indices = random.sample(list(range(len(dataset))), num_subset)
    else:
        indices = [i for i in range(num_subset)]
    return Subset(dataset, indices)


def unnormalize(x, mean, std):
    mean = torch.tensor(mean, device=x.device)[None, :, None, None]
    std = torch.tensor(std, device=x.device)[None, :, None, None]
    return  x.mul(std).add(mean)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def accuracy(output, target, topk=(1,5)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, dim=1) # top-k index: size (B, k)
        pred = pred.t() # size (k, B)
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        acc = []
        for k in topk:
            correct_k = correct[:k].float().sum()
            acc.append(correct_k * 100.0 / batch_size)

        if len(acc) == 1:
            return acc[0]
        else:
            return acc


def save_fouriermap(mat :torch.FloatTensor) -> None:
    plt.figure(figsize=(12, 12))
    plt.rcParams['font.size'] = 18

    plt.imshow(mat.numpy(), cmap=plt.get_cmap('jet'))
    plt.clim(0.0, 1.0)
    plt.colorbar(shrink=0.67)

    plt.subplots_adjust(left=0.05, right=1.05, top=1.1, bottom=-0.1)
    plt.tick_params(
        bottom=False, labelbottom=False,
        left=False, labelleft=False,
        right=False, labelright=False,
        top=False, labeltop=False,
    )
    plt.savefig('fouriermap.png')