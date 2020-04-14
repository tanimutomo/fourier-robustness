import os
import random
import sys

import hydra
import numpy as np
import torch
import torchvision
from tqdm import tqdm

from hydra import utils
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image

from modules.models import ResNet56
from modules.fourier import FourierBasisNoise
from modules.misc import set_seed, unnormalize, extract_subset


@hydra.main(config_path='config/cifar10.yaml')
def main(cfg):
    is_config_valid(cfg)
    if cfg.seed is not None:
        set_seed(cfg.seed)

    # device
    device = torch.device(f'cuda:{cfg.gpu_ids[0]}')

    # dataset and loader
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(cfg.data.mean, cfg.data.std)
    ])
    dataset = datasets.CIFAR10(
        root=os.path.join(cfg.data.root, 'cifar10'),
        train=False, transform=transform
    )
    dataset = extract_subset(dataset, cfg.data.num_samples, False)
    loader = DataLoader(dataset, batch_size=cfg.data.batch_size, shuffle=False, num_workers=4)

    # model
    model = ResNet56()
    weight_path = os.path.join(utils.get_original_cwd(), cfg.weight_path)
    model.load_state_dict(torch.load(weight_path, map_location='cpu'))
    model.to(device)
    if len(cfg.gpu_ids) > 1:
        model = nn.DataParallel(model, device_ids=cfg.gpu_ids)
    model.eval()

    # frequency noise
    noiser = FourierBasisNoise(cfg.eps, cfg.norm, cfg.data.mean,
                               cfg.data.std, cfg.data.resol, device)

    # basis loop
    xs = list()
    with tqdm(total=cfg.data.resol**2, ncols=80) as pbar:
        for i in range(cfg.data.resol):
            for j in range(cfg.data.resol):
                acc_sum, sx = test(device, model, loader, noiser, i, j)
                acc = acc_sum / cfg.data.num_samples
                xs.append(sx)

    xs = torch.stack(xs, axis=0)
    save_image(
        unnormalize(xs, cfg.data.mean, cfg.data.std),
        f'input.png',
    )


def test(device, model, loader, noiser, i, j):
    acc_sum, num_samples = 0, 0
    with torch.no_grad():
        for itr, (x, y) in enumerate(loader):
            x, y = x.to(device), y.to(device)
            x = noiser(x, i, j)

            z = model(x)
            loss = F.cross_entropy(z, y)
            acc, _ = accuracy(z, y, topk=(1, 5))

            acc_sum += acc
            if itr == 0: save_x = x[0].detach()

    return acc_sum, save_x


def is_config_valid(cfg):
    if cfg.experiment == 'save':
        assert cfg.experiment.prefix
    assert cfg.weight_path

    print(cfg.pretty())


if __name__ == '__main__':
    main()
