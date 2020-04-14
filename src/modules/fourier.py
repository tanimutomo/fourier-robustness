import torch
from typing import NewType

Tensor = NewType('Tensor', torch.FloatTensor)


class FourierBasisNoise(object):
    def __init__(self, eps :int, norm :str, mean :list,
                 std :list, resol :int, device :torch.device):
        self.eps = eps
        self.norm = norm
        self.device = device
        self.normalize = Normalize(mean, std, device)
        self.unnormalize = Unnormalize(mean, std, device)

    def __call__(self, x :Tensor, i :int, j :int) -> Tensor:
        # unnormalize
        x = self.unnormalize(x)
        # sample fourier noise
        d = self.sample_noise(x, i, j)
        
        # norm constraint
        if str(self.norm) == 'inf':
            d *= self.eps / 255
        elif str(self.norm) == '2':
            norm_d = torch.norm(d.view(d.shape[0], -1), 2, dim=1)
            d /= norm_d[:, None, None, None] + 1e-12
            d *= self.eps / 255
        else:
            raise ValueError()

        # into image space
        y_ = torch.clamp(x + d, 0, 1)
        d = y_ - x
        # normalize
        y = self.normalize(x + d)

        return y

    def sample_noise(self, x :Tensor, i :int, j :int) -> Tensor:
        z = torch.zeros_like(x)
        z[:, :, i, j] = torch.rand(3, device=self.device)
        z = zshift(torch.stack([z, z], dim=-1))
        d = torch.ifft(z, 2)[..., 0]
        return batch_standadize(d) * 2 - 1


class Normalize(object):
    def __init__(self, mean: list, std: list, device: torch.device):
        self.mean = torch.tensor(mean, device=device)
        self.std = torch.tensor(std, device=device)

    def __call__(self, x: Tensor) -> Tensor:
        return x.sub(self.mean[None, :, None, None]).div(self.std[None, :, None, None])


class Unnormalize(object):
    def __init__(self, mean: list, std: list, device: torch.device):
        self.mean = torch.tensor(mean, device=device)
        self.std = torch.tensor(std, device=device)

    def __call__(self, x: Tensor) -> Tensor:
        return x.mul(self.std[None, :, None, None]).add(self.mean[None, :, None, None])


def zshift(z :Tensor) -> Tensor:
    assert z.ndim == 5 and z.shape[-1] == 2 and z.shape[-2] == z.shape[-3]
    resol = z.shape[-2]
    return torch.cat([
        torch.cat([
            z[..., resol//2:, resol//2:, :], # bottom right
            z[..., resol//2:, :resol//2, :], # bottom left
        ], dim=-2),
        torch.cat([
            z[..., :resol//2, resol//2:, :], # top right
            z[..., :resol//2, :resol//2, :], # top left
        ], dim=-2),
    ], dim=-3)


def batch_standadize(x):
    mn, _ = x.view(x.shape[0], -1).min(dim=1)
    mx, _ = x.view(x.shape[0], -1).max(dim=1)
    return (x - mn[:, None, None, None]) / (mx - mn)[:, None, None, None]