import torch
import torch.nn as nn
import torch.nn.functional as F

from .networks import get_network, LinLayers
from .utils import get_state_dict


class LPIPS(nn.Module):
    r"""Creates a criterion that measures
    Learned Perceptual Image Patch Similarity (LPIPS).

    Arguments:
        net_type (str): the network type to compare the features: 
                        'alex' | 'squeeze' | 'vgg'. Default: 'alex'.
        version (str): the version of LPIPS. Default: 0.1.
    """
    def __init__(self, net_type: str = 'alex', version: str = '0.1'):

        assert version in ['0.1'], 'v0.1 is only supported now'

        super(LPIPS, self).__init__()

        # pretrained network
        self.net = get_network(net_type)

        # linear layers
        self.lin = LinLayers(self.net.n_channels_list)
        self.lin.load_state_dict(get_state_dict(net_type, version))

    def distance_maps(self, x: torch.Tensor, y: torch.Tensor):
        """Return learned, channel-reduced distance maps at every feature scale."""
        feat_x, feat_y = self.net(x), self.net(y)

        diff = [(fx - fy) ** 2 for fx, fy in zip(feat_x, feat_y)]
        return [layer(difference) for difference, layer in zip(diff, self.lin)]

    @staticmethod
    def reduce_distance_maps(distance_maps):
        res = [distance_map.mean((2, 3), True) for distance_map in distance_maps]
        return torch.sum(torch.cat(res, 0), 0, True)

    @staticmethod
    def upsample_distance_maps(distance_maps, output_size):
        upsampled = [
            F.interpolate(
                distance_map, size=output_size, mode="bilinear", align_corners=False
            )
            for distance_map in distance_maps
        ]
        return torch.stack(upsampled, dim=0).sum(dim=0)

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        return self.reduce_distance_maps(self.distance_maps(x, y))

    def forward_with_map(self, x: torch.Tensor, y: torch.Tensor):
        """Return the scalar LPIPS value and an input-resolution spatial map."""
        distance_maps = self.distance_maps(x, y)
        scalar = self.reduce_distance_maps(distance_maps)
        spatial = self.upsample_distance_maps(distance_maps, x.shape[-2:])
        return scalar, spatial
