# -*- coding: utf-8 -*-
import math
from collections import OrderedDict
from itertools import chain

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint


__all__ = ["resnet"]


# from https://github.com/rwightman/pytorch-image-models/blob/main/timm/models/helpers.py#L722
def checkpoint_seq(
    functions, x, every=1, flatten=False, skip_last=False, preserve_rng_state=True
):
    r"""A helper function for checkpointing sequential models.

    Sequential models execute a list of modules/functions in order
    (sequentially). Therefore, we can divide such a sequence into segments
    and checkpoint each segment. All segments except run in :func:`torch.no_grad`
    manner, i.e., not storing the intermediate activations. The inputs of each
    checkpointed segment will be saved for re-running the segment in the backward pass.

    See :func:`~torch.utils.checkpoint.checkpoint` on how checkpointing works.

    .. warning::
        Checkpointing currently only supports :func:`torch.autograd.backward`
        and only if its `inputs` argument is not passed. :func:`torch.autograd.grad`
        is not supported.

    .. warning:
        At least one of the inputs needs to have :code:`requires_grad=True` if
        grads are needed for model inputs, otherwise the checkpointed part of the
        model won't have gradients.

    Args:
        functions: A :class:`torch.nn.Sequential` or the list of modules or functions to run sequentially.
        x: A Tensor that is input to :attr:`functions`
        every: checkpoint every-n functions (default: 1)
        flatten (bool): flatten nn.Sequential of nn.Sequentials
        skip_last (bool): skip checkpointing the last function in the sequence if True
        preserve_rng_state (bool, optional, default=True):  Omit stashing and restoring
            the RNG state during each checkpoint.

    Returns:
        Output of running :attr:`functions` sequentially on :attr:`*inputs`

    Example:
        >>> model = nn.Sequential(...)
        >>> input_var = checkpoint_seq(model, input_var, every=2)
    """

    def run_function(start, end, functions):
        def forward(_x):
            for j in range(start, end + 1):
                _x = functions[j](_x)
            return _x

        return forward

    if isinstance(functions, torch.nn.Sequential):
        functions = functions.children()
    if flatten:
        functions = chain.from_iterable(functions)
    if not isinstance(functions, (tuple, list)):
        functions = tuple(functions)

    num_checkpointed = len(functions)
    if skip_last:
        num_checkpointed -= 1
    end = -1
    for start in range(0, num_checkpointed, every):
        end = min(start + every - 1, num_checkpointed - 1)
        x = checkpoint(
            run_function(start, end, functions),
            x,
            preserve_rng_state=preserve_rng_state,
        )
    if skip_last:
        return run_function(end + 1, len(functions) - 1, functions)(x)
    return x


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding."
    return nn.Conv2d(
        in_channels=in_planes,
        out_channels=out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False,
    )


def norm2d(group_norm_num_groups, planes):
    if group_norm_num_groups is not None and group_norm_num_groups > 0:
        # group_norm_num_groups == planes -> InstanceNorm
        # group_norm_num_groups == 1 -> LayerNorm
        return nn.GroupNorm(group_norm_num_groups, planes)
    else:
        return nn.BatchNorm2d(planes)


class ViewFlatten(nn.Module):
    def __init__(self):
        super(ViewFlatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


class BasicBlock(nn.Module):
    """
    [3 * 3, 64]
    [3 * 3, 64]
    """

    expansion = 1

    def __init__(
        self,
        in_planes,
        out_planes,
        stride=1,
        downsample=None,
        group_norm_num_groups=None,
    ):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, out_planes, stride)
        self.bn1 = norm2d(group_norm_num_groups, planes=out_planes)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = conv3x3(out_planes, out_planes)
        self.bn2 = norm2d(group_norm_num_groups, planes=out_planes)

        self.downsample = downsample
        self.stride = stride

        # some stats
        self.nn_mass = in_planes + out_planes

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out.expand_as(residual) + residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    """
    [1 * 1, x]
    [3 * 3, x]
    [1 * 1, x * 4]
    """

    expansion = 4

    def __init__(
        self,
        in_planes,
        out_planes,
        stride=1,
        downsample=None,
        group_norm_num_groups=None,
    ):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_planes, out_channels=out_planes, kernel_size=1, bias=False
        )
        self.bn1 = norm2d(group_norm_num_groups, planes=out_planes)

        self.conv2 = nn.Conv2d(
            in_channels=out_planes,
            out_channels=out_planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn2 = norm2d(group_norm_num_groups, planes=out_planes)

        self.conv3 = nn.Conv2d(
            in_channels=out_planes,
            out_channels=out_planes * self.expansion,
            kernel_size=1,
            bias=False,
        )
        self.bn3 = norm2d(group_norm_num_groups, planes=out_planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = downsample
        self.stride = stride

        # some stats
        self.nn_mass = (
            (in_planes + 2 * out_planes) * in_planes / (2 * in_planes + out_planes)
        )

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out.expand_as(residual) + residual
        out = self.relu(out)
        return out


class ResNetBase(nn.Module):
    def _decide_num_classes(self):
        if self.dataset in ["svhn", "cifar10"]:
            return 10
        elif "cifar100" in self.dataset:
            return 100
        elif "imagenet" in self.dataset:
            return 1000
        elif "officehome" in self.dataset:
            return 65
        elif "pacs" in self.dataset:
            return 7
        elif self.dataset in ["coloredmnist", "waterbirds"]:
            return 2
        else:
            raise NotImplementedError(f"Unsupported dataset: {self.dataset}.")

    def _init_conv(self, module):
        out_channels, _, kernel_size0, kernel_size1 = module.weight.size()
        n = kernel_size0 * kernel_size1 * out_channels
        module.weight.data.normal_(0, math.sqrt(2.0 / n))

    def _init_bn(self, module):
        module.weight.data.fill_(1)
        module.bias.data.zero_()

    def _init_fc(self, module):
        module.weight.data.normal_(mean=0, std=0.01)
        if module.bias is not None:
            module.bias.data.zero_()

    def _weight_initialization(self):
        for name, module in self.named_modules():
            if isinstance(module, nn.Conv2d):
                self._init_conv(module)
            elif isinstance(module, nn.BatchNorm2d):
                self._init_bn(module)
            elif isinstance(module, nn.Linear):
                self._init_fc(module)

    def _make_block(
        self, block_fn, planes, block_num, stride=1, group_norm_num_groups=None
    ):
        downsample = None
        if stride != 1 or self.inplanes != planes * block_fn.expansion:
            downsample = nn.Sequential(
                OrderedDict(
                    [
                        (
                            "conv",
                            nn.Conv2d(
                                self.inplanes,
                                planes * block_fn.expansion,
                                kernel_size=1,
                                stride=stride,
                                bias=False,
                            ),
                        ),
                        (
                            "bn",
                            norm2d(
                                group_norm_num_groups,
                                planes=planes * block_fn.expansion,
                            ),
                        ),
                    ]
                )
            )

        layers = []
        layers.append(
            block_fn(
                in_planes=self.inplanes,
                out_planes=planes,
                stride=stride,
                downsample=downsample,
                group_norm_num_groups=group_norm_num_groups,
            )
        )
        self.inplanes = planes * block_fn.expansion

        for _ in range(1, block_num):
            layers.append(
                block_fn(
                    in_planes=self.inplanes,
                    out_planes=planes,
                    group_norm_num_groups=group_norm_num_groups,
                )
            )
        return nn.Sequential(*layers)


class ResNetImagenet(ResNetBase):
    def __init__(
        self,
        dataset,
        depth,
        split_point="layer4",
        group_norm_num_groups=None,
        grad_checkpoint=False,
    ):
        super(ResNetImagenet, self).__init__()
        self.dataset = dataset
        assert split_point in ["layer3", "layer4", None], "invalid split position."
        self.split_point = split_point
        self.grad_checkpoint = grad_checkpoint

        # define model param.
        self.depth = depth
        model_params = {
            18: {"block": BasicBlock, "layers": [2, 2, 2, 2]},
            34: {"block": BasicBlock, "layers": [3, 4, 6, 3]},
            50: {"block": Bottleneck, "layers": [3, 4, 6, 3]},
            101: {"block": Bottleneck, "layers": [3, 4, 23, 3]},
            152: {"block": Bottleneck, "layers": [3, 8, 36, 3]},
        }
        block_fn = model_params[depth]["block"]
        block_nums = model_params[depth]["layers"]

        # decide the num of classes.
        self.num_classes = self._decide_num_classes()

        # define layers.
        self.inplanes = 64
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )
        self.bn1 = norm2d(group_norm_num_groups, planes=64)
        self.relu = nn.ReLU(inplace=True)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_block(
            block_fn=block_fn,
            planes=64,
            block_num=block_nums[0],
            group_norm_num_groups=group_norm_num_groups,
        )
        self.layer2 = self._make_block(
            block_fn=block_fn,
            planes=128,
            block_num=block_nums[1],
            stride=2,
            group_norm_num_groups=group_norm_num_groups,
        )
        self.layer3 = self._make_block(
            block_fn=block_fn,
            planes=256,
            block_num=block_nums[2],
            stride=2,
            group_norm_num_groups=group_norm_num_groups,
        )
        self.layer4 = self._make_block(
            block_fn=block_fn,
            planes=512,
            block_num=block_nums[3],
            stride=2,
            group_norm_num_groups=group_norm_num_groups,
        )

        self.avgpool = nn.AvgPool2d(kernel_size=7, stride=1)
        self.classifier = nn.Linear(
            in_features=512 * block_fn.expansion,
            out_features=self.num_classes,
            bias=False,
        )

        # weight initialization based on layer type.
        self._weight_initialization()
        self.train()

    def forward_features(self, x):
        """Temporally fix the configuration of extrator. We will improve this function in the future version
        (e.g., support selecting the position where the auxiliary head begins)."""
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        if self.grad_checkpoint:
            x = checkpoint_seq(self.layer1, x, preserve_rng_state=True)
            x = checkpoint_seq(self.layer2, x, preserve_rng_state=True)
            x = checkpoint_seq(self.layer3, x, preserve_rng_state=True)
            if self.split_point in ["layer4", None]:
                x = checkpoint_seq(self.layer4, x, preserve_rng_state=True)
                x = self.avgpool(x)
                x = x.view(x.size(0), -1)
        else:
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            if self.split_point in ["layer4", None]:
                x = self.layer4(x)
                x = self.avgpool(x)
                x = x.view(x.size(0), -1)

        return x

    def forward_head(self, x):
        """Temporally fix the configuration of head. We will improve this function in the future version
        (e.g., support selecting the position where the auxiliary head begins)."""
        if self.split_point == "layer3":
            if self.grad_checkpoint:
                x = checkpoint_seq(self.layer4, x, preserve_rng_state=True)
                x = self.avgpool(x)
                x = x.view(x.size(0), -1)
            else:
                x = self.layer4(x)
                x = self.avgpool(x)
                x = x.view(x.size(0), -1)

        x = self.classifier(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


class ResNetCifar(ResNetBase):
    def __init__(
        self,
        dataset,
        depth,
        scaling=1,
        split_point="layer3",
        group_norm_num_groups=None,
        grad_checkpoint=False,
    ):
        super(ResNetCifar, self).__init__()
        self.dataset = dataset
        assert split_point in ["layer2", "layer3", None], "invalid split position."
        self.split_point = split_point
        self.grad_checkpoint = grad_checkpoint

        # define model.
        self.depth = depth
        if depth % 6 != 2:
            raise ValueError("depth must be 6n + 2:", depth)
        block_nums = (depth - 2) // 6
        block_fn = Bottleneck if depth >= 44 else BasicBlock
        self.block_nums = block_nums
        self.block_fn_name = "Bottleneck" if depth >= 44 else "BasicBlock"

        # decide the num of classes.
        self.num_classes = self._decide_num_classes()

        # define layers.
        self.inplanes = int(16 * scaling)
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=int(16 * scaling),
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn1 = norm2d(group_norm_num_groups, planes=int(16 * scaling))
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_block(
            block_fn=block_fn,
            planes=int(16 * scaling),
            block_num=block_nums,
            group_norm_num_groups=group_norm_num_groups,
        )
        self.layer2 = self._make_block(
            block_fn=block_fn,
            planes=int(32 * scaling),
            block_num=block_nums,
            stride=2,
            group_norm_num_groups=group_norm_num_groups,
        )
        self.layer3 = self._make_block(
            block_fn=block_fn,
            planes=int(64 * scaling),
            block_num=block_nums,
            stride=2,
            group_norm_num_groups=group_norm_num_groups,
        )

        self.avgpool = nn.AvgPool2d(kernel_size=8)
        self.classifier = nn.Linear(
            in_features=int(64 * scaling * block_fn.expansion),
            out_features=self.num_classes,
            bias=False,
        )

        # weight initialization based on layer type.
        self._weight_initialization()
        self.train()

    def forward_features(self, x):
        """Temporally fix the configuration of extrator. We will improve this function in the future version
        (e.g., support selecting the position where the auxiliary head begins)."""
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        if self.grad_checkpoint:
            x = checkpoint_seq(self.layer1, x, preserve_rng_state=True)
            x = checkpoint_seq(self.layer2, x, preserve_rng_state=True)
            if self.split_point in ["layer3", None]:
                x = checkpoint_seq(self.layer3, x, preserve_rng_state=True)
                x = self.avgpool(x)
                x = x.view(x.size(0), -1)
        else:
            x = self.layer1(x)
            x = self.layer2(x)
            if self.split_point in ["layer3", None]:
                x = self.layer3(x)
                x = self.avgpool(x)
                x = x.view(x.size(0), -1)

        return x

    def forward_head(self, x):
        """Temporally fix the configuration of head. We will improve this function in the future version
        (e.g., support selecting the position where the auxiliary head begins)."""
        if self.split_point == "layer2":
            if self.grad_checkpoint:
                x = checkpoint_seq(self.layer3, x, preserve_rng_state=True)
                x = self.avgpool(x)
                x = x.view(x.size(0), -1)
            else:
                x = self.layer3(x)
                x = self.avgpool(x)
                x = x.view(x.size(0), -1)

        x = self.classifier(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


class ResNetMNIST(ResNetBase):
    def __init__(
        self,
        dataset,
        depth,
        split_point="layer4",
        scaling=1,
        group_norm_num_groups=None,
    ):
        super(ResNetMNIST, self).__init__()
        self.dataset = dataset
        assert split_point in ["layer3", "layer4", None], "invalid split position."
        self.split_point = split_point
        if self.dataset == "coloredmnist":
            in_dim = 3
        else:
            in_dim = 1

        # define model.
        self.depth = depth
        model_params = {
            18: {"block": BasicBlock, "layers": [2, 2, 2, 2]},
            34: {"block": BasicBlock, "layers": [3, 4, 6, 3]},
            50: {"block": Bottleneck, "layers": [3, 4, 6, 3]},
            101: {"block": Bottleneck, "layers": [3, 4, 23, 3]},
            152: {"block": Bottleneck, "layers": [3, 8, 36, 3]},
        }
        block_fn = model_params[depth]["block"]
        block_nums = model_params[depth]["layers"]

        # decide the num of classes.
        self.num_classes = self._decide_num_classes()

        # define layers.
        self.inplanes = 64
        self.conv1 = nn.Conv2d(
            in_channels=in_dim,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )
        self.bn1 = norm2d(group_norm_num_groups, planes=64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_block(
            block_fn=block_fn,
            planes=64,
            block_num=block_nums[0],
            group_norm_num_groups=group_norm_num_groups,
        )
        self.layer2 = self._make_block(
            block_fn=block_fn,
            planes=128,
            block_num=block_nums[1],
            stride=2,
            group_norm_num_groups=group_norm_num_groups,
        )
        self.layer3 = self._make_block(
            block_fn=block_fn,
            planes=256,
            block_num=block_nums[2],
            stride=2,
            group_norm_num_groups=group_norm_num_groups,
        )
        self.layer4 = self._make_block(
            block_fn=block_fn,
            planes=512,
            block_num=block_nums[3],
            stride=2,
            group_norm_num_groups=group_norm_num_groups,
        )

        self.classifier = nn.Linear(
            in_features=int(512 * block_fn.expansion),
            out_features=self.num_classes,
            bias=False,
        )

        # weight initialization based on layer type.
        self._weight_initialization()
        self.train()

    def forward_features(self, x):
        """Temporally fix the configuration of extrator. We will improve this function in the future version
        (e.g., support selecting the position where the auxiliary head begins)."""
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        if self.split_point in ["layer4", None]:
            x = self.layer4(x)
            x = x.view(x.size(0), -1)

        return x

    def forward_head(self, x):
        """Temporally fix the configuration of head. We will improve this function in the future version
        (e.g., support selecting the position where the auxiliary head begins)."""
        if self.split_point == "layer3":
            x = self.layer4(x)
            x = x.view(x.size(0), -1)

        x = self.classifier(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x

    # def forward(self, x):
    #     x = self.conv1(x)
    #     x = self.bn1(x)
    #     x = self.relu(x)
    #     x = self.maxpool(x)

    #     x = self.layer1(x)
    #     x = self.layer2(x)
    #     x = self.layer3(x)
    #     x = self.layer4(x)
    #     # x = self.avgpool(x)
    #     x = x.view(x.size(0), -1)
    #     return self.classifier(x)


def resnet(
    dataset,
    depth,
    split_point=None,
    scaling=1,
    group_norm_num_groups=None,
    grad_checkpoint=False,
):
    if "cifar" in dataset:
        return ResNetCifar(
            dataset, depth, scaling, split_point, group_norm_num_groups, grad_checkpoint
        )
    elif "imagenet" in dataset:
        return ResNetImagenet(
            dataset, depth, split_point, group_norm_num_groups, grad_checkpoint
        )
    elif "office" in dataset:
        return ResNetImagenet(
            dataset, depth, split_point, group_norm_num_groups, grad_checkpoint
        )
    elif "pacs" in dataset:
        return ResNetImagenet(
            dataset, depth, split_point, group_norm_num_groups, grad_checkpoint
        )
    elif "mnist" in dataset:
        return ResNetMNIST(dataset, depth, split_point, scaling, group_norm_num_groups)
    elif "waterbirds" in dataset:
        return ResNetImagenet(
            dataset, depth, split_point, group_norm_num_groups, grad_checkpoint
        )
