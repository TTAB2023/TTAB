# -*- coding: utf-8 -*-
import os

import torch
from torch import nn
import torchvision.models as models

import timm

from ttab.loads.models import resnet, WideResNet
from ttab.loads.models.resnet import ResNetCifar, ResNetImagenet, ResNetMNIST
from ttab.loads.models.big_resnet import SupervisedContrastResNet, LinearClassifier
import ttab.model_adaptation.utils as adaptation_utils


class SelfSupervisedModel(nn.Module):
    """This class is built for ttt.

    It adds the rotation prediction self-supervision task to the original pretraining pipeline.
    """

    def __init__(self, model, config):
        super(SelfSupervisedModel, self).__init__()
        self._config = config
        self.main_model = model
        self.ext, self.head = self._define_head()
        self.ssh = adaptation_utils.ExtractorHead(self.ext, self.head)

    def _define_resnet_head(self):
        assert hasattr(
            self._config, "entry_of_shared_layers"
        ), "Need to set up the number of shared layers as feature extractor."

        if isinstance(self.main_model, ResNetImagenet):
            if (
                self._config.entry_of_shared_layers == "layer4"
                or self._config.entry_of_shared_layers == None
            ):
                ext = adaptation_utils.shared_ext_from_layer4(self.main_model)
                head = adaptation_utils.head_from_classifier(
                    self.main_model, self._config.dim_out
                )
            elif self._config.entry_of_shared_layers == "layer3":
                ext = adaptation_utils.shared_ext_from_layer3(self.main_model)
                head = adaptation_utils.head_from_last_layer1(
                    self.main_model, self._config.dim_out
                )
            elif self._config.entry_of_shared_layers == "layer2":
                ext = adaptation_utils.shared_ext_from_layer2(self.main_model)
                head = adaptation_utils.head_from_last_layer2(
                    self.main_model, self._config.dim_out
                )
        elif isinstance(self.main_model, ResNetCifar) or isinstance(
            self.main_model, WideResNet
        ):
            if (
                self._config.entry_of_shared_layers == "layer3"
                or self._config.entry_of_shared_layers == None
            ):
                ext = adaptation_utils.shared_ext_from_layer3(self.main_model)
                head = adaptation_utils.head_from_classifier(
                    self.main_model, self._config.dim_out
                )
            elif self._config.entry_of_shared_layers == "layer2":
                ext = adaptation_utils.shared_ext_from_layer2(self.main_model)
                head = adaptation_utils.head_from_last_layer1(
                    self.main_model, self._config.dim_out
                )
        elif isinstance(self.main_model, ResNetMNIST):
            if (
                self._config.entry_of_shared_layers == "layer4"
                or self._config.entry_of_shared_layers == None
            ):
                ext = adaptation_utils.shared_ext_from_layer4(self.main_model)
                head = adaptation_utils.head_from_classifier(
                    self.main_model, self._config.dim_out
                )
            elif self._config.entry_of_shared_layers == "layer3":
                ext = adaptation_utils.shared_ext_from_layer3(self.main_model)
                head = adaptation_utils.head_from_last_layer1(
                    self.main_model, self._config.dim_out
                )
            elif self._config.entry_of_shared_layers == "layer2":
                ext = adaptation_utils.shared_ext_from_layer2(self.main_model)
                head = adaptation_utils.head_from_last_layer2(
                    self.main_model, self._config.dim_out
                )
        else:
            raise ValueError(
                f"invalid configuration={self._config.entry_of_shared_layers}."
            )
            # if (
            #     self._config.entry_of_shared_layers == "layer4"
            #     or self._config.entry_of_shared_layers == None
            # ):
            #     ext = adaptation_utils.shared_ext_from_layer4(self.main_model)
            #     head = adaptation_utils.head_from_classifier(
            #         self.main_model, self._config.dim_out
            #     )
            # elif self._config.entry_of_shared_layers == "layer3":
            #     ext = adaptation_utils.shared_ext_from_layer3(self.main_model)
            #     head = adaptation_utils.head_from_last_layer1(
            #         self.main_model, self._config.dim_out
            #     )
            # elif self._config.entry_of_shared_layers == "layer2":
            #     ext = adaptation_utils.shared_ext_from_layer2(self.main_model)
            #     head = adaptation_utils.head_from_last_layer2(
            #         self.main_model, self._config.dim_out
            #     )
        return ext, head

    def _define_vit_head(self):
        ext = adaptation_utils.VitExtractor(self.main_model)
        head = nn.Linear(
            in_features=self.main_model.head.in_features,
            out_features=self._config.dim_out,
            bias=True,
        )
        return ext, head

    def _define_head(self):
        if "resnet" in self._config.model_name:
            return self._define_resnet_head()
        elif "vit" in self._config.model_name:
            return self._define_vit_head()

    def load_pretrained_parameters(self, ckpt_path):
        """This function helps to load pretrained parameters given the checkpoint path."""
        ckpt = torch.load(ckpt_path, map_location=self._config.device)
        self.main_model.load_state_dict(ckpt["model"])
        self.head.load_state_dict(ckpt["head"])


class SupervisedContrastModel(nn.Module):
    """This class is built for ttt++.

    It uses contrastive learning to adjust the pretraining pipeline.
    """

    def __init__(self, config):
        super(SupervisedContrastModel, self).__init__()
        self._config = config
        self.ssh = SupervisedContrastResNet()
        self.ext = self.ssh.encoder
        self.head = self.ssh.head
        self.classifier = LinearClassifier(
            num_classes=self._config.statistics["n_classes"]
        )
        self.main_model = adaptation_utils.ExtractorHead(self.ext, self.classifier)

    def load_pretrained_parameters(self, ckpt_path):
        """This function helps to load pretrained parameters given the checkpoint path.
        The ability of this function is limited because we only have one officially resnet50 pretrained model for ttt++.
        """
        ckpt = torch.load(ckpt_path, map_location=self._config.device)
        state_dict = ckpt["model"]

        model_dict = {}
        head_dict = {}
        for k, v in state_dict.items():
            if k[:4] == "head":
                k = k.replace("head.", "")
                head_dict[k] = v
            else:
                k = k.replace("encoder.", "ext.")
                k = k.replace("fc.", "head.fc.")
                model_dict[k] = v

        self.main_model.load_state_dict(model_dict)
        self.head.load_state_dict(head_dict)
        self.source_statistics = ckpt["source_domain_statistics"]


def define_model(config):
    if "imagenet" in config.data_names:
        init_model = models.resnet50(pretrained=True)
        return init_model
    elif "wideresnet" in config.model_name:
        components = config.model_name.split("_")
        depth = int(components[0].replace("wideresnet", ""))
        widen_factor = int(components[1])

        init_model = WideResNet(
            depth,
            widen_factor,
            config.statistics["n_classes"],
            split_point=config.entry_of_shared_layers,
            dropout_rate=0.0,
        )
        if config.model_adaptation_method == "ttt":
            return SelfSupervisedModel(init_model, config)
        else:
            return init_model
    elif "resnet" in config.model_name:
        # if config.base_data_name == "officehome":
        #     # For temporary use (resnet50, OfficeHome dataset).
        #     model = models.resnet50(pretrained=False)
        #     model.fc = nn.Linear(
        #         in_features=model.fc.in_features, out_features=65, bias=False
        #     )
        #     return model
        # else:
        depth = int(config.model_name.replace("resnet", ""))

        if config.model_adaptation_method == "ttt":
            init_model = resnet(
                config.base_data_name,
                depth,
                split_point=config.entry_of_shared_layers,
                group_norm_num_groups=config.group_norm_num_groups,
                grad_checkpoint=config.grad_checkpoint,
            )
            return SelfSupervisedModel(init_model, config)
        elif config.model_adaptation_method == "ttt_plus_plus":
            return SupervisedContrastModel(config)
        elif config.model_adaptation_method == "memo":
            return resnet(
                config.base_data_name,
                depth,
                group_norm_num_groups=config.group_norm_num_groups,
                grad_checkpoint=config.grad_checkpoint,
            )
        else:
            return resnet(
                config.base_data_name,
                depth,
                group_norm_num_groups=config.group_norm_num_groups,
            )
    elif "vit" in config.model_name:
        if config.model_adaptation_method == "ttt":
            init_model = timm.create_model(config.model_name, pretrained=False)
            init_model.head = nn.Linear(
                init_model.head.in_features, config.statistics["n_classes"]
            )
            if config.grad_checkpoint:
                init_model.set_grad_checkpointing()
            return SelfSupervisedModel(init_model, config)
        elif config.model_adaptation_method == "ttt_plus_plus":
            return SupervisedContrastModel(config)
        else:
            model = timm.create_model(config.model_name, pretrained=False)
            model.head = nn.Linear(
                model.head.in_features, config.statistics["n_classes"]
            )
            if config.grad_checkpoint:
                model.set_grad_checkpointing()
            return model
    else:
        raise NotImplementedError(f"invalid model_name={config.model_name}.")


def remove_prefix(state_dict, prefix):
    new_state_dict = {}
    for key, value in state_dict.items():
        if prefix in key:
            new_key = key.replace(prefix, "")
            new_state_dict[new_key] = value
    return new_state_dict


def load_pretrained_model(config, model):
    """load pretrained model params."""

    # safety check.
    assert os.path.exists(
        config.ckpt_path
    ), "The user-provided path for the checkpoint does not exist."

    # load parameters.
    if hasattr(config, "iabn") and config.iabn:
        # check IABN layers
        iabn_flag = False
        for name_module, module in model.named_modules():
            if isinstance(
                module, adaptation_utils.InstanceAwareBatchNorm2d
            ) or isinstance(module, adaptation_utils.InstanceAwareBatchNorm1d):
                iabn_flag = True
        if not iabn_flag:
            return

    # TODO: add some comments.
    if isinstance(model, SelfSupervisedModel) or isinstance(
        model, SupervisedContrastModel
    ):
        model.load_pretrained_parameters(config.ckpt_path)
    elif "imagenet" in config.data_names:
        pass
    else:
        ckpt = torch.load(config.ckpt_path, map_location=config.device)
        model.load_state_dict(ckpt["model"])  # ignore the auxiliary branch.
        # try:
        #     model.load_state_dict(ckpt["model"])  # ignore the auxiliary branch.
        # except:
        #     # new_state_dict = remove_prefix(ckpt["model"], prefix="module.")
        #     # model.load_state_dict(new_state_dict)
        #     model.load_state_dict(ckpt)


def update_pytorch_model_key(state_dict):
    """This function is used to modify the state dict key of pretrained model from pytorch."""
    new_state_dict = {}
    for key, value in state_dict.items():
        if "downsample" in key:
            name_split = key.split(".")
            if name_split[-2] == "0":
                name_split[-2] = "conv"
                new_key = ".".join(name_split)
                new_state_dict[new_key] = value
            elif name_split[-2] == "1":
                name_split[-2] = "bn"
                new_key = ".".join(name_split)
                new_state_dict[new_key] = value
        elif "fc" in key:
            name_split = key.split(".")
            if name_split[0] == "fc":
                name_split[0] = "classifier"
                new_key = ".".join(name_split)
                new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value

    return new_state_dict
