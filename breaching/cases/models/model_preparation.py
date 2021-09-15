"""Helper code to instantiate various models."""

import torch
import torchvision

from .resnets import ResNet, resnet_depths_to_config
from .densenets import DenseNet, densenet_depths_to_config
from .nfnets import NFNet
from .vgg import VGG
from .imprint import imprint_model


def construct_model(cfg_model, cfg_data, pretrained=False):
    """Construct the neural net that is used."""
    channels = cfg_data.shape[0]
    classes = cfg_data.classes

    if cfg_data.name == 'ImageNet':
        try:
            model = getattr(torchvision.models, cfg_model.lower())(pretrained=pretrained)
        except AttributeError:
            if 'nfnet' in cfg_model:
                model = NFNet(channels, classes, variant='F0', stochdepth_rate=0.25, alpha=0.2, se_ratio=0.5,
                              activation='ReLU', stem='ImageNet', use_dropout=True)
            elif 'resnet50wsl' in cfg_model:
                model = torch.hub.load('facebookresearch/WSL-Images', 'resnext101_32x48d_wsl')
            elif 'resnet50swsl' in cfg_model:
                model = torch.hub.load('facebookresearch/semi-supervised-ImageNet1K-models', 'resnet50_swsl')
            elif 'resnet50ssl' in cfg_model:
                model = torch.hub.load('facebookresearch/semi-supervised-ImageNet1K-models', 'resnet50_ssl')
            elif 'linear' == cfg_model:
                input_dim = cfg_data.shape[0] * cfg_data.shape[1] * cfg_data.shape[2]
                model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(input_dim, classes))
            else:
                raise ValueError(f'Could not find ImageNet model {cfg_model} in torchvision.models or custom models.')
    else:
        # CIFAR Model from here:
        if 'resnet' in cfg_model.lower():
            block, layers = resnet_depths_to_config(int("".join(filter(str.isdigit, cfg_model))))
            model = ResNet(block, layers, channels, classes, stem='CIFAR', convolution_type='Standard',
                           nonlin='ReLU', norm='BatchNorm2d',
                           downsample='C', width_per_group=64,
                           zero_init_residual=False)
        elif 'densenet' in cfg_model.lower():
            growth_rate, block_config, num_init_features = densenet_depths_to_config(
                int("".join(filter(str.isdigit, cfg_model))))
            model = DenseNet(growth_rate=growth_rate,
                             block_config=block_config,
                             num_init_features=num_init_features,
                             bn_size=4,
                             drop_rate=0,
                             channels=channels,
                             num_classes=classes,
                             memory_efficient=False,
                             norm='BatchNorm2d',
                             nonlin='ReLU',
                             stem='CIFAR',
                             convolution_type='Standard')
        elif 'vgg' in cfg_model.lower():
            model = VGG(cfg_model, in_channels=channels, num_classes=classes, norm='BatchNorm2d',
                        nonlin='ReLU', head='CIFAR', convolution_type='Standard',
                        drop_rate=0, classical_weight_init=True)
        elif 'linear' in cfg_model:
            input_dim = cfg_data.shape[0] * cfg_data.shape[1] * cfg_data.shape[2]
            model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(input_dim, classes))
        elif 'imprint' in cfg_model:
            input_dim = cfg_data.shape[0] * cfg_data.shape[1] * cfg_data.shape[2]
            imprint = imprint_model(input_dim, 300) # 100 bins? Can be changed later :) hard coded for now
            model = torch.nn.Sequential(torch.nn.Flatten(), imprint)
        elif 'nfnet' in cfg_model:
            model = NFNet(channels, classes, variant='F0', stochdepth_rate=0.25, alpha=0.2, se_ratio=0.5,
                          activation='ReLU', stem='CIFAR', use_dropout=True)
        elif 'convnet' == cfg_model.lower():
            model = ConvNet(width=64, num_channels=channels, num_classes=classes)
        elif 'convnet_beyond' == cfg_model.lower():
            model = torch.nn.Sequential(dict([
                ('conv1', torch.nn.Conv2d(channels, 32, 3, stride=2, padding=1)),
                ('relu0', torch.nn.LeakyReLU()),
                ('conv2', torch.nn.Conv2d(32, 64, 3, stride=1, padding=1)),
                ('relu1', torch.nn.LeakyReLU()),
                ('conv3', torch.nn.Conv2d(64, 128, 3, stride=2, padding=1)),
                ('relu2', torch.nn.LeakyReLU()),
                ('conv4', torch.nn.Conv2d(128, 256, 3, stride=1, padding=1)),
                ('relu3', torch.nn.LeakyReLU()),
                ('flatt', torch.nn.Flatten()),
                ('linear0', torch.nn.Linear(12544, 12544)),
                ('relu4', torch.nn.LeakyReLU()),
                ('linear1', torch.nn.Linear(12544, classes)),
                ('softmax', torch.nn.Softmax(dim=1))
            ]))
        elif 'lenet_zhu' == cfg_model.lower():
            model = LeNetZhu(num_channels=channels, num_classes=classes)
        elif cfg_model == 'MLP':
            width = 1024
            model = torch.nn.Sequential(OrderedDict([
                ('flatten', torch.nn.Flatten()),
                ('linear0', torch.nn.Linear(3072, width)),
                ('relu0', torch.nn.ReLU()),
                ('linear1', torch.nn.Linear(width, width)),
                ('relu1', torch.nn.ReLU()),
                ('linear2', torch.nn.Linear(width, width)),
                ('relu2', torch.nn.ReLU()),
                ('linear3', torch.nn.Linear(width, classes))]))
        else:
            raise ValueError('Model could not be found.')

    return model

class ConvNet(torch.nn.Module):
    """ConvNetBN."""

    def __init__(self, width=32, num_classes=10, num_channels=3):
        """Init with width and num classes."""
        super().__init__()
        self.model = torch.nn.Sequential(dict([
            ('conv0', torch.nn.Conv2d(num_channels, 1 * width, kernel_size=3, padding=1)),
            ('bn0', torch.nn.BatchNorm2d(1 * width)),
            ('relu0', torch.nn.ReLU()),

            ('conv1', torch.nn.Conv2d(1 * width, 2 * width, kernel_size=3, padding=1)),
            ('bn1', torch.nn.BatchNorm2d(2 * width)),
            ('relu1', torch.nn.ReLU()),

            ('conv2', torch.nn.Conv2d(2 * width, 2 * width, kernel_size=3, padding=1)),
            ('bn2', torch.nn.BatchNorm2d(2 * width)),
            ('relu2', torch.nn.ReLU()),

            ('conv3', torch.nn.Conv2d(2 * width, 4 * width, kernel_size=3, padding=1)),
            ('bn3', torch.nn.BatchNorm2d(4 * width)),
            ('relu3', torch.nn.ReLU()),

            ('conv4', torch.nn.Conv2d(4 * width, 4 * width, kernel_size=3, padding=1)),
            ('bn4', torch.nn.BatchNorm2d(4 * width)),
            ('relu4', torch.nn.ReLU()),

            ('conv5', torch.nn.Conv2d(4 * width, 4 * width, kernel_size=3, padding=1)),
            ('bn5', torch.nn.BatchNorm2d(4 * width)),
            ('relu5', torch.nn.ReLU()),

            ('pool0', torch.nn.MaxPool2d(3)),

            ('conv6', torch.nn.Conv2d(4 * width, 4 * width, kernel_size=3, padding=1)),
            ('bn6', torch.nn.BatchNorm2d(4 * width)),
            ('relu6', torch.nn.ReLU()),

            ('conv7', torch.nn.Conv2d(4 * width, 4 * width, kernel_size=3, padding=1)),
            ('bn7', torch.nn.BatchNorm2d(4 * width)),
            ('relu7', torch.nn.ReLU()),

            ('pool1', torch.nn.MaxPool2d(3)),
            ('flatten', torch.nn.Flatten()),
            ('linear', torch.nn.Linear(36 * width, num_classes))
        ]))

    def forward(self, input):
        return self.model(input)

class LeNetZhu(torch.nn.Module):
    """LeNet variant from https://github.com/mit-han-lab/dlg/blob/master/models/vision.py."""

    def __init__(self, num_classes=10, num_channels=3):
        """3-Layer sigmoid Conv with large linear layer."""
        super().__init__()
        act = torch.nn.Sigmoid
        self.body = torch.nn.Sequential(
            torch.nn.Conv2d(num_channels, 12, kernel_size=5, padding=5 // 2, stride=2),
            act(),
            torch.nn.Conv2d(12, 12, kernel_size=5, padding=5 // 2, stride=2),
            act(),
            torch.nn.Conv2d(12, 12, kernel_size=5, padding=5 // 2, stride=1),
            act(),
        )
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(768, num_classes)
        )
        for module in self.modules():
            self.weights_init(module)

    @staticmethod
    def weights_init(m):
        if hasattr(m, "weight"):
            m.weight.data.uniform_(-0.5, 0.5)
        if hasattr(m, "bias"):
            m.bias.data.uniform_(-0.5, 0.5)

    def forward(self, x):
        out = self.body(x)
        out = out.view(out.size(0), -1)
        # print(out.size())
        out = self.fc(out)
        return out
