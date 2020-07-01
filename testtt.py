import torch
from PIL import Image
from torch import nn
import numpy as np
from torchvision.transforms import transforms


class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        conv_block = [nn.ReflectionPad2d(1),
                      nn.Conv2d(in_features, in_features, 3),
                      nn.InstanceNorm2d(in_features),  # Нормализация чутьчуть покруче чем LayerNorm
                      nn.ReLU(inplace=True),
                      nn.ReflectionPad2d(1),
                      nn.Conv2d(in_features, in_features, 3),
                      nn.InstanceNorm2d(in_features)]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)


class CycleGenerator(nn.Module):
    def __init__(self, conv_dim, output_nc=3, n_res_blocks=9):
        super(CycleGenerator, self).__init__()

        model = [nn.ReflectionPad2d(3),  # Начальный блок свертки
                 nn.Conv2d(conv_dim, 64, 7),
                 nn.InstanceNorm2d(64),
                 nn.ReLU(inplace=True)]

        in_features = 64  # Свертка
        out_features = in_features * 2
        for _ in range(2):
            model += [nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                      nn.InstanceNorm2d(out_features),
                      nn.ReLU(inplace=True)]
            in_features = out_features
            out_features = in_features * 2

        for _ in range(n_res_blocks):  # Residual blocks
            model += [ResidualBlock(in_features)]

        out_features = in_features // 2  # Развертка
        for _ in range(2):
            model += [nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                      nn.InstanceNorm2d(out_features),
                      nn.ReLU(inplace=True)]
            in_features = out_features
            out_features = in_features // 2

        model += [nn.ReflectionPad2d(3),  # Выходные слои
                  nn.Conv2d(64, output_nc, 7),
                  nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


def scale(x, feature_range=(-1, 1)):                                            # Маштабируем изображение в пределах (-1,1) вместо (0,1)
    min, max = feature_range
    x = x * (max - min) + min
    return x


def to_data(x):                                                                 # Переводит переменные в numpy
    x = x.cpu().data.numpy()
    x = ((x + 1)*255 / (2)).astype(np.uint8)                                     # rescale to 0-255
    return x[0].transpose((1, 2, 0))


def get_img(path):
  img = Image.open(path)
  img = transforms.Resize(256, Image.BICUBIC)(img)
  img = torch.tensor(np.array(img).transpose((2, 0, 1)), dtype=torch.float)
  return img
