import torch.nn
import torch.nn as nn
from torch import Tensor
from typing import Type, Any, Callable, List, Optional, Tuple, Union
from utils import *

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None, w_bits=32, a_bits=32,g_bits=32, g_q=False):
        super(BasicBlock, self).__init__()

        # self.conv1 = nn.Conv2d(in_planes, planes, 3, stride=stride, padding=1, bias=False)
        self.quant_conv1 = QuantizationConv2d(in_planes, planes, 3, stride=stride, padding=1, bias=False, w_bits=w_bits, g_bits=g_bits, g_q=g_q)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.quant_activation = QuantizationActivation(a_bits, g_bits=g_bits, g_q=g_q)

        # self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.quant_conv2 = QuantizationConv2d(planes, planes, 3, padding=1, bias=False, w_bits=w_bits, g_bits=g_bits, g_q=g_q)
        self.bn2 = nn.BatchNorm2d(planes)

        self.downsample= downsample

    def forward(self, x):
        residual = x

        out = self.quant_conv1(x)
        # out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.quant_activation(out)

        out = self.quant_conv2(out)
        # out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        out = self.quant_activation(out)

        return out


class ResNet20(torch.nn.Module):
    """
    Cifar10 데이터셋을 활용하기 위한 ResNet20 모델.

    Fitst Layer, Last Layer는 양자화를 진행하지 않는다(FP32).

    Layer 순서는 Quant_Conv -> BN -> ReLU -> Quant_Activation

    Reference:
            Paper 참조(3~4page): "https://arxiv.org/pdf/1512.03385.pdf"
    """
    def __init__(
        self,
        block: BasicBlock,
        layers: List[int],
        w_bits: int,
        a_bits: int,
        g_bits: int,
        num_classes: int=10,
        RRM: bool=False,
        g_q: bool=False
    ) -> None:
        super(ResNet20, self).__init__()

        self.w_bits = w_bits
        self.a_bits = a_bits
        self.in_planes = 16
        self.g_bits=g_bits
        self.RRM=RRM
        self.g_q=g_q
        self.conv = nn.Conv2d(3, self.in_planes, 3, 1, 1, bias=False)
        self.bn = nn.BatchNorm2d(self.in_planes)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block, 16, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64 * block.expansion, num_classes)

    def _make_layer(self, block: Type[BasicBlock], planes: int, num_blocks: int, stride: int) -> nn.Sequential:
        downsample = None
        layers = []

        if (stride != 1) or (self.in_planes != planes * block.expansion): # block.expansion은 항상 1
            downsample = nn.Sequential(
                QuantizationConv2d(self.in_planes, planes, 1, stride=stride, w_bits=self.w_bits, g_bits=self.g_bits, g_q=self.g_q),
                nn.BatchNorm2d(planes)
            )

        layers.append(block(self.in_planes, planes, stride, downsample, w_bits=self.w_bits, a_bits=self.a_bits, g_bits=self.g_bits, g_q=self.g_q))

        self.in_planes = planes * block.expansion

        for num_block in range(1, num_blocks):
            layers.append(block(self.in_planes, planes, w_bits=self.w_bits, a_bits=self.a_bits, g_bits=self.g_bits, g_q=self.g_q))

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)

        out = self.avgpool(out)
        out = torch.flatten(out, 1)

        out = self.fc(out)

        return out

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


def resnet20(w_bits, a_bits,g_bits,num_classes,rrm,g_q, **kwargs):
    """
    ResNet-20 모델 함수.

    Parameters
    ----------
    w_bits: int
        가중치 양자화 bits

    a_bits: int
        활성함수 양자화 bits
    """
    model = ResNet20(BasicBlock, [3, 3, 3], w_bits, a_bits,g_bits,num_classes,rrm,g_q, **kwargs)

    return model


if __name__ == '__main__':
    w_bits, a_bits = 32, 32
    model = resnet20(w_bits, a_bits)
    print(model)

    for i, (name, param) in enumerate(model.named_parameters()):
        print(param)
        print(name)

        if i == 3:
            break