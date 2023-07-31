import torch.nn
import torch.nn as nn
from utils import *
class AlexNet(torch.nn.Module):
 
    def __init__(self, w_bits, a_bits, g_bits,num_classes=10,RRM=False,g_q=False):
        super(AlexNet, self).__init__()
        self.w_bits = w_bits
        self.a_bits = a_bits
        self.g_bits=g_bits
        self.g_q=g_q
        self.num_classes = num_classes

        # First and Last Layer는 Quantization 하지 않음
        self.first_layer = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2))

        if RRM: # Reducing Runtime Memory
            self.quantized_layer_1 = nn.Sequential(
                QuantizationConv2d(96, 256, kernel_size=5, padding=2, w_bits=self.w_bits,g_bits=self.g_bits,g_q=self.g_q),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                QuantizationActivation(self.a_bits,g_bits=self.g_bits,g_q=self.g_q),
                nn.MaxPool2d(3, 2),

                QuantizationConv2d(256, 384, kernel_size=3, w_bits=self.w_bits,g_bits=self.g_bits,g_q=self.g_q),
                nn.BatchNorm2d(384),
                nn.ReLU(inplace=True),
                QuantizationActivation(self.a_bits,g_bits=self.g_bits,g_q=self.g_q),

                QuantizationConv2d(384, 384, kernel_size=3, w_bits=self.w_bits,g_bits=self.g_bits,g_q=self.g_q),
                nn.BatchNorm2d(384),
                nn.ReLU(inplace=True),
                QuantizationActivation(self.a_bits,g_bits=self.g_bits,g_q=self.g_q),

                QuantizationConv2d(384, 256, kernel_size=3, w_bits=self.w_bits,g_bits=self.g_bits,g_q=self.g_q),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                QuantizationActivation(self.a_bits,g_bits=self.g_bits,g_q=self.g_q),
                nn.MaxPool2d(3, 2),

            )
        else:
            self.quantized_layer_1 = nn.Sequential(
                QuantizationConv2d(96, 256, kernel_size=5, padding=2, w_bits=self.w_bits,g_bits=self.g_bits,g_q=self.g_q),
                nn.BatchNorm2d(256),
                nn.MaxPool2d(3, 2),
                nn.ReLU(inplace=True),
                QuantizationActivation(self.a_bits,g_bits=self.g_bits,g_q=self.g_q),

                QuantizationConv2d(256, 384, kernel_size=3, w_bits=self.w_bits,g_bits=self.g_bits,g_q=self.g_q),
                nn.BatchNorm2d(384),
                nn.ReLU(inplace=True),
                QuantizationActivation(self.a_bits,g_bits=self.g_bits,g_q=self.g_q),

                QuantizationConv2d(384, 384, kernel_size=3, w_bits=self.w_bits,g_bits=self.g_bits,g_q=self.g_q),
                nn.BatchNorm2d(384),
                nn.ReLU(inplace=True),
                QuantizationActivation(self.a_bits,g_bits=self.g_bits,g_q=self.g_q),

                QuantizationConv2d(384, 256, kernel_size=3, w_bits=self.w_bits,g_bits=self.g_bits,g_q=self.g_q),
                nn.BatchNorm2d(256),
                nn.MaxPool2d(3, 2),
                nn.ReLU(inplace=True),
                QuantizationActivation(self.a_bits,g_bits=self.g_bits,g_q=self.g_q),

            )
        self.quantized_layer_2 = nn.Sequential(
            QuantizationFullyConnected(256*3*3, 4096, w_bits=self.w_bits,g_bits=self.g_bits,g_q=self.g_q),
            nn.ReLU(inplace=True),
            QuantizationActivation(self.a_bits,g_bits=self.g_bits,g_q=self.g_q),

            QuantizationFullyConnected(4096, 4096, w_bits=self.w_bits,g_bits=self.g_bits,g_q=self.g_q),
            nn.ReLU(inplace=True),
            QuantizationActivation(self.a_bits,g_bits=self.g_bits,g_q=self.g_q)
        )

        self.last_layer = nn.Linear(4096, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)

    def forward(self, x):
        x = self.first_layer(x)
        x = self.quantized_layer_1(x)
        x = torch.flatten(x, 1)
        x = self.quantized_layer_2(x)
        x = self.last_layer(x)
        return x
