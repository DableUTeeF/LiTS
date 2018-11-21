import torch
from torch import nn
import torchvision


class TimeDistDense201(nn.Module):
    def __init__(self, pretrained, classes):
        super().__init__()
        self.densenet = self.convert_top(classes, pretrained)
        self.lstm1 = nn.LSTM(224,
                             hidden_size=224,
                             num_layers=1,
                             bias=False,
                             batch_first=True)
        self.lstm2 = nn.LSTM(224,
                             hidden_size=224,
                             num_layers=1,
                             bias=False,
                             batch_first=True)

    @staticmethod
    def convert_top(classes, pretrained):
        model = torchvision.models.densenet201(pretrained=pretrained)
        model.conv0 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        if classes == 1000:
            return model
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, classes)
        return model

    def forward(self, x):
        x = x.view(x.size(0), -1, 224)
        x, _ = self.lstm1(x)
        x = x.view(x.size(0), 1, 224, 224)
        x = x.permute(0, 1, 3, 2)
        x = x.contiguous().view(x.size(0), 224, 224)
        x, _ = self.lstm2(x)
        x = x.view(x.size(0), 3, -1, 224)
        x = self.densenet(x)
        return x


class Ublock(nn.Module):
    def __init__(self, in_channel, out_channel, depth):
        super().__init__()
        self.depth = depth
        self.conv1 = nn.Conv2d(in_channel, out_channel, 3, padding=1, stride=2, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.conv2 = nn.Conv2d(out_channel, out_channel, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        if depth > 0:
            self.unet = Ublock(out_channel, out_channel*2, depth-1)
            self.upsampling = nn.UpsamplingNearest2d(scale_factor=2)
            self.conv3 = nn.Conv2d(out_channel, out_channel, 3, padding=1, bias=False)
            self.bn3 = nn.BatchNorm2d(out_channel)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = nn.ReLU()(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = nn.ReLU()(out)
        if self.depth > 0:
            out = self.upsampling(out)
            out = self.conv3(out)
            out = self.bn3(out)
            out = nn.ReLU()(out)
            out += x
        return out


class Unet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.unet = Ublock(32, 32, 5)
        self.conv2 = nn.Conv2d(32, 3, 1, bias=False)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = nn.ReLU()(out)
        out = self.unet(out)
        out += x
        out = self.conv2(out)
        return out
