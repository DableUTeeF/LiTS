import torch
from torch import nn


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
        model = densenet201(pretrained=pretrained)
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

