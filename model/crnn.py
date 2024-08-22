import torch.nn as nn
from torch.autograd import Variable

class Conv_2d(nn.Module):
    def __init__(self, input_channels, output_channels, shape=3, stride=1, pooling=2):
        super(Conv_2d, self).__init__()
        self.conv = nn.Conv2d(input_channels, output_channels, shape, stride=stride, padding=shape // 2)
        self.bn = nn.BatchNorm2d(output_channels)
        self.relu = nn.ReLU()
        self.mp = nn.MaxPool2d(pooling)

    def forward(self, x):
        out = self.mp(self.relu(self.bn(self.conv(x))))
        return out


class CRNN(nn.Module):
    def __init__(self, n_class=15):  # Changed to 15 classes for consistency
        super(CRNN, self).__init__()
        self.spec_bn = nn.BatchNorm2d(1)

        # CNN Layers
        self.layer1 = Conv_2d(1, 64, pooling=(2, 4))
        self.layer2 = Conv_2d(64, 128, pooling=(2, 4))
        self.layer3 = Conv_2d(128, 128, pooling=(2, 4))
        self.layer4 = Conv_2d(128, 128, pooling=(3, 5))
        self.layer5 = Conv_2d(128, 64, pooling=(4, 4))

        # RNN Layer
        self.gru = nn.GRU(64, 32, 2, batch_first=True)

        # Dense Layer
        self.dropout = nn.Dropout(0.5)
        self.dense = nn.Linear(32, n_class)

    def forward(self, x):
        # Spectrogram transformation
        # x = x.unsqueeze(1)
        x = self.spec_bn(x)

        # CNN
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)

        # RNN
        x = x.squeeze(2)
        x = x.squeeze(-1)
        x = x.permute(0, 2, 1)
        x, _ = self.gru(x)
        x = x[:, -1, :]

        # Dense
        x = self.dropout(x)
        x = self.dense(x)
        x = nn.Softmax(dim=1)(x)

        return x

# class CRNN(nn.Module):
#     def __init__(self, num_class=15):
#         super(CRNN, self).__init__()

#         # init bn
#         self.bn_init = nn.BatchNorm2d(1)

#         # layer 1
#         self.conv_1 = nn.Conv2d(1, 64, 3, padding=1)
#         self.bn_1 = nn.BatchNorm2d(64)
#         self.mp_1 = nn.MaxPool2d((2, 4))

#         # layer 2
#         self.conv_2 = nn.Conv2d(64, 128, 3, padding=1)
#         self.bn_2 = nn.BatchNorm2d(128)
#         self.mp_2 = nn.MaxPool2d((2, 4))

#         # layer 3
#         self.conv_3 = nn.Conv2d(128, 128, 3, padding=1)
#         self.bn_3 = nn.BatchNorm2d(128)
#         self.mp_3 = nn.MaxPool2d((2, 4))

#         # layer 4
#         self.conv_4 = nn.Conv2d(128, 128, 3, padding=1)
#         self.bn_4 = nn.BatchNorm2d(128)
#         self.mp_4 = nn.MaxPool2d((3, 5))

#         # layer 5
#         self.conv_5 = nn.Conv2d(128, 64, 3, padding=1)
#         self.bn_5 = nn.BatchNorm2d(64)
#         self.mp_5 = nn.MaxPool2d((4, 4))

#         # classifier
#         self.dense = nn.Linear(64, num_class)
#         self.dropout = nn.Dropout(0.5)

#     def forward(self, x):
#         # x = x.unsqueeze(1)

#         # init bn
#         x = self.bn_init(x)

#         # layer 1
#         x = self.mp_1(nn.ELU()(self.bn_1(self.conv_1(x))))

#         # layer 2
#         x = self.mp_2(nn.ELU()(self.bn_2(self.conv_2(x))))

#         # layer 3
#         x = self.mp_3(nn.ELU()(self.bn_3(self.conv_3(x))))

#         # layer 4
#         x = self.mp_4(nn.ELU()(self.bn_4(self.conv_4(x))))

#         # layer 5
#         x = self.mp_5(nn.ELU()(self.bn_5(self.conv_5(x))))

#         # classifier
#         x = x.view(x.size(0), -1)
#         x = self.dropout(x)
#         logit = nn.Sigmoid()(self.dense(x))

#         return logit
