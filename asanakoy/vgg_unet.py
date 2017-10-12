import torch
import torch.nn as nn
import torchvision.models
import torch.nn.functional as F


class UnetVgg11(nn.Module):
    def __init__(self, n_classes=1, num_filters=64, v=1):
        super(UnetVgg11, self).__init__()
        print 'UnetVgg11 version={}'.format(v)
        print 'base num_filters={}'.format(num_filters)

        self.pool = nn.MaxPool2d(2, 2)
        self.encoder = torchvision.models.vgg11(pretrained=True).features
        self.relu = self.encoder[1]
        self.conv1 = self.encoder[0]
        self.conv2 = self.encoder[3]
        self.conv3s = self.encoder[6]
        self.conv3 = self.encoder[8]
        self.conv4s = self.encoder[11]
        self.conv4 = self.encoder[13]
        self.conv5s = self.encoder[16]
        self.conv5 = self.encoder[18]
        if v == 1:
            self.center = DecoderBlock(num_filters * 8, num_filters * 16, num_filters * 8)
            self.dec5 = DecoderBlock(num_filters * 16, num_filters * 8, num_filters * 8)
            self.dec4 = DecoderBlock(num_filters * 16, num_filters * 8, num_filters * 4)
            self.dec3 = DecoderBlock(num_filters * 8, num_filters * 4, num_filters * 2)
            self.dec2 = DecoderBlock(num_filters * 4, num_filters * 2, num_filters)
            self.dec1 = ConvRelu(num_filters * 2, num_filters)
        elif v == 2:
            self.center = DecoderBlock(num_filters * 8, num_filters * 16, num_filters * 8, is_deconv=False)
            self.dec5 = DecoderBlock(num_filters * 16, num_filters * 8, num_filters * 8, is_deconv=False)
            self.dec4 = DecoderBlock(num_filters * 16, num_filters * 8, num_filters * 4, is_deconv=False)
            self.dec3 = DecoderBlock(num_filters * 8, num_filters * 4, num_filters * 2, is_deconv=False)
            self.dec2 = DecoderBlock(num_filters * 4, num_filters * 2, num_filters, is_deconv=False)
            self.dec1 = ConvRelu(num_filters * 2, num_filters)
        else:
            raise NotImplementedError('Unknown v={}'.format(v))

        self.final = nn.Conv2d(num_filters, n_classes, kernel_size=1)

    def forward(self, x):
        conv1 = self.relu(self.conv1(x))
        conv2 = self.relu(self.conv2(self.pool(conv1)))
        conv3s = self.relu(self.conv3s(self.pool(conv2)))
        conv3 = self.relu(self.conv3(conv3s))
        conv4s = self.relu(self.conv4s(self.pool(conv3)))
        conv4 = self.relu(self.conv4(conv4s))
        conv5s = self.relu(self.conv5s(self.pool(conv4)))
        conv5 = self.relu(self.conv5(conv5s))

        center = self.center(self.pool(conv5))

        # print 'dec5.in_channels', self.dec5.in_channels
        # print center.size(), conv5.size()
        dec5 = self.dec5(torch.cat([center, conv5], 1))
        dec4 = self.dec4(torch.cat([dec5, conv4], 1))
        dec3 = self.dec3(torch.cat([dec4, conv3], 1))
        dec2 = self.dec2(torch.cat([dec3, conv2], 1))
        dec1 = self.dec1(torch.cat([dec2, conv1], 1))
        return self.final(dec1)


class Vgg11a(nn.Module):
    def __init__(self, n_classes=1, num_filters=32, v=1):
        super(Vgg11a, self).__init__()
        assert v in [1, 2]
        print 'UnetVgg11a version={}'.format(v)

        self.pool = nn.MaxPool2d(2, 2)
        self.encoder = torchvision.models.vgg11(pretrained=True).features
        self.relu = self.encoder[1]
        self.conv1 = self.encoder[0]
        self.conv2 = self.encoder[3]
        self.conv3s = self.encoder[6]
        self.conv3 = self.encoder[8]
        self.conv4s = self.encoder[11]
        self.conv4 = self.encoder[13]
        self.conv5s = self.encoder[16]
        self.conv5 = self.encoder[18]

        self.center = DecoderBlock(num_filters * 8 * 2, num_filters * 8 * 2, num_filters * 8,
                                   is_deconv=v == 1)
        self.dec5 = DecoderBlock(num_filters * (16 + 8), num_filters * 8 * 2, num_filters * 8,
                                 is_deconv=v == 1)
        self.dec4 = DecoderBlock(num_filters * (16 + 8), num_filters * 8 * 2, num_filters * 4,
                                 is_deconv=v == 1)
        self.dec3 = DecoderBlock(num_filters * (8 + 4), num_filters * 4 * 2, num_filters * 2,
                                 is_deconv=v == 1)
        self.dec2 = DecoderBlock(num_filters * (4 + 2), num_filters * 2 * 2, num_filters,
                                 is_deconv=v == 1)
        self.dec1 = ConvRelu(num_filters * (2 + 1), num_filters)

        self.final = nn.Conv2d(num_filters, n_classes, kernel_size=1)

    def forward(self, x):
        conv1 = self.relu(self.conv1(x))
        conv2 = self.relu(self.conv2(self.pool(conv1)))
        conv3s = self.relu(self.conv3s(self.pool(conv2)))
        conv3 = self.relu(self.conv3(conv3s))
        conv4s = self.relu(self.conv4s(self.pool(conv3)))
        conv4 = self.relu(self.conv4(conv4s))
        conv5s = self.relu(self.conv5s(self.pool(conv4)))
        conv5 = self.relu(self.conv5(conv5s))

        center = self.center(self.pool(conv5))

        dec5 = self.dec5(torch.cat([center, conv5], 1))
        dec4 = self.dec4(torch.cat([dec5, conv4], 1))
        dec3 = self.dec3(torch.cat([dec4, conv3], 1))
        dec2 = self.dec2(torch.cat([dec3, conv2], 1))
        dec1 = self.dec1(torch.cat([dec2, conv1], 1))
        return self.final(dec1)


class ConvRelu(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvRelu, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            # nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class DoubleConvRelu(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConvRelu, self).__init__()

        self.block = nn.Sequential(
            ConvRelu(in_channels, out_channels),
            ConvRelu(out_channels, out_channels),
        )

    def forward(self, x):
        return self.block(x)


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, is_deconv=True):
        super(DecoderBlock, self).__init__()
        self.in_channels = in_channels

        if is_deconv:
            self.block = nn.Sequential(
                ConvRelu(in_channels, middle_channels),
                nn.ConvTranspose2d(middle_channels, out_channels, kernel_size=3, stride=2,
                                   padding=1, output_padding=1),
                nn.ReLU(inplace=True)
                # nn.Upsample(scale_factor=2)
            )
        else:
            self.block = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear'),
                ConvRelu(in_channels, middle_channels),
                ConvRelu(middle_channels, out_channels),
            )

    def forward(self, x):
        return self.block(x)


if __name__ == '__main__':
    model = torch.nn.DataParallel(UnetVgg11(n_classes=1)).cuda()
    pass