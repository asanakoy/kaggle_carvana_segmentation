import torch
import torch.nn as nn


class Unet4(nn.Module):
    def __init__(self, feature_scale=1, n_classes=1, is_deconv=True, in_channels=3,
                 is_batchnorm=True, filters=None):
        super(Unet4, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale

        if filters is None:
            filters = [64, 128, 256, 512, 1024]
        print 'Unet4 filter sizes:', filters

        filters = [x / self.feature_scale for x in filters]

        self.down1 = UnetDown(self.in_channels, filters[0], self.is_batchnorm)
        self.down2 = UnetDown(filters[0], filters[1], self.is_batchnorm)
        self.down3 = UnetDown(filters[1], filters[2], self.is_batchnorm)
        self.down4 = UnetDown(filters[2], filters[3], self.is_batchnorm)

        self.center = UnetConvBlock(filters[3], filters[4], self.is_batchnorm)

        self.up4 = UnetUp(filters[4], filters[3], self.is_deconv,
                          is_batch_norm=self.is_batchnorm)
        self.up3 = UnetUp(filters[3], filters[2], self.is_deconv,
                          is_batch_norm=self.is_batchnorm)
        self.up2 = UnetUp(filters[2], filters[1], self.is_deconv,
                          is_batch_norm=self.is_batchnorm)
        self.up1 = UnetUp(filters[1], filters[0], self.is_deconv,
                          is_batch_norm=self.is_batchnorm)
        self.final = nn.Conv2d(filters[0], n_classes, kernel_size=1)

    def forward(self, inputs):
        res1, out = self.down1(inputs)
        res2, out = self.down2(out)
        res3, out = self.down3(out)
        res4, out = self.down4(out)
        out = self.center(out)
        out = self.up4(res4, out)
        out = self.up3(res3, out)
        out = self.up2(res2, out)
        out = self.up1(res1, out)
        return self.final(out)


class Unet5(nn.Module):
    def __init__(self, feature_scale=1, n_classes=1, is_deconv=True, in_channels=3,
                 is_batchnorm=True, filters=None):
        super(Unet5, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale

        if filters is None:
            filters = [64, 128, 256, 512, 1024, 1024]
        print 'Unet5 filter sizes:', filters

        filters = [x / self.feature_scale for x in filters]

        self.down1 = UnetDown(self.in_channels, filters[0], self.is_batchnorm)
        self.down2 = UnetDown(filters[0], filters[1], self.is_batchnorm)
        self.down3 = UnetDown(filters[1], filters[2], self.is_batchnorm)
        self.down4 = UnetDown(filters[2], filters[3], self.is_batchnorm)
        self.down5 = UnetDown(filters[3], filters[4], self.is_batchnorm)

        self.center = UnetConvBlock(filters[4], filters[5], self.is_batchnorm)

        self.up5 = UnetUp(filters[5], filters[4], self.is_deconv, is_batch_norm=self.is_batchnorm)
        self.up4 = UnetUp(filters[4], filters[3], self.is_deconv, is_batch_norm=self.is_batchnorm)
        self.up3 = UnetUp(filters[3], filters[2], self.is_deconv, is_batch_norm=self.is_batchnorm)
        self.up2 = UnetUp(filters[2], filters[1], self.is_deconv, is_batch_norm=self.is_batchnorm)
        self.up1 = UnetUp(filters[1], filters[0], self.is_deconv, is_batch_norm=self.is_batchnorm)
        self.final = nn.Conv2d(filters[0], n_classes, kernel_size=1)

    def forward(self, inputs):
        res1, out = self.down1(inputs)
        res2, out = self.down2(out)
        res3, out = self.down3(out)
        res4, out = self.down4(out)
        res5, out = self.down5(out)
        out = self.center(out)
        out = self.up5(res5, out)
        out = self.up4(res4, out)
        out = self.up3(res3, out)
        out = self.up2(res2, out)
        out = self.up1(res1, out)
        return self.final(out)


class Unet(nn.Module):

    def __init__(self, feature_scale=1, n_classes=1, is_deconv=True, in_channels=3,
                 is_batchnorm=True, filters=None):
        super(Unet, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale

        if filters is None:
            filters = [32, 64, 128, 256, 512, 1024, 1024]
        print 'Unet filter sizes:', filters

        filters = [x / self.feature_scale for x in filters]

        self.down1 = UnetDown(self.in_channels, filters[0], self.is_batchnorm)
        self.down2 = UnetDown(filters[0], filters[1], self.is_batchnorm)
        self.down3 = UnetDown(filters[1], filters[2], self.is_batchnorm)
        self.down4 = UnetDown(filters[2], filters[3], self.is_batchnorm)
        self.down5 = UnetDown(filters[3], filters[4], self.is_batchnorm)
        self.down6 = UnetDown(filters[4], filters[5], self.is_batchnorm)

        self.center = UnetConvBlock(filters[5], filters[6], self.is_batchnorm)

        self.up6 = UnetUp(filters[6], filters[5], self.is_deconv, is_batch_norm=self.is_batchnorm)
        self.up5 = UnetUp(filters[5], filters[4], self.is_deconv, is_batch_norm=self.is_batchnorm)
        self.up4 = UnetUp(filters[4], filters[3], self.is_deconv, is_batch_norm=self.is_batchnorm)
        self.up3 = UnetUp(filters[3], filters[2], self.is_deconv, is_batch_norm=self.is_batchnorm)
        self.up2 = UnetUp(filters[2], filters[1], self.is_deconv, is_batch_norm=self.is_batchnorm)
        self.up1 = UnetUp(filters[1], filters[0], self.is_deconv, is_batch_norm=self.is_batchnorm)
        self.final = nn.Conv2d(filters[0], n_classes, kernel_size=1)

    def forward(self, inputs):
        res1, out = self.down1(inputs)
        res2, out = self.down2(out)
        res3, out = self.down3(out)
        res4, out = self.down4(out)
        res5, out = self.down5(out)
        res6, out = self.down6(out)
        out = self.center(out)
        out = self.up6(res6, out)
        out = self.up5(res5, out)
        out = self.up4(res4, out)
        out = self.up3(res3, out)
        out = self.up2(res2, out)
        out = self.up1(res1, out)
        return self.final(out)


UNarrow = Unet


class Unet7(nn.Module):
    def __init__(self, feature_scale=1, n_classes=1, is_deconv=True, in_channels=3,
                 is_batchnorm=True, filters=None):
        super(Unet7, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale

        if filters is None:
            filters = [32, 64, 128, 256, 512, 1024, 1024, 2048]
        print 'Unet7 filter sizes:', filters

        filters = [x / self.feature_scale for x in filters]

        self.down1 = UnetDown(self.in_channels, filters[0], self.is_batchnorm)
        self.down2 = UnetDown(filters[0], filters[1], self.is_batchnorm)
        self.down3 = UnetDown(filters[1], filters[2], self.is_batchnorm)
        self.down4 = UnetDown(filters[2], filters[3], self.is_batchnorm)
        self.down5 = UnetDown(filters[3], filters[4], self.is_batchnorm)
        self.down6 = UnetDown(filters[4], filters[5], self.is_batchnorm)
        self.down7 = UnetDown(filters[5], filters[6], self.is_batchnorm)

        self.center = UnetConvBlock(filters[6], filters[7], self.is_batchnorm)

        self.up7 = UnetUp(filters[7], filters[6], self.is_deconv,
                          is_batch_norm=self.is_batchnorm)
        self.up6 = UnetUp(filters[6], filters[5], self.is_deconv,
                          is_batch_norm=self.is_batchnorm)
        self.up5 = UnetUp(filters[5], filters[4], self.is_deconv,
                          is_batch_norm=self.is_batchnorm)
        self.up4 = UnetUp(filters[4], filters[3], self.is_deconv,
                          is_batch_norm=self.is_batchnorm)
        self.up3 = UnetUp(filters[3], filters[2], self.is_deconv,
                          is_batch_norm=self.is_batchnorm)
        self.up2 = UnetUp(filters[2], filters[1], self.is_deconv,
                          is_batch_norm=self.is_batchnorm)
        self.up1 = UnetUp(filters[1], filters[0], self.is_deconv,
                          is_batch_norm=self.is_batchnorm)
        self.final = nn.Conv2d(filters[0], n_classes, kernel_size=1)

    def forward(self, inputs):
        res1, out = self.down1(inputs)
        res2, out = self.down2(out)
        res3, out = self.down3(out)
        res4, out = self.down4(out)
        res5, out = self.down5(out)
        res6, out = self.down6(out)
        res7, out = self.down7(out)
        out = self.center(out)
        out = self.up7(res7, out)
        out = self.up6(res6, out)
        out = self.up5(res5, out)
        out = self.up4(res4, out)
        out = self.up3(res3, out)
        out = self.up2(res2, out)
        out = self.up1(res1, out)
        return self.final(out)


class UnetConvBlock(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm, num_layers=2):
        super(UnetConvBlock, self).__init__()

        self.convs = nn.ModuleList()
        if is_batchnorm:
            conv = nn.Sequential(nn.Conv2d(in_size, out_size, 3, 1, padding=1),
                                 nn.BatchNorm2d(out_size),
                                 nn.ReLU())
            self.convs.append(conv)
            for i in xrange(1, num_layers):
                conv = nn.Sequential(nn.Conv2d(out_size, out_size, 3, 1, padding=1),
                                     nn.BatchNorm2d(out_size),
                                     nn.ReLU())
                self.convs.append(conv)
        else:
            conv = nn.Sequential(nn.Conv2d(in_size, out_size, 3, 1, padding=1),
                                 nn.ReLU())

            self.convs.append(conv)
            for i in xrange(1, num_layers):
                conv = nn.Sequential(nn.Conv2d(out_size, out_size, 3, 1, padding=1),
                                     nn.ReLU())
                self.convs.append(conv)

    def forward(self, inputs):
        outputs = inputs
        for conv in self.convs:
            outputs = conv(outputs)
        return outputs


class UnetDown(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm):
        super(UnetDown, self).__init__()
        self.conv = UnetConvBlock(in_size, out_size, is_batchnorm, num_layers=2)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, inputs):
        residual = self.conv(inputs)
        outputs = self.pool(residual)
        return residual, outputs


class UnetUp(nn.Module):
    def __init__(self, in_size, out_size, is_deconv=False, residual_size=None, is_batch_norm=False):
        super(UnetUp, self).__init__()
        if residual_size is None:
            residual_size = out_size
        if is_deconv:
            # TODO: fixme. Some dimensions could be wrong
            self.up = nn.ConvTranspose2d(in_size, in_size, kernel_size=2, stride=2)
            self.conv = UnetConvBlock(in_size + residual_size, out_size, is_batchnorm=is_batch_norm, num_layers=2)
        else:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear')
            self.conv = UnetConvBlock(in_size + residual_size, out_size, is_batchnorm=is_batch_norm, num_layers=3)
        # print 'UnetUp convBlock::{}->{}'.format(in_size + residual_size, out_size)

    def forward(self, residual, previous):
        upsampled = self.up(previous)
        # print 'previous ({}) -> upsampled ({})'.format(previous.size()[1], upsampled.size()[1])
        # print 'residual.size(), upsampled.size()', residual.size(), upsampled.size()
        result = self.conv(torch.cat([residual, upsampled], 1))
        # print 'Result size:', result.size()
        return result
