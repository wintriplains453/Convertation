import math
import random
import torch
from torch import nn
from torch.nn import functional as F
from editings.styleclip.models.stylegan2.op import FusedLeakyReLU, fused_leaky_relu, upfirdn2d


def make_kernel(k):
    k = torch.tensor(k, dtype=torch.float32)

    if k.ndim == 1:
        k = k[None, :] * k[:, None]

    k /= k.sum()

    return k


class Upsample(nn.Module):
    def __init__(self, kernel, factor=2):
        super().__init__()

        self.factor = factor
        kernel = make_kernel(kernel) * (factor ** 2)
        self.register_buffer("kernel", kernel)

        p = kernel.shape[0] - factor

        pad0 = (p + 1) // 2 + factor - 1
        pad1 = p // 2

        self.pad = (pad0, pad1)

    def forward(self, input):
        out = upfirdn2d(input, self.kernel, up=self.factor, down=1, pad=self.pad)

        return out


class Downsample(nn.Module):
    def __init__(self, kernel, factor=2):
        super().__init__()

        self.factor = factor
        kernel = make_kernel(kernel)
        self.register_buffer("kernel", kernel)

        p = kernel.shape[0] - factor

        pad0 = (p + 1) // 2
        pad1 = p // 2

        self.pad = (pad0, pad1)

    def forward(self, input):
        out = upfirdn2d(input, self.kernel, up=1, down=self.factor, pad=self.pad)

        return out


class Blur(nn.Module):
    def __init__(self, kernel, pad, upsample_factor=1):
        super().__init__()

        kernel = make_kernel(kernel)

        if upsample_factor > 1:
            kernel = kernel * (upsample_factor ** 2)

        self.register_buffer("kernel", kernel)

        self.pad = pad

    def forward(self, input):
        out = upfirdn2d(input, self.kernel, pad=self.pad)

        return out


class EqualLinear(nn.Module):
    def __init__(
            self, in_dim, out_dim, bias=True, bias_init=0, lr_mul=1, activation=None
    ):
        super().__init__()

        self.weight = nn.Parameter(torch.randn(out_dim, in_dim).div_(lr_mul))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim).fill_(bias_init))

        else:
            self.bias = None

        self.activation = activation

        self.scale = (1 / math.sqrt(in_dim)) * lr_mul
        self.lr_mul = lr_mul

    def forward(self, input):
        if self.activation:
            out = F.linear(input, self.weight * self.scale)
            out = fused_leaky_relu(out, self.bias * self.lr_mul)

        else:
            out = F.linear(
                input, self.weight * self.scale, bias=self.bias * self.lr_mul
            )

        return out

    def __repr__(self):
        return (
            f"{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]})"
        )


class ModulatedConv2d(nn.Module):
    def __init__(
            self,
            in_channel,
            out_channel,
            kernel_size,
            style_dim,
            demodulate=True,
            upsample=False,
            downsample=False,
            blur_kernel=[1, 3, 3, 1],
    ):
        super().__init__()

        self.eps = 1e-8
        self.kernel_size = kernel_size
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.upsample = upsample
        self.downsample = downsample

        if upsample:
            factor = 2
            p = (len(blur_kernel) - factor) - (kernel_size - 1)
            pad0 = (p + 1) // 2 + factor - 1
            pad1 = p // 2 + 1

            self.blur = Blur(blur_kernel, pad=(pad0, pad1), upsample_factor=factor)

        if downsample:
            factor = 2
            p = (len(blur_kernel) - factor) + (kernel_size - 1)
            pad0 = (p + 1) // 2
            pad1 = p // 2

            self.blur = Blur(blur_kernel, pad=(pad0, pad1))

        fan_in = in_channel * kernel_size ** 2
        self.scale = 1 / math.sqrt(fan_in)
        self.padding = kernel_size // 2

        self.weight = nn.Parameter(
            torch.randn(1, out_channel, in_channel, kernel_size, kernel_size)
        )

        self.modulation = EqualLinear(style_dim, in_channel, bias_init=1)

        self.demodulate = demodulate

    def __repr__(self):
        return (
            f"{self.__class__.__name__}({self.in_channel}, {self.out_channel}, {self.kernel_size}, "
            f"upsample={self.upsample}, downsample={self.downsample})"
        )

    def forward(self, input, style):
        batch, in_channel, height, width = input.shape

        weight = self.weight

        style = style.view(batch, 1, in_channel, 1, 1)
        weight = self.scale * weight * style

        if self.demodulate:
            demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + 1e-8)
            weight = weight * demod.view(batch, self.out_channel, 1, 1, 1)

        weight = weight.view(
            batch * self.out_channel, in_channel, self.kernel_size, self.kernel_size
        )

        if self.upsample:
            input = input.view(1, batch * in_channel, height, width)
            weight = weight.view(
                batch, self.out_channel, in_channel, self.kernel_size, self.kernel_size
            )
            weight = weight.transpose(1, 2).reshape(
                batch * in_channel, self.out_channel, self.kernel_size, self.kernel_size
            )
            out = F.conv_transpose2d(input, weight, padding=0, stride=2, groups=batch)
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)
            out = self.blur(out)

        elif self.downsample:
            input = self.blur(input)
            _, _, height, width = input.shape
            input = input.view(1, batch * in_channel, height, width)
            out = F.conv2d(input, weight, padding=0, stride=2, groups=batch)
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)

        else:
            padding = self.padding
            input = input.view(1, batch * in_channel, height, width)
            out = F.conv2d(input, weight, padding=padding, groups=batch)
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)

        return out


class NoiseInjection(nn.Module):
    def __init__(self):
        super().__init__()

        self.weight = nn.Parameter(torch.zeros(1))

    def forward(self, image, noise=None):
        if noise is None:
            batch, _, height, width = image.shape
            noise = image.new_empty(batch, 1, height, width).normal_()

        return image + self.weight * noise


class ConstantInput(nn.Module):
    def __init__(self, channel, size=4):
        super().__init__()

        self.input = nn.Parameter(torch.randn(1, channel, size, size))

    def forward(self, input):
        batch = input.shape[0]
        out = self.input.repeat(batch, 1, 1, 1)

        return out


class StyledConv(nn.Module):
    def __init__(
            self,
            in_channel,
            out_channel,
            kernel_size,
            style_dim,
            upsample=False,
            blur_kernel=[1, 3, 3, 1],
            demodulate=True,
    ):
        super().__init__()

        self.conv = ModulatedConv2d(
            in_channel,
            out_channel,
            kernel_size,
            style_dim,
            upsample=upsample,
            blur_kernel=blur_kernel,
            demodulate=demodulate,
        )

        self.noise = NoiseInjection()
        # self.bias = nn.Parameter(torch.zeros(1, out_channel, 1, 1))
        self.activate = FusedLeakyReLU(out_channel)

    def forward(self, input, style, noise=None):
        out = self.conv(input, style)
        out = self.noise(out, noise=noise)
        # out = out + self.bias
        out = self.activate(out)

        return out


class ToRGB(nn.Module):
    def __init__(self, in_channel, style_dim, upsample=True, blur_kernel=[1, 3, 3, 1]):
        super().__init__()

        if upsample:
            self.upsample = Upsample(blur_kernel)

        self.conv = ModulatedConv2d(in_channel, 3, 1, style_dim, demodulate=False)
        self.bias = nn.Parameter(torch.zeros(1, 3, 1, 1))

    def forward(self, input, style, skip=None):
        out = self.conv(input, style)
        out = out + self.bias

        if skip is not None:
            skip = self.upsample(skip)

            out = out + skip

        return out


class Generator(nn.Module):
    def __init__(
            self,
            size,
            style_dim,
            n_mlp,
            channel_multiplier=2,
            blur_kernel=[1, 3, 3, 1],
            lr_mlp=0.01,
    ):
        super().__init__()

        self.size = size

        self.style_dim = style_dim

        self.channels = {
            4: 512,
            8: 512,
            16: 512,
            32: 512,
            64: 256 * channel_multiplier,
            128: 128 * channel_multiplier,
            256: 64 * channel_multiplier,
            512: 32 * channel_multiplier,
            1024: 16 * channel_multiplier,
        }

        self.input = ConstantInput(self.channels[4])
        self.conv1 = StyledConv(
            self.channels[4], self.channels[4], 3, style_dim, blur_kernel=blur_kernel
        )
        self.to_rgb1 = ToRGB(self.channels[4], style_dim, upsample=False)

        self.log_size = int(math.log(size, 2))
        self.num_layers = (self.log_size - 2) * 2 + 1

        self.convs = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        self.to_rgbs = nn.ModuleList()
        self.noises = nn.Module()

        in_channel = self.channels[4]

        for layer_idx in range(self.num_layers):
            res = (layer_idx + 5) // 2
            shape = [1, 1, 2 ** res, 2 ** res]
            self.noises.register_buffer(f"noise_{layer_idx}", torch.randn(*shape))

        for i in range(3, self.log_size + 1):
            out_channel = self.channels[2 ** i]

            self.convs.append(
                StyledConv(
                    in_channel,
                    out_channel,
                    3,
                    style_dim,
                    upsample=True,
                    blur_kernel=blur_kernel,
                )
            )

            self.convs.append(
                StyledConv(
                    out_channel, out_channel, 3, style_dim, blur_kernel=blur_kernel
                )
            )

            self.to_rgbs.append(ToRGB(out_channel, style_dim))

            in_channel = out_channel

        self.n_latent = self.log_size * 2 - 2

    def forward_no_feature_stylespace(
        self,
        style_1,
        style_2,
        style_3,
        style_4,
        style_5,
        style_6,
        style_7,
        style_8,
        style_9,
        to_rgb_stylespace_1,
        to_rgb_stylespace_2,
        to_rgb_stylespace_3,
        to_rgb_stylespace_4,
        to_rgb_stylespace_5,
    ):
        """
        Forward pass when new_features is NOT provided (i.e. None) and is_stylespace=True.
        """

        noise = [getattr(self.noises, f"noise_{i}") for i in range(self.num_layers)]

        out = self.input(style_1)  # styles[0] is the style-space tensor for the first layer
        out = self.conv1(out, style_1.float(), noise=noise[0])
        skip = self.to_rgb1(out, to_rgb_stylespace_1.float())

        out = self.convs[0](out, style_2.float(), noise=noise[1])
        out = self.convs[1](out, style_3.float(), noise=noise[2])
        skip = self.to_rgbs[0](out, to_rgb_stylespace_2.float(), skip)

        out = self.convs[2](out, style_4.float(), noise=noise[3])
        out = self.convs[3](out, style_5.float(), noise=noise[4])
        skip = self.to_rgbs[1](out, to_rgb_stylespace_3.float(), skip)

        out = self.convs[4](out, style_6.float(), noise=noise[5])
        out = self.convs[5](out, style_7.float(), noise=noise[6])
        skip = self.to_rgbs[2](out, to_rgb_stylespace_4.float(), skip)

        out = self.convs[6](out, style_8.float(), noise=noise[7])
        out = self.convs[7](out, style_9.float(), noise=noise[8])
        skip = self.to_rgbs[3](out, to_rgb_stylespace_5.float(), skip)

        return skip, out

    def forward_with_feature_stylespace(
        self,
        style_1,
        style_2,
        style_3,
        style_4,
        style_5,
        style_6,
        style_7,
        style_8,
        style_9,
        style_10,
        style_11,
        style_12,
        style_13,
        style_14,
        style_15,
        style_16,
        style_17,
        to_rgb_stylespace_1,
        to_rgb_stylespace_2,
        to_rgb_stylespace_3,
        to_rgb_stylespace_4,
        to_rgb_stylespace_5,
        to_rgb_stylespace_6,
        to_rgb_stylespace_7,
        to_rgb_stylespace_8,
        to_rgb_stylespace_9,
        new_feature,
    ):
        """
        Forward pass when new_features is provided and is_stylespace=True.
        We only overwrite the feature at layer index 9.

        Note: `new_features` is assumed to be a list of length >= 10, where new_features[9] may be a tensor or None.
        """
        noise = [getattr(self.noises, f"noise_{i}") for i in range(self.num_layers)]

        out = self.input(style_1)
        out = self.conv1(out, style_1.float(), noise=noise[0])
        skip = self.to_rgb1(out, to_rgb_stylespace_1.float())

        out = self.convs[0](out, style_2.float(), noise=noise[1])
        out = self.convs[1](out, style_3.float(), noise=noise[2])
        skip = self.to_rgbs[0](out, to_rgb_stylespace_2.float(), skip)

        out = self.convs[2](out, style_4.float(), noise=noise[3])
        out = self.convs[3](out, style_5.float(), noise=noise[4])
        skip = self.to_rgbs[1](out, to_rgb_stylespace_3.float(), skip)

        out = self.convs[4](out, style_6.float(), noise=noise[5])
        out = self.convs[5](out, style_7.float(), noise=noise[6])
        skip = self.to_rgbs[2](out, to_rgb_stylespace_4.float(), skip)

        out = self.convs[6](out, style_8.float(), noise=noise[7])
        out = self.convs[7](out, style_9.float(), noise=noise[8])
        skip = self.to_rgbs[3](out, to_rgb_stylespace_5.float(), skip)

        out = self.convs[8](new_feature.type_as(out), style_10.float(), noise=noise[9])
        out = self.convs[9](out, style_11.float(), noise=noise[10])
        skip = self.to_rgbs[4](out, to_rgb_stylespace_6.float(), skip)

        out = self.convs[10](out, style_12.float(), noise=noise[11])
        out = self.convs[11](out, style_13.float(), noise=noise[12])
        skip = self.to_rgbs[5](out, to_rgb_stylespace_7.float(), skip)

        out = self.convs[12](out, style_14.float(), noise=noise[13])
        out = self.convs[13](out, style_15.float(), noise=noise[14])
        skip = self.to_rgbs[6](out, to_rgb_stylespace_8.float(), skip)

        out = self.convs[14](out, style_16.float(), noise=noise[15])
        out = self.convs[15](out, style_17.float(), noise=noise[16])
        skip = self.to_rgbs[7](out, to_rgb_stylespace_9.float(), skip)

        return skip

    def forward(
        self,
        style_1,
        style_2,
        style_3,
        style_4,
        style_5,
        style_6,
        style_7,
        style_8,
        style_9,
        style_10,
        style_11,
        style_12,
        style_13,
        style_14,
        style_15,
        style_16,
        style_17,
        to_rgb_stylespace_1,
        to_rgb_stylespace_2,
        to_rgb_stylespace_3,
        to_rgb_stylespace_4,
        to_rgb_stylespace_5,
        to_rgb_stylespace_6,
        to_rgb_stylespace_7,
        to_rgb_stylespace_8,
        to_rgb_stylespace_9,
        new_feature=None
    ):
        """
        Main forward method for the is_stylespace=True case.
        """

        if new_feature is None:
            return self.forward_no_feature_stylespace(
                style_1,
                style_2,
                style_3,
                style_4,
                style_5,
                style_6,
                style_7,
                style_8,
                style_9,
                to_rgb_stylespace_1,
                to_rgb_stylespace_2,
                to_rgb_stylespace_3,
                to_rgb_stylespace_4,
                to_rgb_stylespace_5,
            )
        else:
            # Forward pass with feature insertion at layer index 9
            return self.forward_with_feature_stylespace(
                style_1,
                style_2,
                style_3,
                style_4,
                style_5,
                style_6,
                style_7,
                style_8,
                style_9,
                style_10,
                style_11,
                style_12,
                style_13,
                style_14,
                style_15,
                style_16,
                style_17,
                to_rgb_stylespace_1,
                to_rgb_stylespace_2,
                to_rgb_stylespace_3,
                to_rgb_stylespace_4,
                to_rgb_stylespace_5,
                to_rgb_stylespace_6,
                to_rgb_stylespace_7,
                to_rgb_stylespace_8,
                to_rgb_stylespace_9,
                new_feature
            )
