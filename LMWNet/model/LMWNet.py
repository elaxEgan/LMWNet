
from torch import Tensor
import torchvision.models as models
import math
from functools import partial
from typing import Optional, Callable

from pytorch_wavelets import DWTForward, DWTInverse

from monai.networks.blocks.convolutions import Convolution
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from timm.models.layers import DropPath, trunc_normal_
from torchvision.ops.deform_conv import deform_conv2d as deform_conv2d_tv
from torch.nn import init
from torch.nn.modules.utils import _pair

DropPath.__repr__ = lambda self: f"timm.DropPath({self.drop_prob})"

from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref


mob_conv1_2 = mob_conv2_2 = mob_conv3_3 = mob_conv4_3 = mob_conv5_3 = None


def conv_1_2_hook(module, input, output):
    global mob_conv1_2
    mob_conv1_2 = output
    return None


def conv_2_2_hook(module, input, output):
    global mob_conv2_2
    mob_conv2_2 = output
    return None


def conv_3_3_hook(module, input, output):
    global mob_conv3_3
    mob_conv3_3 = output
    return None


def conv_4_3_hook(module, input, output):
    global mob_conv4_3
    mob_conv4_3 = output
    return None


def conv_5_3_hook(module, input, output):
    global mob_conv5_3
    mob_conv5_3 = output
    return None




class MobileNet(nn.Module):
    def __init__(self):
        super(MobileNet,self).__init__()
        #self.mbv = shufflenet_v2_x2_0(pretrained=True).features
        self.mbv = models.mobilenet_v3_large(pretrained=True).features
        self.mbv[1].register_forward_hook(conv_1_2_hook)
        self.mbv[3].register_forward_hook(conv_2_2_hook)
        self.mbv[6].register_forward_hook(conv_3_3_hook)
        self.mbv[12].register_forward_hook(conv_4_3_hook)
        self.mbv[15].register_forward_hook(conv_5_3_hook)

    def forward(self, x: Tensor) -> Tensor:
        global mob_conv1_2, mob_conv2_2, mob_conv3_3, mob_conv4_3, mob_conv5_3
        self.mbv(x)

        return mob_conv1_2, mob_conv2_2, mob_conv3_3, mob_conv4_3, mob_conv5_3


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class TransBasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=2, stride=2, padding=0, dilation=1, bias=False):
        super(TransBasicConv2d, self).__init__()
        self.Deconv = nn.ConvTranspose2d(in_planes, out_planes,
                                         kernel_size=kernel_size, stride=stride,
                                         padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.Deconv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Reduction(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Reduction, self).__init__()
        self.reduce = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
        )

    def forward(self, x):
        return self.reduce(x)


def get_dwconv_layer(
        spatial_dims: int, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1,
        bias: bool = False
):
    depth_conv = Convolution(spatial_dims=spatial_dims, in_channels=in_channels, out_channels=in_channels,
                             strides=stride, kernel_size=kernel_size, bias=bias, conv_only=True, groups=in_channels)
    point_conv = Convolution(spatial_dims=spatial_dims, in_channels=in_channels, out_channels=out_channels,
                             strides=stride, kernel_size=1, bias=bias, conv_only=True, groups=1)
    return torch.nn.Sequential(depth_conv, point_conv)


class ChannelAttentionModule(nn.Module):
    def __init__(self, in_channels, reduction=2):
        super(ChannelAttentionModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))  # 执行平均池化:(B,d,H,W)-avg_pool->(B,d,1,1)-fc->(B,d,1,1)
        max_out = self.fc(self.max_pool(x))  # 执行最大池化:(B,d,H,W)-max_pool->(B,d,1,1)-fc->(B,d,1,1)
        out = avg_out + max_out  # (B,d,1,1) + (B,d,1,1) == (B,d,1,1)
        return self.sigmoid(out)  # 通过sigmoid生成权重表示:(B,d,1,1)


class SpatialAttentionModule(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttentionModule, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)  # 计算通道方向平均值:(B,d,H,W)-avg->(B,1,H,W)
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # 计算通道方向最大值:(B,d,H,W)-max->(B,1,H,W)
        x = torch.cat([avg_out, max_out], dim=1)  # 通道方向拼接: (B,1,H,W)-cat-(B,1,H,W)-->(B,2,H,W);
        x = self.conv1(x)  # 降维: (B,2,H,W)-->(B,1,H,W)
        return self.sigmoid(x)  # 通过sigmoid生成权重表示:(B,1,H,W)


class SS2D(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=16,
            # d_state="auto", # 20240109
            d_conv=3,
            expand=0.5,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            dropout=0.,
            conv_bias=True,
            bias=False,
            device=None,
            dtype=None,
            mode=0,
            dwt=None,
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        # self.d_state = math.ceil(self.d_model / 6) if d_state == "auto" else d_model # 20240109
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.channel_attention = ChannelAttentionModule(self.d_inner)
        self.spatial_attention = SpatialAttentionModule()
        self.DeMixer = DefMixer(self.d_inner, self.d_inner)
        self.wt = DWTForward(J=1, mode='zero', wave='haar')
        self.wti = DWTInverse(mode='zero', wave='haar')
        self.conv_bn_relu = nn.Sequential(
            nn.Conv2d(self.d_inner * 3, self.d_inner, kernel_size=1, stride=1),
            nn.BatchNorm2d(self.d_inner),
            nn.ReLU(inplace=True),
        )
        self.outconv_bn_relu_L = nn.Sequential(
            nn.Conv2d(self.d_inner, self.d_inner, kernel_size=1, stride=1),
            nn.BatchNorm2d(self.d_inner),
            nn.ReLU(inplace=True),
        )
        self.outconv_bn_relu_H = nn.Sequential(
            nn.Conv2d(self.d_inner, self.d_inner, kernel_size=1, stride=1),
            nn.BatchNorm2d(self.d_inner),
            nn.ReLU(inplace=True),
        )
        self.outconv_bn_relu_H2 = nn.Sequential(
            nn.Conv2d(3 * self.d_inner, 3 * self.d_inner, kernel_size=1, stride=1),
            nn.BatchNorm2d(3 * self.d_inner),
            nn.ReLU(inplace=True),
        )
        self.gamma1 = nn.Parameter(torch.ones(1))
        self.gamma2 = nn.Parameter(torch.ones(1))
        self.gamma3 = nn.Parameter(torch.ones(1))
        self.act = nn.SiLU()

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K=4, N, inner)
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))  # (K=4, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))  # (K=4, inner)
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True)  # (K=4, D, N)
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True)  # (K=4, D, N)

        self.forward_core = self.forward_corev0
        # self.forward_core = self.forward_core_windows
        # self.forward_core = self.forward_corev0_seq
        # self.forward_core = self.forward_corev1

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None
        self.mode = mode
        self.dwt = dwt

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward_corev0(self, x: torch.Tensor):
        self.selective_scan = selective_scan_fn
        B, C, H, W = x.shape
        L = H * W

        K = 4
        ### Scanning Expansion ###
        # First sequence: row-wise order; Second sequence: column-wise order.
        # For the first sequence x1: (B,d,H,W)-view->(B,d,L);
        # For the second sequence x2: (B,d,H,W)-transpose->(B,d,W,H)-view->(B,d,L);
        # Finally, stack x1 and x2 along the first dimension: (B,2d,L)-view->(B,2,d,L);
        # Equivalent to constructing patch sequences in two different ordering directions.
        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)],
                             dim=1).view(B, 2, -1, L)
        # After flipping x_hwwh, the first sequence: row-wise reverse order; second sequence: column-wise reverse order.
        # Concatenation yields four directional sequence representations.
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])],
                       dim=1)  # (B,2,d,L)-cat-(B,2,d,L) == (B,K,d,L); K=4; flip is used to reverse along a given dimension.

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L),
                             self.x_proj_weight)  # (B,K,d,L)-einsum-(K,C,d) == (B,K,C,L)
        # x_dbl = x_dbl + self.x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state],
                                  dim=2)  # Split x_dbl into dts, B, C. (B,K,C,L)-> dts:(B,K,dt_rank,L); Bs:(B,K,d_state,L); Cs:(B,K,d_state,L)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L),
                           self.dt_projs_weight)  # Transform dts: (B,K,dt_rank,L)-einsum-(K,d,dt_rank) == (B,K,d,L)

        xs = xs.float().view(B, -1, L)  # (B,K,d,L)-view->(B,Kd,L)
        dts = dts.contiguous().float().view(B, -1, L)  # (B,K,d,L)-view->(B,Kd,L)
        Bs = Bs.float().view(B, K, -1, L)  # (B,K,d_state,L)-view->(B,K,d_state,L)
        Cs = Cs.float().view(B, K, -1, L)  # (B,K,d_state,L)-view->(B,K,d_state,L)

        Ds = self.Ds.float().view(-1)  # (k * d)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)  # (k*d, d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)  # (k * d)

        ### S6 Block ###
        # xs contains patch sequences in four directions
        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)  # out_y:(B,K,d,L)
        assert out_y.dtype == torch.float

        ### Sequence Merging ###
        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1,
                                                          L)  # out_y[:, 2:4]: flip the last two sequences (row-reverse, col-reverse) back to normal order; (B,2,d,L), [row-forward, col-forward]
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1,
                                                                                                L)  # Convert the column-forward sequence back to default row-first ordering; out_y[:, 1]: (B,d,L); (B,d,L)-view->(B,d,W,H)-transpose->(B,d,H,W)-view->(B,d,L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1,
                                                                                                   L)  # Similarly, convert the second sequence in inv_y (col-forward) back to default row-first ordering
        y = out_y[:, 0] + inv_y[:,
                          0] + wh_y + invwh_y  # out_y[:,0] is already row-forward; inv_y[:,0] is corrected row-reverse; wh_y restores col-forward to row-forward; invwh_y does the same.
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W,
                                                                 -1)  # (B,d,L)-transpose->(B,L,d)-view->(B,H,W,d)
        y = self.out_norm(y).to(x.dtype)  # (B,H,W,d)

        return y

    def forward(self, x: torch.Tensor, layer=1, **kwargs):
        if self.mode == 0:
            B, H, W, C = x.shape

            xz = self.in_proj(x)  # (B,H,W,C)-->(B,H,W,D)
            x, z = xz.chunk(2, dim=-1)  # Split: (B,H,W,D)-> x:(B,H,W,d), z:(B,H,W,d); D=2d
            z = z.permute(0, 3, 1, 2)  # (B,H,W,d)-->(B,d,H,W)
            z = self.channel_attention(
                z) * z  # Apply channel attention: (B,d,H,W)-channel_attention->(B,d,1,1); elementwise multiply -> (B,d,H,W)
            z = z.permute(0, 2, 3, 1).contiguous()
            x = x.permute(0, 3, 1, 2).contiguous()  # (B,H,W,d)-->(B,d,H,W)
            x = self.act(self.conv2d(x))  # (B,d,H,W)-conv->(B,d,H,W)

            y = self.forward_core(x)  # (B,d,H,W)->(B,H,W,d)
            y = y * F.silu(z)

            out = self.out_proj(y)
            if self.dropout is not None:
                out = self.dropout(out)
        if self.mode == 1:
            B, H, W, C = x.shape

            xz = self.in_proj(x)  # (B,H,W,C)-->(B,H,W,D)
            x, z = xz.chunk(2, dim=-1)  # Split: (B,H,W,D)-> x:(B,H,W,d), z:(B,H,W,d); D=2d

            z = z.permute(0, 3, 1, 2)  # (B,H,W,d)-->(B,d,H,W)
            zL, zH = self.wt(z)

            z_HL = zH[0][:, :, 0, ::]
            z_LH = zH[0][:, :, 1, ::]
            z_HH = zH[0][:, :, 2, ::]

            zH = torch.cat([z_HL, z_LH, z_HH], dim=1)

            if self.dwt == 'h':
                zH = self.conv_bn_relu(zH)

                # zL = self.outconv_bn_relu_L(zL)
                # zH = self.outconv_bn_relu_H(zH)
                zH = F.interpolate(zH, size=(H, W), mode='bilinear', align_corners=True)

                z = self.DeMixer(z)  # Spatial adaptive deformable operation
                z = self.channel_attention(
                    z) * z  # Channel attention: (B,d,H,W)-channel_attention->(B,d,1,1); multiply -> (B,d,H,W)
                # z = self.spatial_attention(z) * z
                z = z + self.gamma1 * zH
            if self.dwt == 'l':
                zH = self.conv_bn_relu(zH)

                # zL = self.outconv_bn_relu_L(zL)
                # zH = self.outconv_bn_relu_H(zH)
                zL = F.interpolate(zL, size=(H, W), mode='bilinear', align_corners=True)

                z = self.DeMixer(z)  # Spatial adaptive deformable operation
                z = self.channel_attention(
                    z) * z  # Channel attention: (B,d,H,W)-channel_attention->(B,d,1,1); multiply -> (B,d,H,W)
                # z = self.spatial_attention(z) * z
                z = z + self.gamma2 * zL
            if self.dwt == 'lh':
                zL = self.outconv_bn_relu_L(zL)
                zH = self.outconv_bn_relu_H2(zH)

                # zLH = self.outconv_bn_relu(torch.cat([zL, zH], dim=1))
                z_HL, z_LH, z_HH = torch.chunk(zH, 3, dim=1)
                zH_reconstructed = [torch.stack([z_HL, z_LH, z_HH], dim=2)]
                zLH = self.wti((zL, zH_reconstructed))
                z = self.DeMixer(z)  # Spatial adaptive deformable operation
                z = self.channel_attention(
                    z) * z  # Channel attention: (B,d,H,W)-channel_attention->(B,d,1,1); multiply -> (B,d,H,W)
                # z = self.spatial_attention(z) * z
                z = z + self.gamma2 * zLH

            z = z.permute(0, 2, 3,
                          1).contiguous()  # Save current z for later multiplication with SSM output: (B,d,H,W)->(B,H,W,d)

            x = x.permute(0, 3, 1, 2).contiguous()  # (B,H,W,d)->(B,d,H,W)
            x = self.act(self.conv2d(x))  # (B,d,H,W)-conv->(B,d,H,W)

            y = self.forward_core(x)  # (B,d,H,W)->(B,H,W,d)

            y = y * F.silu(z)

            out = self.out_proj(y)
            if self.dropout is not None:
                out = self.dropout(out)

        return out


def channel_shuffle(x: Tensor, groups: int) -> Tensor:
    batch_size, num_channels, height, width = x.size()
    channels_per_group = num_channels // groups

    # 重塑
    # [batch_size, num_channels, height, width] -> [batch_size, groups, channels_per_group, height, width]
    x = x.view(batch_size, groups, channels_per_group, height, width)

    # 交换维度，实现通道的打乱
    x = torch.transpose(x, 1, 2).contiguous()

    # 恢复形状
    # [batch_size, channels_per_group, groups, height, width] -> [batch_size, num_channels, height, width]
    x = x.view(batch_size, -1, height, width)

    return x


class SplitBlock(nn.Module):
    def __init__(self, ratio=(3 / 8)):
        super(SplitBlock, self).__init__()
        self.ratio = ratio

    def forward(self, x):
        ## AugshuffleNet的分割方式，为了中间层实现channel crossover,输入通道提前被分割成3份，避免重复切分操作。

        c = x.size(1)
        c1 = int(c * self.ratio)
        c2 = (c - c1) // 2

        out = torch.split(x, [c2, c2, c1], dim=1)

        return out


class Block(nn.Module):
    def __init__(
            self,
            hidden_dim: int = 0,
            drop_path: float = 0,
            norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
            attn_drop_rate: float = 0,
            d_state: int = 16,
            layer: int = 1,
            split_ratio: float = 0.5,
            **kwargs,
    ):
        super().__init__()
        assert split_ratio <= 0.5
        factor = 2.0
        d_model = int(hidden_dim // factor)
        self.down = nn.Linear(hidden_dim // 2, d_model // 2)
        self.up = nn.Linear(d_model // 2, hidden_dim // 2)
        self.ln_1 = norm_layer(d_model // 2)
        self.self_attention = SS2D(d_model=d_model // 2, dropout=attn_drop_rate, d_state=d_state, **kwargs)
        self.drop_path = DropPath(drop_path)
        self.layer = layer
        self.split = SplitBlock(split_ratio)
        cin = int(split_ratio * hidden_dim)
        cout = int(0.5 * hidden_dim)

        self.conv2 = nn.Conv2d(cin, cin,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               groups=cin,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(cin)

        self.conv3 = nn.Conv2d(cout, cout, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(cout)

    def forward(self, input: torch.Tensor):
        x1, x2, x3 = self.split(input)

        left = self.bn2(self.conv2(x3))

        c1, c2 = torch.chunk(left, 2, 1)
        left = torch.cat([c2, x2], 1)

        left = self.bn3(self.conv3(left))
        left = F.relu(left, inplace=True)

        right = torch.cat([c1, x1], dim=1)
        right = self.down(right.permute(0, 2, 3, 1))  # (B,H,W,C)-->(B,H,W,d_model)
        right = right + self.drop_path(self.self_attention(self.ln_1(right)))  #
        right = self.up(right)

        out = torch.cat([left, right.permute(0, 3, 1, 2)], dim=1)
        out = channel_shuffle(out, 2) + input

        return out


class Encoder(nn.Module):
    def __init__(self, channel=32):
        super(Encoder, self).__init__()

        # self.get_dwconv_layer = get_dwconv_layer(spatial_dims=2, in_channels=3, out_channels=8)
        # self.conv1 = BasicConv2d(3, 24, 3, padding=1)
        self.attention1 = Block(24, 0.5, attn_drop_rate=0.5, in_channel=channel // 2, out_channel=channel // 2,
                                stride=1, expand_ratio=3)
        self.attention2 = Block(24, 0.5, attn_drop_rate=0.5, in_channel=channel // 2, out_channel=channel // 2,
                                stride=1, expand_ratio=3)
        self.attention3 = Block(24, 0.5, attn_drop_rate=0.5, in_channel=channel // 2, out_channel=channel // 2,
                                stride=1, expand_ratio=3)
        self.attention4 = Block(40, 0.5, attn_drop_rate=0.5, in_channel=channel, out_channel=channel, stride=1,
                                expand_ratio=3)
        self.attention5 = Block(40, 0.5, attn_drop_rate=0.5, in_channel=channel, out_channel=channel, stride=1,
                                expand_ratio=3)
        self.attention6 = Block(40, 0.5, attn_drop_rate=0.5, in_channel=channel, out_channel=channel, stride=1,
                                expand_ratio=3)
        self.attention7 = Block(64, 0.5, attn_drop_rate=0.5, in_channel=channel * 2, out_channel=channel * 2, stride=1,
                                expand_ratio=3)
        self.attention8 = Block(64, 0.5, attn_drop_rate=0.5, in_channel=channel * 2, out_channel=channel * 2, stride=1,
                                expand_ratio=3)
        self.attention9 = Block(64, 0.5, attn_drop_rate=0.5, in_channel=channel * 2, out_channel=channel * 2, stride=1,
                                expand_ratio=3)
        self.attention10 = Block(104, 0.5, attn_drop_rate=0.5, in_channel=channel * 4, out_channel=channel * 4,
                                 stride=1, expand_ratio=3)
        self.attention11 = Block(104, 0.5, attn_drop_rate=0.5, in_channel=channel * 4, out_channel=channel * 4,
                                 stride=1, expand_ratio=3)
        self.attention12 = Block(104, 0.5, attn_drop_rate=0.5, in_channel=channel * 4, out_channel=channel * 4,
                                 stride=1, expand_ratio=3)
        self.attention13 = Block(216, 0.5, attn_drop_rate=0.5, in_channel=channel * 8, out_channel=channel * 8,
                                 stride=1, expand_ratio=3)
        self.attention14 = Block(216, 0.5, att8n_drop_rate=0.5, in_channel=channel * 8, out_channel=channel * 8,
                                 stride=1, expand_ratio=3)
        self.attention15 = Block(216, 0.5, attn_drop_rate=0.5, in_channel=channel * 8, out_channel=channel * 8,
                                 stride=1, expand_ratio=3)

        self.conv1x1_1 = BasicConv2d(3, 24, 1, 1)
        self.conv1x1_2 = BasicConv2d(40, 40, 1, 1)
        self.conv1x1_3 = BasicConv2d(64, 64, 1, 1)
        self.conv1x1_4 = BasicConv2d(104, 104, 1, 1)
        self.conv1x1_5 = BasicConv2d(216, 216, 1, 1)

        self.down_sample = nn.MaxPool2d(kernel_size=2, stride=2)

        self.ChannelAtt = ChannelAttentionModule(24)
        self.ChannelAtt1 = ChannelAttentionModule(40)
        self.ChannelAtt2 = ChannelAttentionModule(64)
        self.ChannelAtt3 = ChannelAttentionModule(104)
        self.ChannelAtt4 = ChannelAttentionModule(216)
        self.ChannelAtt5 = ChannelAttentionModule(376)

    def forward(self, input, s1, s2, s3, s4, s5):
        x = self.conv1x1_1(input)

        x1 = self.down_sample(self.attention1(x))
        x1 = self.attention3(self.attention2(x1))

        x1 = torch.cat([s1, x1], dim=1)
        x1 = channel_shuffle(x1, groups=2)
        x1 = self.ChannelAtt1(x1) * x1

        xc2 = self.conv1x1_2(x1)
        x2 = self.down_sample(self.attention4(xc2))
        x2 = self.attention6(self.attention5(x2))

        x2 = torch.cat([s2, x2], dim=1)
        x2 = channel_shuffle(x2, groups=2)
        x2 = self.ChannelAtt2(x2) * x2

        xc3 = self.conv1x1_3(x2)
        x3 = self.down_sample(self.attention7(xc3))
        x3 = self.attention9(self.attention8(x3))

        x3 = torch.cat([s3, x3], dim=1)
        x3 = channel_shuffle(x3, groups=2)
        x3 = self.ChannelAtt3(x3) * x3

        xc4 = self.conv1x1_4(x3)
        x4 = self.down_sample(self.attention10(xc4))

        x4 = torch.cat([s4, x4], dim=1)
        x4 = channel_shuffle(x4, groups=2)
        x4 = self.ChannelAtt4(x4) * x4

        xc5 = self.conv1x1_5(x4)
        x5 = self.down_sample(self.attention13(xc5))
        x5 = self.attention15(self.attention14(x5))

        x5 = torch.cat([s5, x5], dim=1)
        x5 = channel_shuffle(x5, groups=2)
        x5 = self.ChannelAtt5(x5) * x5

        return x1, x2, x3, x4, x5

class VSSBlock(nn.Module):
    def __init__(
            self,
            hidden_dim: int = 0,
            drop_path: float = 0,
            norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
            attn_drop_rate: float = 0,
            d_state: int = 16,
            layer: int = 1,
            dwt: str = None,
            **kwargs,
    ):
        super().__init__()
        factor = 2.0
        d_model = int(hidden_dim // factor)
        self.down = nn.Linear(hidden_dim, d_model)
        self.up = nn.Linear(d_model, hidden_dim)
        self.ln_1 = norm_layer(d_model)
        self.self_attention = SS2D(d_model=d_model, dropout=attn_drop_rate, d_state=d_state, mode=1, dwt=dwt, **kwargs)
        self.drop_path = DropPath(drop_path)
        self.layer = layer
        self.dpconv = DepthwiseSeparableConv(hidden_dim, hidden_dim, 3, padding=1)

    def forward(self, input: torch.Tensor, input2=None):

        if input2 is not None:
            B, C, H, W = input.size()
            input2 = F.interpolate(input2, size=(H, W), mode='bilinear', align_corners=True)
            input2 = self.dpconv(input2)

            input = input + input2

        input = input.permute(0, 2, 3, 1)
        input_x = self.down(input)  # (B,H,W,C)-->(B,H,W,d_model)
        input_x = input_x + self.drop_path(self.self_attention(self.ln_1(input_x)))  #
        x = self.up(input_x) + input

        return x



class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x):
        x1=self.fn(x)
        return x1+x


class DefMixer(nn.Module):
    def __init__(self,dim_in, dim, depth=1, kernel_size=1):
        super(DefMixer, self).__init__()

        self.blocks = nn.Sequential(
            *[nn.Sequential(
                    nn.Conv2d(dim, dim, kernel_size=kernel_size),
                    nn.GELU(),
                    nn.BatchNorm2d(dim),
                    Residual(nn.Sequential(
                        ChlSpl(dim, dim, (1, 3), 1, 0),
                        nn.GELU(),
                        nn.BatchNorm2d(dim)
                    )),
            ) for i in range(depth)],
        )

    def forward(self, x):
        x = self.blocks(x)
        return x


class ChlSpl(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
    ):
        super(ChlSpl, self).__init__()

        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        if stride != 1:
            raise ValueError('stride must be 1')
        if padding != 0:
            raise ValueError('padding must be 0')

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = groups

        self.weight = nn.Parameter(torch.empty(out_channels, in_channels // groups, 1, 1))

        self.get_offset = Offset(dim=in_channels, kernel_size=3)

        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)
        self.register_buffer('offset', self.gen_offset())

        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)
    def gen_offset(self):
        offset = torch.empty(1, self.in_channels*2, 1, 1)
        start_idx = (self.kernel_size[0] * self.kernel_size[1]) // 2
        assert self.kernel_size[0] == 1 or self.kernel_size[1] == 1, self.kernel_size
        for i in range(self.in_channels):
            if self.kernel_size[0] == 1:
                offset[0, 2 * i + 0, 0, 0] = 0
                offset[0, 2 * i + 1, 0, 0] = (i + start_idx) % self.kernel_size[1] - (self.kernel_size[1] // 2)
            else:
                offset[0, 2 * i + 0, 0, 0] = (i + start_idx) % self.kernel_size[0] - (self.kernel_size[0] // 2)
                offset[0, 2 * i + 1, 0, 0] = 0
        return offset

    def forward(self, input: Tensor) -> Tensor:
        """
            input: Tensor[b,c,h,w]
        """
        offset_2 = self.get_offset(input)
        B, C, H, W = input.size()

        return deform_conv2d_tv(input, offset_2, self.weight, self.bias, stride=self.stride, padding=self.padding, dilation=self.dilation)

    def extra_repr(self) -> str:
        s = self.__class__.__name__ + '('
        s += '{in_channels}'
        s += ', {out_channels}'
        s += ', kernel_size={kernel_size}'
        s += ', stride={stride}'
        s += ', padding={padding}' if self.padding != (0, 0) else ''
        s += ', dilation={dilation}' if self.dilation != (1, 1) else ''
        s += ', groups={groups}' if self.groups != 1 else ''
        s += ', bias=False' if self.bias is None else ''
        s += ')'
        return s.format(**self.__dict__)


class Offset(nn.Module):
    def __init__(self, dim, kernel_size):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = 1
        self.p_conv = nn.Conv2d(dim, 2*kernel_size*kernel_size, kernel_size=3, padding=1, stride=1)
        nn.init.constant_(self.p_conv.weight, 0)
        self.p_conv.register_backward_hook(self._set_lr)
        self.opt = nn.Conv2d(2*self.kernel_size*self.kernel_size, dim*2, kernel_size=3, padding=1, stride=1, groups=2)


    @staticmethod
    def _set_lr(module, grad_input, grad_output):
        grad_input = (grad_input[i] * 0.1 for i in range(len(grad_input)))
        grad_output = (grad_output[i] * 0.1 for i in range(len(grad_output)))

    def _get_p(self, offset, dtype):
        N, h, w = offset.size(1)//2, offset.size(2), offset.size(3)

        p_n = self._get_p_n(N, dtype)
        p_0 = self._get_p_0(h, w, N, dtype)
        p = p_0 + p_n + offset
        return p

    def _get_p_n(self, N, dtype):
        p_n_x, p_n_y = torch.meshgrid(
            torch.arange(-(self.kernel_size-1)//2, (self.kernel_size-1)//2+1),
            torch.arange(-(self.kernel_size-1)//2, (self.kernel_size-1)//2+1))
        p_n = torch.cat([torch.flatten(p_n_x), torch.flatten(p_n_y)], 0)
        p_n = p_n.view(1, 2*N, 1, 1).type(dtype)
        return p_n

    def _get_p_0(self, h, w, N, dtype):
        p_0_x, p_0_y = torch.meshgrid(
            torch.arange(1, h*self.stride+1, self.stride),
            torch.arange(1, w*self.stride+1, self.stride))
        p_0_x = torch.flatten(p_0_x).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0_y = torch.flatten(p_0_y).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0 = torch.cat([p_0_x, p_0_y], 1).type(dtype)
        return p_0

    def forward(self, x):
        offset = self.p_conv(x)
        dtype = offset.data.type()
        N = offset.size(1) // 2
        p = self._get_p(offset, dtype)  #1,18,107,140
        p =self.opt(p)
        return p


class AdaptGate(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.conv1 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv2 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv3 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv = BasicConv2d(2*channel, channel, 3, padding=1)
        self.ca = ChannelAttentionModule(channel)

    def forward(self, left, right):
        left = self.conv1(left)
        right = self.conv2(right)
        fuse = self.conv(torch.cat([left, right], dim=1))
        att = torch.sigmoid(fuse)
        out = left * att + (1- att) * right
        out = self.ca(out) * out
        out = self.conv3(out)
        return out


class Decoder(nn.Module):
    def __init__(self, channel, mode=0):
        super(Decoder, self).__init__()
        self.mode = mode
        if self.mode == 1:
            self.conv1 = DepthwiseSeparableConv(channel, channel, 3, padding=1)
            self.conv2 = DepthwiseSeparableConv(channel, channel, 3, padding=1)
            self.conv3 = DepthwiseSeparableConv(channel, channel, 3, padding=1)
            self.conv4 = DepthwiseSeparableConv(channel, channel, 3, padding=1)
            self.conv1_s = DepthwiseSeparableConv(channel, channel, 3, padding=1)
            self.conv2_s = DepthwiseSeparableConv(channel, channel, 3, padding=1)
            self.conv3_s = DepthwiseSeparableConv(channel, channel, 3, padding=1)
            self.conv4_s = DepthwiseSeparableConv(channel, channel, 3, padding=1)
            self.S_conv = nn.Sequential(
                DepthwiseSeparableConv(2 * channel, channel, 3, padding=1),
                DepthwiseSeparableConv(channel, channel, 1)
            )
            self.S_conv2 = nn.Sequential(
                DepthwiseSeparableConv(2 * channel, channel, 3, padding=1),
                DepthwiseSeparableConv(channel, channel, 1)
            )
            self.AdaptGate = AdaptGate(channel)

        if self.mode == 2:
            self.conv1 = DepthwiseSeparableConv(channel, channel, 3, padding=1)
            self.conv2 = DepthwiseSeparableConv(channel, channel, 3, padding=1)
            self.conv1_s = DepthwiseSeparableConv(channel, channel, 3, padding=1)
            self.conv2_s = DepthwiseSeparableConv(channel, channel, 3, padding=1)
            self.S_conv = nn.Sequential(
                DepthwiseSeparableConv(2 * channel, channel, 3, padding=1),
                DepthwiseSeparableConv(channel, channel, 1)
            )

        if self.mode == 3:
            self.conv1 = DepthwiseSeparableConv(channel, channel, 3, padding=1)
            self.conv2 = DepthwiseSeparableConv(channel, channel, 3, padding=1)
            self.conv1_s = DepthwiseSeparableConv(channel, channel, 3, padding=1)
            self.conv2_s = DepthwiseSeparableConv(channel, channel, 3, padding=1)
            self.S_conv = nn.Sequential(
                DepthwiseSeparableConv(2 * channel, channel, 3, padding=1),
                DepthwiseSeparableConv(channel, channel, 1)
            )

    def forward(self, f1, fl, fh, f5):
        if self.mode == 1:
            fgl1 = F.interpolate(f5, size=fl.size()[2:], mode='bilinear', align_corners=True)
            fgl2 = F.interpolate(f1, size=fl.size()[2:], mode='bilinear', align_corners=True)
            fh = self.conv1(torch.sigmoid(self.conv1_s(fgl1)) * fh) + fh
            fl = self.conv2(torch.sigmoid(self.conv2_s(fgl1)) * fl) + fl
            fh2 = self.conv3(torch.sigmoid(self.conv3_s(fgl2)) * fh) + fh
            fl2 = self.conv4(torch.sigmoid(self.conv4_s(fgl2)) * fl) + fl
            out = self.S_conv(torch.cat((fh, fl), 1))
            out2 = self.S_conv2(torch.cat((fh2, fl2), 1))

            out = self.AdaptGate(out, out2) + out + out2
        if self.mode == 2:
            fgl1 = F.interpolate(f5, size=fl.size()[2:], mode='bilinear', align_corners=True)
            fh = self.conv1(torch.sigmoid(self.conv1_s(fgl1)) * fh) + fh
            fl = self.conv2(torch.sigmoid(self.conv2_s(fgl1)) * fl) + fl
            out = self.S_conv(torch.cat((fh, fl), 1))

        if self.mode == 3:
            fgl1 = F.interpolate(f1, size=fl.size()[2:], mode='bilinear', align_corners=True)
            fh = self.conv1(torch.sigmoid(self.conv1_s(fgl1)) * fh) + fh
            fl = self.conv2(torch.sigmoid(self.conv2_s(fgl1)) * fl) + fl
            out = self.S_conv(torch.cat((fh, fl), 1))

        return out


class LMWNet(nn.Module):
    def __init__(self, channel=32):
        super(LMWNet, self).__init__()

        self.Encoder1 = MobileNet()
        self.Encoder2 = Encoder(channel)

        self.Translayer1 = Reduction(40, channel)
        self.Translayer2 = Reduction(64, channel)
        self.Translayer3 = Reduction(104, channel)
        self.Translayer4 = Reduction(216, channel)
        self.Translayer5 = Reduction(376, channel)

        self.s_conv = nn.Conv2d(channel, 1, 3, padding=1)
        self.s_conv1 = nn.Conv2d(channel, 1, 3, padding=1)
        self.s_conv2 = nn.Conv2d(channel, 1, 3, padding=1)
        self.s_conv3 = nn.Conv2d(channel, 1, 3, padding=1)
        self.s_conv4 = nn.Conv2d(channel, 1, 3, padding=1)
        self.s_conv5 = nn.Conv2d(channel, 1, 3, padding=1)

        self.VSSBlock1 = VSSBlock(channel, 0.5, attn_drop_rate=0.5, dwt='h')
        self.VSSBlock2 = VSSBlock(channel, 0.5, attn_drop_rate=0.5, dwt='lh')
        self.VSSBlock3 = VSSBlock(channel, 0.5, attn_drop_rate=0.5, dwt='lh')
        self.VSSBlock4 = VSSBlock(channel, 0.5, attn_drop_rate=0.5, dwt='l')
        self.VSSBlock5 = VSSBlock(channel, 0.5, attn_drop_rate=0.5, dwt='l')

        self.Decoder1 = Decoder(channel, mode=3)
        self.Decoder2 = Decoder(channel, mode=1)
        self.Decoder3 = Decoder(channel, mode=1)
        self.Decoder4 = Decoder(channel, mode=2)

        self.trans_conv = TransBasicConv2d(channel, channel, kernel_size=2, stride=2,
                                           padding=0, dilation=1, bias=False)

        self.trans_conv1 = TransBasicConv2d(channel, channel, kernel_size=2, stride=2,
                                            padding=0, dilation=1, bias=False)
        self.trans_conv2 = TransBasicConv2d(channel, channel, kernel_size=2, stride=2,
                                            padding=0, dilation=1, bias=False)
        self.trans_conv3 = TransBasicConv2d(channel, channel, kernel_size=2, stride=2,
                                            padding=0, dilation=1, bias=False)
        self.trans_conv4 = TransBasicConv2d(channel, channel, kernel_size=2, stride=2,
                                            padding=0, dilation=1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        size = input.size()[2:]
        all_dict = {}

        x1, x2, x3, x4, x5 = self.Encoder1(input)
        x1, x2, x3, x4, x5 = self.Encoder2(input, x1, x2, x3, x4, x5)

        s1 = self.Translayer1(x1)
        s2 = self.Translayer2(x2)
        s3 = self.Translayer3(x3)
        s4 = self.Translayer4(x4)
        s5 = self.Translayer5(x5)

        x5_maf = s5
        x4_maf = self.VSSBlock4(s4, s5).permute(0, 3, 1, 2)
        x3_maf = self.VSSBlock3(s3, s4).permute(0, 3, 1, 2)
        x2_maf = self.VSSBlock2(s2, s3).permute(0, 3, 1, 2)
        x1_maf = self.VSSBlock1(s1, s2).permute(0, 3, 1, 2)

        x5_fraf = x5_maf
        x4_fraf = self.Decoder1(f1=x1_maf, fl=x4_maf, fh=self.trans_conv1(x5_maf), f5=None)
        x3_fraf = self.Decoder2(x1_maf, x3_maf, self.trans_conv2(x4_fraf), x5_maf)
        x2_fraf = self.Decoder3(x1_maf, x2_maf, self.trans_conv3(x3_fraf), x5_maf)
        x1_fraf = self.Decoder4(f1=None, fl=x1_maf, fh=self.trans_conv4(x2_fraf), f5=x5_maf)
        sal = self.trans_conv(x1_fraf)

        sal_out = self.s_conv(sal)
        x1_out = self.s_conv1(x1_fraf)
        x2_out = self.s_conv2(x2_fraf)
        x3_out = self.s_conv3(x3_fraf)
        x4_out = self.s_conv4(x4_fraf)
        x5_out = self.s_conv5(x5_fraf)

        x1_out = F.interpolate(x1_out, size=size, mode='bilinear', align_corners=True)
        x2_out = F.interpolate(x2_out, size=size, mode='bilinear', align_corners=True)
        x3_out = F.interpolate(x3_out, size=size, mode='bilinear', align_corners=True)
        x4_out = F.interpolate(x4_out, size=size, mode='bilinear', align_corners=True)
        x5_out = F.interpolate(x5_out, size=size, mode='bilinear', align_corners=True)



        return sal_out, self.sigmoid(sal_out), x1_out, self.sigmoid(x1_out), x2_out, self.sigmoid(
            x2_out), x3_out, self.sigmoid(
            x3_out), x4_out, self.sigmoid(x4_out), x5_out, self.sigmoid(x5_out)