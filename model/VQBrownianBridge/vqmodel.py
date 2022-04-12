import pdb

import torch
import torch.nn as nn

from model.VQBrownianBridge.base import ResnetBlock, AttnBlock, Downsample, Upsample, nonlinear, normalize
from model.VQBrownianBridge.quantize import VectorQuantizer


class Encoder(nn.Module):

    def __init__(self, *, in_channels, map_channels, out_channels, ch_mult=(1, 2, 4, 8),
                 attn_resolutions, resolution, z_channels,
                 num_res_blocks, dropout, resamp_with_conv=True):
        """
        :param in_channels: 输入数据的维度
        :param map_channels: 将输入数据的维度进行映射的维度
        :param out_channels: 输出数据的维度
        :param ch_mult: 进行维度翻倍的层
        :param attn_resolutions: 需要用attention操作的分辨率
        :param resolution: 输入数据的分辨率
        :param z_channels: 量化的codebook的维度
        :param num_res_blocks: 每个子层的ResnetBlock的个数
        :param dropout: dropout
        :param resamp_with_conv: 是否用卷积来进行重采样
        """
        super().__init__()
        self.in_channels = in_channels
        self.map_channels = map_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.num_resolutions = len(ch_mult)
        self.resolution = resolution

        # 将in_channels调整到map_channels
        self.conv_in = torch.nn.Conv2d(in_channels, map_channels, kernel_size=3, stride=1, padding=1)

        curr_res = resolution
        in_ch_mult = (1,) + tuple(ch_mult)  # 相对ch_mult后延一位
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()

            block_in = map_channels * in_ch_mult[i_level]
            block_out = map_channels * ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out, dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample(block_in, resamp_with_conv)
                curr_res = curr_res // 2
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in, dropout=dropout)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in, dropout=dropout)

        self.norm_out = normalize(block_in)
        self.conv_out = nn.Conv2d(block_in, z_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        # pdb.set_trace()
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1])
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # middle
        h = hs[-1]
        # pdb.set_trace()
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)

        # end
        h = self.norm_out(h)
        h = nonlinear(h)
        h = self.conv_out(h)
        return h


class Decoder(nn.Module):
    def __init__(self, *, in_channels, map_channels, out_channels, ch_mult=(1, 2, 4, 8),
                 attn_resolutions, resolution, z_channels,
                 num_res_blocks, dropout, resamp_with_conv=True):
        super().__init__()
        self.in_channels = in_channels
        self.map_channels = map_channels
        self.out_channels = out_channels
        self.num_resolutions = len(ch_mult)
        self.resolution = resolution
        self.num_res_blocks = num_res_blocks

        in_ch_mult = (1,) + tuple(ch_mult)
        block_in = map_channels * ch_mult[self.num_resolutions - 1]
        curr_res = resolution // 2 ** (self.num_resolutions - 1)
        self.conv_in = torch.nn.Conv2d(z_channels, block_in, kernel_size=3, stride=1, padding=1)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in, dropout=dropout)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in, dropout=dropout)

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = map_channels * ch_mult[i_level]
            for i_block in range(self.num_res_blocks + 1):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out, dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up)

        # end
        self.norm_out = normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, z):
        h = self.conv_in(z)

        # middle
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end
        h = self.norm_out(h)
        h = nonlinear(h)
        h = self.conv_out(h)
        return h


class VQModel(nn.Module):
    def __init__(self, ed_config, n_embed, embed_dim, ckpt_path=None, ignore_keys=[]):
        """
        :param ed_config: encoder decoder config
        :param n_embed: codebook大小
        :param embed_dim: 嵌入维度
        :param ckpt_path: check point path, 用于从保存的断点恢复模型
        :param ignore_keys: 恢复断点时需要忽略的部分
        """
        super(VQModel, self).__init__()
        self.n_embed = n_embed
        self.encoder = Encoder(**ed_config)
        self.decoder = Decoder(**ed_config)
        self.quantize = VectorQuantizer(n_embed, embed_dim, beta=0.25)
        self.quant_conv = nn.Conv2d(ed_config['z_channels'], embed_dim, 1)
        self.post_quant_conv = nn.Conv2d(embed_dim, ed_config['z_channels'], 1)
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["vqmodel"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

    def encode(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        quant, emb_loss, info = self.quantize(h)
        return quant, emb_loss, info

    def decode(self, quant):
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)
        return dec

    def forward(self, input):
        quant, diff, _ = self.encode(input)
        dec = self.decode(quant)
        return dec, diff