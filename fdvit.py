# Copyright (c) 2024 Advanced Micro Devices, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License") and the MIT License (the "License2");

import torch
from einops import rearrange
from torch import nn
import torch.nn.functional as F
import math

from functools import partial
from timm.models.layers import trunc_normal_
from vision_transformer import Block as transformer_block
from timm.models.registry import register_model

class Transformer(nn.Module):
    def __init__(self, base_dim, depth, heads, mlp_ratio,
                 drop_rate=.0, attn_drop_rate=.0, drop_path_prob=None):
        super(Transformer, self).__init__()
        self.layers = nn.ModuleList([])
        embed_dim = base_dim * heads

        if drop_path_prob is None:
            drop_path_prob = [0.0 for _ in range(depth)]

        self.blocks = nn.ModuleList([
            transformer_block(
                dim=embed_dim,
                num_heads=heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=True,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=drop_path_prob[i],
                norm_layer=partial(nn.LayerNorm, eps=1e-6)
            )
            for i in range(depth)])

    def forward(self, x):
        h, w = x.shape[2:4]
        x = rearrange(x, 'b c h w -> b (h w) c')
        for blk in self.blocks:
            x = blk(x)

        return x


class bilinear_conv_pooling(nn.Module):
    def __init__(self, in_feature, out_feature, out_size):
        super(bilinear_conv_pooling, self).__init__()

        d = torch.linspace(-1, 1, out_size)
        meshx, meshy = torch.meshgrid((d, d))
        self.grid = torch.stack((meshy, meshx), 2)

        self.conv = nn.Conv2d(in_feature, out_feature, kernel_size=3,
                              padding=1, stride=1)
        self.ln = nn.LayerNorm(in_feature)

    def forward(self, x):
        h = w = int(math.sqrt(x.shape[1]))
        x = self.ln(x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

        grid = self.grid.expand(x.shape[0], -1, -1, -1)
        x = F.grid_sample(x, grid.to(x.device).type_as(x),align_corners=True)
        x = self.conv(x)


        return x

class decoder(nn.Module):
    def __init__(self, in_feature, out_feature, out_size):
        super(decoder, self).__init__()

        self.up = nn.Upsample(size=out_size, mode='bilinear', align_corners=True)
        self.conv = nn.Conv2d(in_feature, out_feature, kernel_size=3,
                              padding=1, stride=1)

    def forward(self, x):
        x = self.conv(x)
        x = self.up(x)
        return x

class conv_embedding(nn.Module):
    def __init__(self, in_channels, out_channels, patch_size,
                 stride, padding):
        super(conv_embedding, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=patch_size,
                              stride=stride, padding=padding, bias=True)

    def forward(self, x):
        x = self.conv(x)
        return x


class PoolingTransformer(nn.Module):
    def __init__(self, image_size, patch_size, stride, base_dims, depth, heads, channels, out_size,
                 mlp_ratio, num_classes=1000, in_chans=3,
                 attn_drop_rate=.0, drop_rate=.0, drop_path_rate=.0, mask_thre=0., pretrained_cfg=None):
        super(PoolingTransformer, self).__init__()

        total_block = sum(depth)
        padding = 0
        block_idx = 0

        width = math.floor(
            (image_size + 2 * padding - patch_size) / stride + 1)

        self.base_dims = base_dims
        self.heads = heads
        self.num_classes = num_classes
        self.mask_thre = mask_thre

        self.patch_size = patch_size
        self.pos_embed = nn.Parameter(
            torch.randn(1, base_dims[0] * heads[0], width, width),
            requires_grad=True
        )
        self.patch_embed = conv_embedding(in_chans, base_dims[0] * heads[0],
                                          patch_size, stride, padding)

        self.pos_drop = nn.Dropout(p=drop_rate)

        self.transformers = nn.ModuleList([])
        self.pools = nn.ModuleList([])
        self.decoders = nn.ModuleList([])

        for stage in range(len(depth)):
            drop_path_prob = [drop_path_rate * i / total_block
                              for i in range(block_idx, block_idx + depth[stage])]
            block_idx += depth[stage]

            self.transformers.append(
                Transformer(base_dims[stage], depth[stage], heads[stage],
                            mlp_ratio,
                            drop_rate, attn_drop_rate, drop_path_prob)
            )
            if stage < len(heads) - 1:
                self.pools.append(
                    bilinear_conv_pooling(channels[stage],
                                      channels[stage+1],
                                      out_size[stage+1]
                                      )
                )
                self.decoders.append(decoder(channels[stage+1], channels[stage], out_size[stage]))

        self.norm = nn.LayerNorm(base_dims[-1] * heads[-1], eps=1e-6)
        self.embed_dim = base_dims[-1] * heads[-1]

        # Classifier head
        if num_classes > 0:
            self.head = nn.Linear(base_dims[-1] * heads[-1], num_classes)
        else:
            self.head = nn.Identity()

        trunc_normal_(self.pos_embed, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        if num_classes > 0:
            self.head = nn.Linear(self.embed_dim, num_classes)
        else:
            self.head = nn.Identity()

    def forward_features(self, x):
        x = self.patch_embed(x)

        pos_embed = self.pos_embed
        x = self.pos_drop(x + pos_embed)

        mid = []
        for stage in range(len(self.pools)):
            xt = self.transformers[stage](x)
            x = self.pools[stage](xt)
            if self.training:
                B, N, C = xt.shape
                mask = torch.rand((B,N,1)).to(xt.device)
                mask = torch.where(mask < self.mask_thre, 0, 1).to(xt.device)
                masked_xt = torch.mul(xt, mask)  / (1 - self.mask_thre)
                masked_x = self.pools[stage](masked_xt)

                out_ae = self.decoders[stage](masked_x)
                h = w = int(math.sqrt(xt.shape[1]))
                xt = rearrange(xt, 'b (h w) c -> b c h w', h=h, w=w)
                mid.append((xt, out_ae))
        x = self.transformers[-1](x)

        x = self.norm(x)
        if self.training:
            return x.mean(dim=1), mid  
        else:
            return x.mean(dim=1)

    def forward(self, x):
        if self.training:
            x, mid = self.forward_features(x)
        else:
            x = self.forward_features(x)
        x = self.head(x)
        if self.training:
            return x, mid  
        else:
            return x


class DistilledPoolingTransformer(PoolingTransformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cls_token = nn.Parameter(
            torch.randn(1, 2, self.base_dims[0] * self.heads[0]),
            requires_grad=True)
        if self.num_classes > 0:
            self.head_dist = nn.Linear(self.base_dims[-1] * self.heads[-1],
                                       self.num_classes)
        else:
            self.head_dist = nn.Identity()

        trunc_normal_(self.cls_token, std=.02)
        self.head_dist.apply(self._init_weights)

    def forward(self, x):
        cls_token = self.forward_features(x)
        x_cls = self.head(cls_token[:, 0])
        x_dist = self.head_dist(cls_token[:, 1])
        if self.training:
            return x_cls, x_dist
        else:
            return (x_cls + x_dist) / 2


@register_model
def fdvit_b(pretrained, **kwargs): 
    model = PoolingTransformer(
        image_size=224,
        patch_size=14,
        stride=7,
        base_dims=[64, 45, 64, 45, 64],
        depth=[3, 3, 3, 2, 2],
        heads=[4, 8, 8, 16, 16],
        channels=[256,360,512,720,1024],
        out_size=[31, 22, 16, 11, 8],
        mlp_ratio=4,
        **kwargs
    )
    return model

@register_model
def fdvit_s(pretrained, **kwargs):
    model = PoolingTransformer(
        image_size=224,
        patch_size=16,
        stride=8,
        base_dims=[48, 34, 32, 34, 38],
        depth=[2, 3, 3, 2, 2],
        heads=[3, 6, 9, 12, 15],
        channels=[144,204,288,408,570],
        out_size=[27, 19, 14, 10, 7],
        mlp_ratio=4,
        **kwargs
    )
    return model

@register_model
def fdvit_ti(pretrained, **kwargs):
    model = PoolingTransformer(
        image_size=224,
        patch_size=16,
        stride=8,
        base_dims=[32, 23, 21, 23, 26],
        depth=[2, 3, 3, 2, 2],
        heads=[2, 4, 6, 8, 10],
        channels=[64,92,126,184,260],
        out_size=[27, 19, 14, 10, 7],
        mlp_ratio=4,
        **kwargs
    )
    return model
