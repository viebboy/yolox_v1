#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) Megvii Inc. All rights reserved.

from torch import nn

from .network_blocks import BaseConv, CSPLayer, DWConv, Focus, ResLayer, SPPBottleneck


class Darknet(nn.Module):
    # number of blocks from dark2 to dark5.
    depth2blocks = {21: [1, 2, 2, 1], 53: [2, 8, 8, 4]}

    def __init__(
        self,
        depth,
        in_channels=3,
        stem_out_channels=32,
        out_features=("dark3", "dark4", "dark5"),
    ):
        """
        Args:
            depth (int): depth of darknet used in model, usually use [21, 53] for this param.
            in_channels (int): number of input channels, for example, use 3 for RGB image.
            stem_out_channels (int): number of output channels of darknet stem.
                It decides channels of darknet layer2 to layer5.
            out_features (Tuple[str]): desired output layer name.
        """
        super().__init__()
        assert out_features, "please provide output features of Darknet"
        self.out_features = out_features
        self.stem = nn.Sequential(
            BaseConv(in_channels, stem_out_channels, ksize=3, stride=1, act="lrelu"),
            *self.make_group_layer(stem_out_channels, num_blocks=1, stride=2),
        )
        in_channels = stem_out_channels * 2  # 64

        num_blocks = Darknet.depth2blocks[depth]
        # create darknet with `stem_out_channels` and `num_blocks` layers.
        # to make model structure more clear, we don't use `for` statement in python.
        self.dark2 = nn.Sequential(
            *self.make_group_layer(in_channels, num_blocks[0], stride=2)
        )
        in_channels *= 2  # 128
        self.dark3 = nn.Sequential(
            *self.make_group_layer(in_channels, num_blocks[1], stride=2)
        )
        in_channels *= 2  # 256
        self.dark4 = nn.Sequential(
            *self.make_group_layer(in_channels, num_blocks[2], stride=2)
        )
        in_channels *= 2  # 512

        self.dark5 = nn.Sequential(
            *self.make_group_layer(in_channels, num_blocks[3], stride=2),
            *self.make_spp_block([in_channels, in_channels * 2], in_channels * 2),
        )

    def make_group_layer(self, in_channels: int, num_blocks: int, stride: int = 1):
        "starts with conv layer then has `num_blocks` `ResLayer`"
        return [
            BaseConv(in_channels, in_channels * 2, ksize=3, stride=stride, act="lrelu"),
            *[(ResLayer(in_channels * 2)) for _ in range(num_blocks)],
        ]

    def make_spp_block(self, filters_list, in_filters):
        m = nn.Sequential(
            *[
                BaseConv(in_filters, filters_list[0], 1, stride=1, act="lrelu"),
                BaseConv(filters_list[0], filters_list[1], 3, stride=1, act="lrelu"),
                SPPBottleneck(
                    in_channels=filters_list[1],
                    out_channels=filters_list[0],
                    activation="lrelu",
                ),
                BaseConv(filters_list[0], filters_list[1], 3, stride=1, act="lrelu"),
                BaseConv(filters_list[1], filters_list[0], 1, stride=1, act="lrelu"),
            ]
        )
        return m

    def forward(self, x):
        outputs = {}
        x = self.stem(x)
        outputs["stem"] = x
        x = self.dark2(x)
        outputs["dark2"] = x
        x = self.dark3(x)
        outputs["dark3"] = x
        x = self.dark4(x)
        outputs["dark4"] = x
        x = self.dark5(x)
        outputs["dark5"] = x
        return {k: v for k, v in outputs.items() if k in self.out_features}


class CSPDarknet(nn.Module):
    def __init__(
        self,
        nb_init_filters,
        groups,
        stages,
        grow_rates,
        out_features=("dark3", "dark4", "dark5"),
        depthwise=False,
        act="silu",
    ):
        super().__init__()
        assert out_features, "please provide output features of Darknet"
        self.out_features = out_features
        Conv = DWConv if depthwise else BaseConv


        # stem
        self.stem = Focus(3, nb_init_filters, ksize=3, act=act)

        # dark2
        if groups[0] is not None:
            self.dark2 = nn.Sequential(
                Conv(nb_init_filters, int(nb_init_filters * grow_rates[0]), 3, 2, act=act, groups=groups[0]),
                CSPLayer(
                    int(nb_init_filters * grow_rates[0]),
                    int(nb_init_filters * grow_rates[0]),
                    n=stages[0],
                    groups=groups[0],
                    act=act,
                ),
            )
        else:
            self.dark2 = nn.Sequential(
                DWConv(nb_init_filters, int(nb_init_filters * grow_rates[0]), 3, 2, act=act),
                CSPLayer(
                    int(nb_init_filters * grow_rates[0]),
                    int(nb_init_filters * grow_rates[0]),
                    n=stages[0],
                    groups=groups[0],
                    act=act,
                ),
            )

        # dark3
        if groups[1] is not None:
            self.dark3 = nn.Sequential(
                Conv(
                    int(nb_init_filters * grow_rates[0]),
                    int(nb_init_filters * grow_rates[1]),
                    3,
                    2,
                    act=act,
                    groups=groups[1]
                ),
                CSPLayer(
                    int(nb_init_filters * grow_rates[1]),
                    int(nb_init_filters * grow_rates[1]),
                    n=stages[1],
                    groups=groups[1],
                    act=act,
                ),
            )
        else:
            self.dark3 = nn.Sequential(
                DWConv(
                    int(nb_init_filters * grow_rates[0]),
                    int(nb_init_filters * grow_rates[1]),
                    3,
                    2,
                    act=act,
                ),
                CSPLayer(
                    int(nb_init_filters * grow_rates[1]),
                    int(nb_init_filters * grow_rates[1]),
                    n=stages[1],
                    groups=groups[1],
                    act=act,
                ),
            )

        # dark4
        if groups[2] is not None:
            self.dark4 = nn.Sequential(
                Conv(
                    int(nb_init_filters * grow_rates[1]),
                    int(nb_init_filters * grow_rates[2]),
                    3,
                    2,
                    act=act,
                    groups=groups[2],
                ),
                CSPLayer(
                    int(nb_init_filters * grow_rates[2]),
                    int(nb_init_filters * grow_rates[2]),
                    n=stages[2],
                    groups=groups[2],
                    act=act,
                ),
            )
        else:
            self.dark4 = nn.Sequential(
                DWConv(
                    int(nb_init_filters * grow_rates[1]),
                    int(nb_init_filters * grow_rates[2]),
                    3,
                    2,
                    act=act,
                ),
                CSPLayer(
                    int(nb_init_filters * grow_rates[2]),
                    int(nb_init_filters * grow_rates[2]),
                    n=stages[2],
                    groups=groups[2],
                    act=act,
                ),
            )

        # dark5
        if groups[3] is not None:
            self.dark5 = nn.Sequential(
                Conv(
                    int(nb_init_filters * grow_rates[2]),
                    int(nb_init_filters * grow_rates[3]),
                    3,
                    2,
                    act=act,
                    groups=groups[3],
                ),
                SPPBottleneck(
                    int(nb_init_filters * grow_rates[3]),
                    int(nb_init_filters * grow_rates[3]),
                    activation=act,
                    groups=groups[3],
                ),
                CSPLayer(
                    int(nb_init_filters * grow_rates[3]),
                    int(nb_init_filters * grow_rates[3]),
                    n=stages[3],
                    groups=groups[3],
                    act=act,
                ),
            )
        else:
            self.dark5 = nn.Sequential(
                DWConv(
                    int(nb_init_filters * grow_rates[2]),
                    int(nb_init_filters * grow_rates[3]),
                    3,
                    2,
                    act=act,
                ),
                SPPBottleneck(
                    int(nb_init_filters * grow_rates[3]),
                    int(nb_init_filters * grow_rates[3]),
                    activation=act,
                    groups=groups[3],
                ),
                CSPLayer(
                    int(nb_init_filters * grow_rates[3]),
                    int(nb_init_filters * grow_rates[3]),
                    n=stages[3],
                    groups=groups[3],
                    act=act,
                ),
            )

    def forward(self, x):
        outputs = {}
        x = self.stem(x)
        outputs["stem"] = x
        x = self.dark2(x)
        outputs["dark2"] = x
        x = self.dark3(x)
        outputs["dark3"] = x
        x = self.dark4(x)
        outputs["dark4"] = x
        x = self.dark5(x)
        outputs["dark5"] = x
        return {k: v for k, v in outputs.items() if k in self.out_features}
