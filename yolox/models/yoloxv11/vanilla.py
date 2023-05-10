import torch
import torch.nn as nn
import numpy as np
from loguru import logger


DEFAULT_CONFIG = [
    {
        'type': 'conv-bn-act',
        'name': 'conv1',
        'input': 'input',
        'is_output': False,
        'in_channels': 3,
        'out_channels': 12,
        'kernel_size': 3,
        'stride': 1,
        'padding': 1,
        'bias': False,
        'activation': 'silu',
        'groups': 1,
    },
    # down sampling: 384 x 384 --> 192 x 192
    {
        'type': 'pool2x2',
        'name': 'pool1',
        'input': 'previous',
        'is_output': False,
    },
    {
        'type': 'conv-bn-act',
        'name': 'conv2',
        'input': 'previous',
        'is_output': False,
        'in_channels': 12,
        'out_channels': 24,
        'kernel_size': 3,
        'stride': 1,
        'padding': 1,
        'bias': True,
        'activation': 'silu',
        'groups': 1,
    },
    # down sampling: 192 x 192 --> 96 x 96
    {
        'type': 'pool2x2',
        'name': 'pool2',
        'input': 'previous',
        'is_output': False,
    },
    {
        'type': 'conv-bn-act',
        'name': 'conv3',
        'input': 'previous',
        'is_output': False,
        'in_channels': 24,
        'out_channels': 48,
        'kernel_size': 3,
        'stride': 1,
        'padding': 1,
        'bias': True,
        'activation': 'silu',
        'groups': 1,
    },
    {
        'type': 'conv-bn-act',
        'name': 'conv4',
        'input': 'previous',
        'is_output': False,
        'in_channels': 48,
        'out_channels': 24,
        'kernel_size': 1,
        'stride': 1,
        'padding': 0,
        'bias': True,
        'activation': 'silu',
        'groups': 1,
    },
    {
        'type': 'conv-bn-act',
        'name': 'conv5',
        'input': 'previous',
        'is_output': False,
        'in_channels': 24,
        'out_channels': 48,
        'kernel_size': 3,
        'stride': 1,
        'padding': 1,
        'bias': True,
        'activation': 'silu',
        'groups': 1,
    },
    # down sampling: 96 x 96 --> 48 x 48
    {
        'type': 'pool2x2',
        'name': 'pool3',
        'input': 'previous',
        'is_output': False,
    },
    {
        'type': 'conv-bn-act',
        'name': 'conv6',
        'input': 'previous',
        'is_output': False,
        'in_channels': 48,
        'out_channels': 96,
        'kernel_size': 3,
        'stride': 1,
        'padding': 1,
        'bias': True,
        'activation': 'silu',
        'groups': 1,
    },
    {
        'type': 'conv-bn-act',
        'name': 'conv7',
        'input': 'previous',
        'is_output': False,
        'in_channels': 96,
        'out_channels': 48,
        'kernel_size': 1,
        'stride': 1,
        'padding': 0,
        'bias': True,
        'activation': 'silu',
        'groups': 1,
    },
    {
        'type': 'conv-bn-act',
        'name': 'backbone_out_48x48',
        'input': 'previous',
        'is_output': True,
        'in_channels': 48,
        'out_channels': 96,
        'kernel_size': 3,
        'stride': 1,
        'padding': 1,
        'bias': True,
        'activation': 'silu',
        'groups': 1,
    },
    # down sampling: 48 x 48 --> 24 x 24
    {
        'type': 'pool2x2',
        'name': 'pool4',
        'input': 'previous',
        'is_output': False,
    },
    {
        'type': 'conv-bn-act',
        'name': 'conv9',
        'input': 'previous',
        'is_output': False,
        'in_channels': 96,
        'out_channels': 192,
        'kernel_size': 3,
        'stride': 1,
        'padding': 1,
        'bias': True,
        'activation': 'silu',
        'groups': 1,
    },
    {
        'type': 'conv-bn-act',
        'name': 'conv10',
        'input': 'previous',
        'is_output': False,
        'in_channels': 192,
        'out_channels': 96,
        'kernel_size': 1,
        'stride': 1,
        'padding': 0,
        'bias': True,
        'activation': 'silu',
        'groups': 1,
    },
    {
        'type': 'conv-bn-act',
        'name': 'backbone_out_24x24',
        'input': 'previous',
        'is_output': True,
        'in_channels': 96,
        'out_channels': 192,
        'kernel_size': 3,
        'stride': 1,
        'padding': 1,
        'bias': True,
        'activation': 'silu',
        'groups': 1,
    },
    # down sampling: 24 x 24 --> 12 x 12
    {
        'type': 'pool2x2',
        'name': 'pool5',
        'input': 'previous',
        'is_output': False,
    },
    {
        'type': 'conv-bn-act',
        'name': 'conv12',
        'input': 'previous',
        'is_output': False,
        'in_channels': 192,
        'out_channels': 384,
        'kernel_size': 3,
        'stride': 1,
        'padding': 1,
        'bias': True,
        'activation': 'silu',
        'groups': 1,
    },
    {
        'type': 'conv-bn-act',
        'name': 'conv13',
        'input': 'previous',
        'is_output': False,
        'in_channels': 384,
        'out_channels': 192,
        'kernel_size': 1,
        'stride': 1,
        'padding': 0,
        'bias': True,
        'activation': 'silu',
        'groups': 1,
    },
    {
        'type': 'conv-bn-act',
        'name': 'backbone_out_12x12',
        'input': 'previous',
        'is_output': True,
        'in_channels': 192,
        'out_channels': 384,
        'kernel_size': 3,
        'stride': 1,
        'padding': 1,
        'bias': True,
        'activation': 'silu',
        'groups': 1,
    }
]

def get_activation(name="silu"):
    if name == "silu":
        module = nn.SiLU(inplace=True)
    elif name == "relu":
        module = nn.ReLU(inplace=True)
    elif name == "lrelu":
        module = nn.LeakyReLU(0.1, inplace=True)
    else:
        raise AttributeError("Unsupported act type: {}".format(name))
    return module


class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias, groups):
        super().__init__()
        assert groups in [-1, 1, None]
        if groups in [-1, None] and kernel_size != 1 and kernel_size != (1, 1):
            self.conv = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=bias,
                groups=in_channels,
            )
        else:
            if groups is None:
                groups = 1
            self.conv = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=bias,
                groups=groups,
            )

    def forward(self, inputs):
        return self.conv(inputs)

class ConvBnAct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias, activation, groups):
        super().__init__()

        if groups in [None, -1] and kernel_size != 1 and kernel_size != (1, 1):
            self.layers = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    bias=False,
                    groups=in_channels,
                ),
                nn.BatchNorm2d(out_channels),
                get_activation(activation),
                nn.Conv2d(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=bias,
                    groups=1,
                ),
                nn.BatchNorm2d(out_channels),
                get_activation(activation),
            )
        else:
            if groups is None:
                groups = 1
            self.layers = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    bias=bias,
                    groups=groups,
                ),
                nn.BatchNorm2d(out_channels),
                get_activation(activation),
            )

    def forward(self, inputs):
        return self.layers(inputs)


class VanillaCNN(nn.Module):
    def __init__(self, nodes):
        super().__init__()
        self.verify_config(nodes)

        self.nodes = self.build_nodes(nodes)
        self.node_metadata, self.output_names = self.build_node_metadata(nodes)

    def get_supported_nodes(self):
        return ['conv-bn-act', 'conv', 'bn', 'act', 'pool2x2', 'sum']

    def verify_config(self, nodes):
        supported_nodes = self.get_supported_nodes()

        # check if node types are supported
        nb_output = 0
        names = []
        prev_node_name = 'input'
        for node in nodes:
            if node['type'] not in supported_nodes:
                logger.error('Unsupported node type: {}'.format(node['type']))
                logger.warning(f'Supported node types: {supported_nodes}')
                raise ValueError('Unsupported node type: {}'.format(node['type']))
            nb_output += int(node['is_output'])
            if node['name'] in names:
                logger.error('Duplicate node name: {}'.format(node['name']))
                raise ValueError('Duplicate node name: {}'.format(node['name']))

            if node['input'] == 'previous':
                node['input'] = prev_node_name
            prev_node_name = node['name']

        # check if there is at least one output node
        if nb_output == 0:
            raise ValueError('None of the provided nodes is an output node')

        # check if the input of each node exists in the previous nodes
        for idx, node in enumerate(nodes):
            if isinstance(node['input'], str) and node['input'] != 'input':
                prev_node_names = [nodes[k]['name'] for k in range(idx)]
                if node['input'] not in prev_node_names:
                    logger.error(f'Node at index {idx} has an input that does not exist in the previous nodes')
                    logger.error(f'node name: {node["name"]} | input: {node["input"]}')
                    logger.error(f'previous node names: {prev_node_names}')
                    raise ValueError(f'Node at index {idx} has an input that does not exist in the previous nodes')
            elif isinstance(node['input'], (tuple, list)):
                prev_node_names = [nodes[k]['name'] for k in range(idx)]
                for input_node in node['input']:
                    if input_node not in prev_node_names:
                        logger.error(f'Node at index {idx} has an input that does not exist in the previous nodes')
                        logger.error(f'node name: {node["name"]} | input: {node["input"]}')
                        logger.error(f'previous node names: {prev_node_names}')
                        raise ValueError(f'Node at index {idx} has an input that does not exist in the previous nodes')

    def build_nodes(self, nodes):
        modules = nn.ModuleList()
        for node in nodes:
            if node['type'] == 'conv-bn-act':
                modules.append(
                    ConvBnAct(
                        in_channels=node['in_channels'],
                        out_channels=node['out_channels'],
                        kernel_size=node['kernel_size'],
                        stride=node['stride'],
                        padding=node['padding'],
                        bias=node['bias'],
                        activation=node['activation'],
                        groups=node['groups'],
                    )
                )
            elif node['type'] == 'conv':
                modules.append(
                    Conv2d(
                        in_channels=node['in_channels'],
                        out_channels=node['out_channels'],
                        kernel_size=node['kernel_size'],
                        stride=node['stride'],
                        padding=node['padding'],
                        bias=node['bias'],
                        activation=node['activation'],
                        groups=node['groups'],
                    )
                )
            elif node['type'] == 'bn':
                modules.append(
                    nn.BatchNorm2d(
                        num_features=node['num_features'],
                        eps=node['eps'],
                        momentum=node['momentum'],
                    )
                )
            elif node['type'] == 'act':
                modules.append(get_activation(node['function']))
            elif node['type'] == 'pool2x2':
                modules.append(
                    nn.MaxPool2d(
                        kernel_size=2,
                        stride=2,
                        padding=0
                    )
                )
            else:
                raise ValueError('Unsupported node type: {}'.format(node['type']))

        return modules

    def build_node_metadata(self, nodes):
        metadata = []
        output_names = []
        for node in nodes:
            metadata.append({
                'name': node['name'],
                'input': node['input'],
                'retain': node['is_output'],
            })
            if node['is_output']:
                output_names.append(node['name'])


        # for node that has inputs containing element from
        # non-immediate-previous nodes
        prev_node = 'input'
        retain = []
        for node in metadata:
            if node['input'] == 'previous':
                node['input'] = prev_node
            else:
                if node['input'] != prev_node:
                    if isinstance(node['input'], str):
                        retain.append(node['input'])
                    else:
                        retain.extend(node['input'])

        for node in metadata:
            if node['name'] in retain:
                node['retain'] = True

        return metadata, output_names

    def forward(self, inputs):
        data = {'input': inputs}
        prev_node = 'input'
        prev_output = inputs

        for node, metadata in zip(self.nodes, self.node_metadata):
            if metadata['input'] == prev_node:
                # if the previous node is the input of the current node
                current_output = node(prev_output)
            else:
                # multiple inputs or the previous node is not the input of the current node
                if isinstance(metadata['input'], str):
                    current_output = node(data[metadata['input']])
                elif isinstance(metadata['input'], (tuple, list)):
                    inputs = [data[name] for name in metadata['input']]
                    current_output = node(*inputs)
                else:
                    raise ValueError('"input" field of a node must be a string or a tuple/list of strings')

            prev_output = current_output
            prev_node = metadata['name']
            if metadata['retain']:
                data[metadata['name']] = current_output

        return [data[name] for name in self.output_names]


if __name__ == '__main__':
    model = VanillaCNN(DEFAULT_CONFIG)
    x = torch.rand(1, 3, 384, 384)
    y = model(x)
    for value in y:
        print(f'shape: {value.shape}')
