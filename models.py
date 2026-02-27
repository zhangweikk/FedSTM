import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision.models.resnet import ResNet, BasicBlock, conv3x3, conv1x1
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from torchvision import models
device = 'cuda' if torch.cuda.is_available() else 'cpu'

channel_dict =  {
    "cifar10": 3,
    "cinic10": 3,
    "cifar100": 3,
    "NWPU_RESISC45": 3,
    "eurosat": 3,
    "mnist": 1,
    "fmnist": 1,
}

############################################################################################################
# MOBILENET
############################################################################################################

# class MLP(nn.Module):
#     def __init__(self, num_classes=10, net_width=128, im_size = (28,28), dataset = 'cifar10'):
#         super(MLP, self).__init__()
#         channel = channel_dict.get(dataset)
#         self.fc1 = nn.Linear(im_size[0]*im_size[1]*channel, net_width)
#         self.fc2 = nn.Linear(net_width, net_width)
#         self.fc3 = nn.Linear(net_width, num_classes)
#
#     def forward(self, x):
#         x = x.view(x.size(0), -1)
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         return self.fc3(x)

class MLP(nn.Module):
    def __init__(self, num_classes=10, net_width=128, im_size=(32, 32), dataset='cifar10'):
        super(MLP, self).__init__()
        channel = channel_dict.get(dataset)
        input_dim = im_size[0] * im_size[1] * channel
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(input_dim, net_width)
        self.fc2 = nn.Linear(net_width, net_width)
        self.classifier = nn.Linear(net_width, num_classes)

    def forward(self, x, return_features=False):
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        features = F.relu(self.fc2(x))
        out = self.classifier(features)
        if return_features:
            return out, features
        else:
            return out

    def get_feature(self, x):
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        features = F.relu(self.fc2(x))
        return features

# class Block(nn.Module):
#     '''expand + depthwise + pointwise'''
#     def __init__(self, in_planes, out_planes, expansion, stride, norm_layer):
#         super(Block, self).__init__()
#         self.stride = stride
#
#         planes = expansion * in_planes
#         self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, stride=1, padding=0, bias=False)
#         self.bn1 = norm_layer(planes)
#         self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, groups=planes, bias=False)
#         self.bn2 = norm_layer(planes)
#         self.conv3 = nn.Conv2d(planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
#         self.bn3 = norm_layer(out_planes)
#
#         self.shortcut = nn.Sequential()
#         if stride == 1 and in_planes != out_planes:
#             self.shortcut = nn.Sequential(
#                 nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False),
#                 norm_layer(out_planes),
#             )
#
#     def forward(self, x):
#         out = F.relu(self.bn1(self.conv1(x)))
#         out = F.relu(self.bn2(self.conv2(out)))
#         out = self.bn3(self.conv3(out))
#         out = out + self.shortcut(x) if self.stride==1 else out
#         return out

# class MobileNetV2(nn.Module):
#     # (expansion, out_planes, num_blocks, stride)
#
#     def __init__(self, num_classes=10, norm_layer=nn.BatchNorm2d,shrink=1, dataset = 'cifar10'):
#         super(MobileNetV2, self).__init__()
#         # NOTE: change conv1 stride 2 -> 1 for CIFAR10
#         self.dataset = dataset
#         channel =  channel_dict.get(dataset)
#         self.norm_layer = norm_layer
#         self.cfg = [(1,  16//shrink, 1, 1),
#                    (6,  24//shrink, 2, 1),  # NOTE: change stride 2 -> 1 for CIFAR10
#                    (6,  32//shrink, 3, 2),
#                    (6,  64//shrink, 4, 2),
#                    (6,  96//shrink, 3, 1),
#                    (6, 160//shrink, 3, 2),
#                    (6, 320//shrink, 1, 1)]
#
#
#         self.conv1 = nn.Conv2d(channel, 32, kernel_size=3, stride=1, padding=1, bias=False)
#         self.bn1 = self.norm_layer(32)
#         self.layers = self._make_layers(in_planes=32)
#         self.conv2 = nn.Conv2d(self.cfg[-1][1], 1280//shrink, kernel_size=1, stride=1, padding=0, bias=False)
#         self.bn2 = self.norm_layer(1280//shrink)
#
#
#         self.classification_layer = nn.Linear(1280//shrink, num_classes)
#
#
#     def _make_layers(self, in_planes):
#         layers = []
#         for expansion, out_planes, num_blocks, stride in self.cfg:
#             strides = [stride] + [1]*(num_blocks-1)
#             for stride in strides:
#                 layers.append(Block(in_planes, out_planes, expansion, stride, self.norm_layer))
#                 in_planes = out_planes
#         return nn.Sequential(*layers)
#
#
#     def extract_features(self, x):
#         out = F.relu(self.bn1(self.conv1(x)))
#         out = self.layers(out)
#         out = F.relu(self.bn2(self.conv2(out)))
#         out = F.avg_pool2d(out, 4)
#         out = out.view(out.size(0), -1)
#         return out
#
#
#     def forward(self, x):
#         feature = self.extract_features(x)
#         out = self.classification_layer(feature)
#         return out
#
#
#
#
# def mobilenetv2(num_classes=10, dataset = 'cifar10'):
#     return MobileNetV2(norm_layer=nn.BatchNorm2d, shrink=2, num_classes=num_classes, dataset = 'cifar10')




############################################################################################################
# MOBILENET 22222222222
############################################################################################################
''' MobileNetV1 '''

class MobileNetV1(nn.Module):
    def __init__(self, num_classes=10, net_width=128, net_depth=3, net_act='relu', net_norm='instancenorm',
                 net_pooling='avgpooling', im_size=(32, 32), dataset='cifar10', alpha=1.0):
        super(MobileNetV1, self).__init__()
        channel = channel_dict.get(dataset)
        self.alpha = alpha  # 宽度乘子，用于控制网络宽度

        self.features, shape_feat = self._make_layers(channel, net_width, net_depth, net_norm, net_act, net_pooling,
                                                      im_size)
        num_feat = shape_feat[0] * shape_feat[1] * shape_feat[2]
        print(f"num feat {num_feat}")
        self.classifier = nn.Linear(num_feat, num_classes)

    def forward(self, x, return_features=False):
        out = self.get_feature(x)
        features = out
        out = self.classifier(out)
        if return_features:
            return out, features
        else:
            return out

    def get_feature(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        return out

    def _get_activation(self, net_act):
        if net_act == 'sigmoid':
            return nn.Sigmoid()
        elif net_act == 'relu':
            return nn.ReLU(inplace=True)
        elif net_act == 'leakyrelu':
            return nn.LeakyReLU(negative_slope=0.01)
        else:
            exit('unknown activation function: %s' % net_act)

    def _get_pooling(self, net_pooling):
        if net_pooling == 'maxpooling':
            return nn.MaxPool2d(kernel_size=2, stride=2)
        elif net_pooling == 'avgpooling':
            return nn.AvgPool2d(kernel_size=2, stride=2)
        elif net_pooling == 'none':
            return None
        else:
            exit('unknown net_pooling: %s' % net_pooling)

    def _get_normlayer(self, net_norm, shape_feat):
        # shape_feat = (c*h*w)
        if net_norm == 'batchnorm':
            return nn.BatchNorm2d(shape_feat[0], affine=True)
        elif net_norm == 'layernorm':
            return nn.LayerNorm(shape_feat, elementwise_affine=True)
        elif net_norm == 'instancenorm':
            return nn.GroupNorm(shape_feat[0], shape_feat[0], affine=True)
        elif net_norm == 'groupnorm':
            return nn.GroupNorm(4, shape_feat[0], affine=True)
        elif net_norm == 'none':
            return None
        else:
            exit('unknown net_norm: %s' % net_norm)

    def _make_depthwise_conv(self, in_channels, stride, net_norm, net_act, shape_feat):
        layers = []
        # 深度卷积 - 每个输入通道有自己的卷积核
        layers.append(nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels))
        shape_feat[0] = in_channels
        if stride == 2:
            shape_feat[1] //= 2
            shape_feat[2] //= 2

        if net_norm != 'none':
            layers.append(self._get_normlayer(net_norm, shape_feat))
        layers.append(self._get_activation(net_act))
        return nn.Sequential(*layers), shape_feat

    def _make_pointwise_conv(self, in_channels, out_channels, net_norm, net_act, shape_feat):
        layers = []
        # 逐点卷积 - 1x1卷积用于改变通道数
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1))
        shape_feat[0] = out_channels

        if net_norm != 'none':
            layers.append(self._get_normlayer(net_norm, shape_feat))
        layers.append(self._get_activation(net_act))
        return nn.Sequential(*layers), shape_feat

    def _make_layers(self, channel, net_width, net_depth, net_norm, net_act, net_pooling, im_size):
        layers = []
        in_channels = channel
        if im_size[0] == 28:
            im_size = (32, 32)
        shape_feat = [in_channels, im_size[0], im_size[1]]

        # 首先使用标准卷积层
        first_filters = int(32 * self.alpha)
        layers.append(nn.Conv2d(in_channels, first_filters, kernel_size=3,
                                padding=1, stride=1))
        shape_feat[0] = first_filters

        if net_norm != 'none':
            layers.append(self._get_normlayer(net_norm, shape_feat))
        layers.append(self._get_activation(net_act))

        # MobileNetV1 配置，(扩展通道数, 步长)
        # 调整深度和宽度以适应框架要求
        base_config = [
            (64, 1), (128, 2), (128, 1),
            (256, 2), (256, 1),
            (512, 2), (512, 1), (512, 1), (512, 1), (512, 1), (512, 1),
            (1024, 2), (1024, 1)
        ]

        # 根据net_depth参数调整网络深度
        config = base_config[:net_depth + 1]  # +1 是因为我们已经添加了第一层

        in_channels = first_filters
        for c, s in config:
            # 使用宽度参数调整通道数
            out_channels = int(c * self.alpha)
            # 采用深度可分离卷积: 深度卷积+逐点卷积
            depthwise, shape_feat = self._make_depthwise_conv(in_channels, s, net_norm, net_act, shape_feat.copy())
            pointwise, shape_feat = self._make_pointwise_conv(in_channels, out_channels, net_norm, net_act,
                                                              shape_feat.copy())

            layers.append(depthwise)
            layers.append(pointwise)

            in_channels = out_channels

        # 如果需要，添加额外的池化层
        if net_pooling != 'none':
            pooling = self._get_pooling(net_pooling)
            if pooling is not None:
                layers.append(pooling)
                shape_feat[1] //= 2
                shape_feat[2] //= 2

        return nn.Sequential(*layers), shape_feat




    

############################################################################################################
# RESNET
############################################################################################################
# class basic_noskip(BasicBlock):
#     expansion: int = 1
#     def __init__(
#             self,
#             *args,
#             **kwargs
#     ) -> None:
#         super(basic_noskip, self).__init__(*args, **kwargs)
#
#     def forward(self, x):
#         out = self.conv1(x)
#         # out = self.bn1(out)
#         out = self.relu(out)
#
#         out = self.conv2(out)
#         # out = self.bn2(out)
#         out = self.relu(out)
#
#         return out
#
# class Model_noskip(nn.Module):
#     def __init__(self, channel=3, feature_dim=128, group_norm=False):
#         super(Model_noskip, self).__init__()
#
#         self.f = []
#         for name, module in ResNet(basic_noskip, [1,1,1,1], num_classes=10).named_children():
#             if name == 'conv1':
#                 module = nn.Conv2d(channel, 64, kernel_size=3, stride=1, padding=1, bias=False)
#             if not isinstance(module, nn.Linear) and not isinstance(module, nn.MaxPool2d):
#                 self.f.append(module)
#         # encoder
#         self.f = nn.Sequential(*self.f)
#         # projection head
#         self.g = nn.Sequential(nn.Linear(512, 512, bias=False), nn.BatchNorm1d(512),
#                                nn.ReLU(inplace=True), nn.Linear(512, feature_dim, bias=True))
#
#         if group_norm:
#             apply_gn(self)
#
#     def forward(self, x):
#         x = self.f(x)
#         feature = torch.flatten(x, start_dim=1)
#         out = self.g(feature)
#         return F.normalize(feature, dim=-1), F.normalize(out, dim=-1)
#
#
# class resnet8_noskip(nn.Module):
#     def __init__(self, num_classes=10, pretrained_path=None, group_norm=False, dataset = 'cifar10'):
#         super(resnet8_noskip, self).__init__()
#         channel =  channel_dict.get(dataset)
#         # encoder
#         self.f = Model_noskip(channel = channel, group_norm=group_norm).f
#         # classifier
#         self.classification_layer = nn.Linear(512, num_classes, bias=True)
#
#
#         if pretrained_path:
#             self.load_state_dict(torch.load(pretrained_path, map_location='cpu'), strict=False)
#
#
#     def extract_features(self, x):
#         return torch.flatten(self.f(x), start_dim=1)
#
#
#     def forward(self, x):
#         feature = self.extract_features(x)
#         out = self.classification_layer(feature)
#         return out
#
# class Model(nn.Module):
#     def __init__(self, channel=3, feature_dim=128, group_norm=False):
#         super(Model, self).__init__()
#
#         self.f = []
#         for name, module in ResNet(BasicBlock, [1,1,1,1], num_classes=10).named_children():
#             if name == 'conv1':
#                 module = nn.Conv2d(channel, 64, kernel_size=3, stride=1, padding=1, bias=False)
#             if not isinstance(module, nn.Linear) and not isinstance(module, nn.MaxPool2d):
#                 self.f.append(module)
#         # encoder
#         self.f = nn.Sequential(*self.f)
#         # projection head
#         self.g = nn.Sequential(nn.Linear(512, 512, bias=False), nn.BatchNorm1d(512),
#                                nn.ReLU(inplace=True), nn.Linear(512, feature_dim, bias=True))
#
#         if group_norm:
#             apply_gn(self)
#
#     def forward(self, x):
#         x = self.f(x)
#         feature = torch.flatten(x, start_dim=1)
#         out = self.g(feature)
#         return F.normalize(feature, dim=-1), F.normalize(out, dim=-1)
#
#
# class resnet8(nn.Module):
#     def __init__(self, num_classes=10, pretrained_path=None, group_norm=False, dataset = 'cifar10'):
#         super(resnet8, self).__init__()
#         channel =  channel_dict.get(dataset)
#         # encoder
#         self.f = Model(channel = channel, group_norm=group_norm).f
#         # classifier
#         self.classification_layer = nn.Linear(512, num_classes, bias=True)
#
#
#         if pretrained_path:
#             self.load_state_dict(torch.load(pretrained_path, map_location='cpu'), strict=False)
#
#
#     def extract_features(self, x):
#         return torch.flatten(self.f(x), start_dim=1)
#
#
#     def forward(self, x):
#         feature = self.extract_features(x)
#         out = self.classification_layer(feature)
#         return out
''' ResNet8 '''

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, net_norm='instancenorm', net_act='relu'):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.norm1 = self._get_normlayer(net_norm, [out_channels, 0, 0])
        self.act = self._get_activation(net_act)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm2 = self._get_normlayer(net_norm, [out_channels, 0, 0])

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                self._get_normlayer(net_norm, [out_channels, 0, 0])
            )

    def forward(self, x):
        out = self.act(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        out += self.shortcut(x)
        out = self.act(out)
        return out

    def _get_activation(self, net_act):
        if net_act == 'sigmoid':
            return nn.Sigmoid()
        elif net_act == 'relu':
            return nn.ReLU(inplace=True)
        elif net_act == 'leakyrelu':
            return nn.LeakyReLU(negative_slope=0.01)
        else:
            exit('unknown activation function: %s' % net_act)

    def _get_normlayer(self, net_norm, shape_feat):
        # shape_feat = [channels, _, _]
        if net_norm == 'batchnorm':
            return nn.BatchNorm2d(shape_feat[0], affine=True)
        elif net_norm == 'layernorm':
            return nn.LayerNorm([shape_feat[0], 1, 1], elementwise_affine=True)
        elif net_norm == 'instancenorm':
            return nn.GroupNorm(shape_feat[0], shape_feat[0], affine=True)
        elif net_norm == 'groupnorm':
            return nn.GroupNorm(4, shape_feat[0], affine=True)
        elif net_norm == 'none':
            return nn.Identity()
        else:
            exit('unknown net_norm: %s' % net_norm)


class ResNet8(nn.Module):
    def __init__(self, num_classes=10, net_width=64, net_depth=8, net_act='relu', net_norm='instancenorm',
                 net_pooling='avgpooling', im_size=(32, 32), dataset='cifar10'):
        super(ResNet8, self).__init__()
        channel = channel_dict.get(dataset)

        # 第一层卷积
        self.conv1 = nn.Conv2d(channel, net_width, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm1 = self._get_normlayer(net_norm, [net_width, im_size[0], im_size[1]])
        self.act = self._get_activation(net_act)

        # 设置通道数和每层块数
        channels = [net_width, net_width * 2, net_width * 4]
        blocks_per_layer = 2

        # 残差层
        self.layer1 = self._make_layer(BasicBlock, channels[0], blocks_per_layer, stride=1, net_norm=net_norm,
                                       net_act=net_act, in_channels=net_width)
        self.layer2 = self._make_layer(BasicBlock, channels[1], blocks_per_layer, stride=2, net_norm=net_norm,
                                       net_act=net_act, in_channels=channels[0])
        self.layer3 = self._make_layer(BasicBlock, channels[2], blocks_per_layer, stride=2, net_norm=net_norm,
                                       net_act=net_act, in_channels=channels[1])

        # 全局池化
        self.pool = self._get_pooling(net_pooling)

        # 正确计算特征维度
        # 考虑两次stride=2的下采样，计算最终特征图大小
        if im_size[0] == 28:  # 对于MNIST等28x28的数据集
            feat_size = 7  # 28 / 4 = 7
        else:  # 对于CIFAR10等32x32的数据集
            feat_size = 8  # 32 / 4 = 8

        # 检查是否使用了池化层
        if net_pooling != 'none' and self.pool is not None:
            # 如果使用了额外的池化层，特征尺寸再次减半
            feat_size = feat_size // 2

        # 计算最终展平后的特征数量
        self.num_feat = channels[2] * feat_size * feat_size
        print(f"计算得到的特征维度: {self.num_feat}")

        # 分类器
        self.classifier = nn.Linear(self.num_feat, num_classes)

    def forward(self, x, return_features=False):
        out = self.get_feature(x)
        features = out
        out = self.classifier(out)
        if return_features:
            return out, features
        else:
            return out

    def get_feature(self, x):
        out = self.act(self.norm1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)

        # 应用池化层（如果有）
        if self.pool is not None:
            out = self.pool(out)

        # 打印特征形状以便调试
        batch_size = out.size(0)
        feat_shape = out.size()[1:]
        feat_size = torch.prod(torch.tensor(feat_shape)).item()

        # 展平操作
        out = out.view(batch_size, -1)
        return out

    def _make_layer(self, block, out_channels, num_blocks, stride, net_norm, net_act, in_channels):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []

        for i, current_stride in enumerate(strides):
            # 第一个块使用传入的in_channels，后续块使用out_channels
            current_in_channels = in_channels if i == 0 else out_channels
            layers.append(block(current_in_channels, out_channels, current_stride, net_norm, net_act))

        return nn.Sequential(*layers)

    def _get_activation(self, net_act):
        if net_act == 'sigmoid':
            return nn.Sigmoid()
        elif net_act == 'relu':
            return nn.ReLU(inplace=True)
        elif net_act == 'leakyrelu':
            return nn.LeakyReLU(negative_slope=0.01)
        else:
            exit('unknown activation function: %s' % net_act)

    def _get_pooling(self, net_pooling):
        if net_pooling == 'maxpooling':
            return nn.MaxPool2d(kernel_size=2, stride=2)
        elif net_pooling == 'avgpooling':
            return nn.AvgPool2d(kernel_size=2, stride=2)
        elif net_pooling == 'none':
            return None
        else:
            exit('unknown net_pooling: %s' % net_pooling)

    def _get_normlayer(self, net_norm, shape_feat):
        # shape_feat = (c*h*w)
        if net_norm == 'batchnorm':
            return nn.BatchNorm2d(shape_feat[0], affine=True)
        elif net_norm == 'layernorm':
            return nn.LayerNorm(shape_feat, elementwise_affine=True)
        elif net_norm == 'instancenorm':
            return nn.GroupNorm(shape_feat[0], shape_feat[0], affine=True)
        elif net_norm == 'groupnorm':
            return nn.GroupNorm(4, shape_feat[0], affine=True)
        elif net_norm == 'none':
            return nn.Identity()
        else:
            exit('unknown net_norm: %s' % net_norm)

############################################################################################################
# GhostNet
############################################################################################################
''' GhostNet '''

class GhostModule(nn.Module):
    def __init__(self, inp, oup, kernel_size=1, ratio=2, dw_size=3, stride=1, relu=True):
        super(GhostModule, self).__init__()
        self.oup = oup
        init_channels = math.ceil(oup / ratio)
        new_channels = init_channels * (ratio - 1)

        self.primary_conv = nn.Sequential(
            nn.Conv2d(inp, init_channels, kernel_size, stride, kernel_size // 2, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size // 2, groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)
        return out[:, :self.oup, :, :]


class GhostBottleneck(nn.Module):
    def __init__(self, inp, hidden_dim, oup, kernel_size, stride, use_se=False):
        super(GhostBottleneck, self).__init__()
        self.stride = stride

        # 短路连接的判断条件
        self.use_shortcut = (stride == 1 and inp == oup)

        # 第一个Ghost模块
        self.ghost1 = GhostModule(inp, hidden_dim, kernel_size=1, relu=True)

        # 深度卷积模块（用来降采样）
        if stride > 1:
            self.downsampling = nn.Sequential(
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=stride,
                          padding=1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True)
            )
        else:
            self.downsampling = nn.Sequential()

        # 第二个Ghost模块
        self.ghost2 = GhostModule(hidden_dim, oup, kernel_size=1, relu=False)

        # shortcut路径上的降采样处理（如果需要）
        if self.use_shortcut:
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inp, inp, kernel_size=3, stride=stride,
                          padding=1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.Conv2d(inp, oup, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        residual = x

        # 1st ghost bottleneck
        x = self.ghost1(x)

        # 降采样
        x = self.downsampling(x)

        # 2nd ghost bottleneck
        x = self.ghost2(x)

        # 短路连接
        if self.use_shortcut:
            x = x + residual
        else:
            x = x + self.shortcut(residual)

        return x


class GhostNet(nn.Module):
    def __init__(self, num_classes=10, net_width=128, net_depth=3, net_act='relu', net_norm='instancenorm',
                 net_pooling='avgpooling', im_size=(32, 32), dataset='cifar10'):
        super(GhostNet, self).__init__()
        channel = channel_dict.get(dataset)

        # 严格按照ConvNet的方式处理特征形状追踪
        shape_feat = [channel, im_size[0], im_size[1]]
        if im_size[0] == 28:
            shape_feat = [channel, 32, 32]

        # 构建特征提取器，完全仿照ConvNet的设计模式
        layers = []
        in_channels = channel

        # 第一层是普通卷积，与ConvNet保持一致
        layers.append(nn.Conv2d(in_channels, net_width, kernel_size=3,
                                padding=3 if channel == 1 and 0 == 0 else 1))
        shape_feat[0] = net_width

        # 添加规范化层
        if net_norm != 'none':
            layers.append(self._get_normlayer(net_norm, shape_feat))
        layers.append(self._get_activation(net_act))
        in_channels = net_width

        # 添加池化层
        if net_pooling != 'none':
            layers.append(self._get_pooling(net_pooling))
            shape_feat[1] //= 2
            shape_feat[2] //= 2

        # 添加剩余的Ghost模块
        for d in range(1, net_depth):
            # 使用GhostBottleneck代替传统卷积
            hidden_dim = net_width * 2

            # 只有在需要降采样时使用stride=2
            if net_pooling != 'none' and d < net_depth - 1:
                layers.append(GhostBottleneck(in_channels, hidden_dim, net_width, kernel_size=3, stride=2))
                shape_feat[1] //= 2
                shape_feat[2] //= 2
            else:
                layers.append(GhostBottleneck(in_channels, hidden_dim, net_width, kernel_size=3, stride=1))

            # 更新输入通道数
            in_channels = net_width

        self.features = nn.Sequential(*layers)

        # 计算最终特征尺寸
        num_feat = shape_feat[0] * shape_feat[1] * shape_feat[2]
        print(f"num feat {num_feat}")
        self.classifier = nn.Linear(num_feat, num_classes)

    def forward(self, x, return_features=False):
        # 与ConvNet保持完全一致的接口
        out = self.get_feature(x)
        features = out
        out = self.classifier(out)
        if return_features:
            return out, features
        else:
            return out

    def get_feature(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        return out

    def _get_activation(self, net_act):
        if net_act == 'sigmoid':
            return nn.Sigmoid()
        elif net_act == 'relu':
            return nn.ReLU(inplace=True)
        elif net_act == 'leakyrelu':
            return nn.LeakyReLU(negative_slope=0.01)
        else:
            exit('unknown activation function: %s' % net_act)

    def _get_pooling(self, net_pooling):
        if net_pooling == 'maxpooling':
            return nn.MaxPool2d(kernel_size=2, stride=2)
        elif net_pooling == 'avgpooling':
            return nn.AvgPool2d(kernel_size=2, stride=2)
        elif net_pooling == 'none':
            return None
        else:
            exit('unknown net_pooling: %s' % net_pooling)

    def _get_normlayer(self, net_norm, shape_feat):
        # shape_feat = (c*h*w)
        if net_norm == 'batchnorm':
            return nn.BatchNorm2d(shape_feat[0], affine=True)
        elif net_norm == 'layernorm':
            return nn.LayerNorm(shape_feat, elementwise_affine=True)
        elif net_norm == 'instancenorm':
            return nn.GroupNorm(shape_feat[0], shape_feat[0], affine=True)
        elif net_norm == 'groupnorm':
            return nn.GroupNorm(4, shape_feat[0], affine=True)
        elif net_norm == 'none':
            return None
        else:
            exit('unknown net_norm: %s' % net_norm)
############################################################################################################
# SHUFFLENET
############################################################################################################

''' ShuffleNet '''

class ShuffleBlock(nn.Module):
    def __init__(self, groups):
        super(ShuffleBlock, self).__init__()
        self.groups = groups

    def forward(self, x):
        '''Channel shuffle: [N,C,H,W] -> [N,g,C/g,H,W] -> [N,C/g,g,H,W] -> [N,C,H,W]'''
        N, C, H, W = x.size()
        g = self.groups
        # 处理通道数不能被groups整除的情况
        if C % g != 0:
            # 如果不能整除，使用对齐到最近的可整除值
            C_new = (C // g) * g
            x = x[:, :C_new, :, :]
            N, C, H, W = x.size()
        return x.view(N, g, C // g, H, W).permute(0, 2, 1, 3, 4).reshape(N, C, H, W)


class ShuffleUnit(nn.Module):
    def __init__(self, in_channels, out_channels, stride, groups=2, net_norm='instancenorm', net_act='relu'):
        super(ShuffleUnit, self).__init__()

        # 确保通道数是groups的整数倍
        self.in_channels = (in_channels // groups) * groups
        self.groups = groups
        self.stride = stride

        # 当stride=1时保持通道数不变，否则增加通道
        if stride == 1:
            self.out_channels = self.in_channels
        else:
            self.out_channels = out_channels

        # 中间通道数设置
        bottleneck_channels = self.out_channels // 4
        bottleneck_channels = max(bottleneck_channels, self.groups)
        # 确保bottleneck_channels是groups的整数倍
        bottleneck_channels = (bottleneck_channels // self.groups) * self.groups

        # 卷积层1：pointwise grouped convolution
        self.conv1 = nn.Conv2d(
            self.in_channels,
            bottleneck_channels,
            kernel_size=1,
            groups=self.groups,
            bias=False
        )
        self.bn1 = self._get_normlayer(net_norm, bottleneck_channels)
        self.relu1 = self._get_activation(net_act)

        # 通道混洗层
        self.shuffle = ShuffleBlock(self.groups)

        # 卷积层2：depthwise convolution
        self.conv2 = nn.Conv2d(
            bottleneck_channels,
            bottleneck_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=bottleneck_channels,
            bias=False
        )
        self.bn2 = self._get_normlayer(net_norm, bottleneck_channels)

        # 卷积层3：pointwise grouped convolution
        # 如果stride=2，使用concat而不是add，输出通道数不同
        if stride == 2:
            self.conv3 = nn.Conv2d(
                bottleneck_channels,
                self.out_channels - self.in_channels,
                kernel_size=1,
                groups=self.groups,
                bias=False
            )
            self.bn3 = self._get_normlayer(net_norm, self.out_channels - self.in_channels)
            # 创建shortcut分支
            self.shortcut = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        else:
            self.conv3 = nn.Conv2d(
                bottleneck_channels,
                self.out_channels,
                kernel_size=1,
                groups=self.groups,
                bias=False
            )
            self.bn3 = self._get_normlayer(net_norm, self.out_channels)
            self.shortcut = nn.Identity()

    def _get_normlayer(self, net_norm, num_channels):
        """返回归一化层"""
        if net_norm == 'batchnorm':
            return nn.BatchNorm2d(num_channels, affine=True)
        elif net_norm == 'instancenorm':
            return nn.GroupNorm(num_channels, num_channels, affine=True)
        elif net_norm == 'groupnorm':
            return nn.GroupNorm(4, num_channels, affine=True)
        elif net_norm == 'layernorm':
            return nn.GroupNorm(1, num_channels, affine=True)  # 使用GroupNorm(1,...)替代LayerNorm
        else:
            return nn.Identity()

    def _get_activation(self, net_act):
        """返回激活函数"""
        if net_act == 'relu':
            return nn.ReLU(inplace=True)
        elif net_act == 'leakyrelu':
            return nn.LeakyReLU(negative_slope=0.01, inplace=True)
        elif net_act == 'sigmoid':
            return nn.Sigmoid()
        else:
            return nn.ReLU(inplace=True)  # 默认使用ReLU

    def forward(self, x):
        # 确保输入通道数与预期一致
        if x.size(1) != self.in_channels:
            # 如果不一致，调整通道数
            if x.size(1) > self.in_channels:
                x = x[:, :self.in_channels, :, :]
            else:
                # 如果输入通道少于预期，这种情况不应该发生
                # 输出警告并使用零填充
                print(f"Warning: Input has {x.size(1)} channels, expected {self.in_channels}")
                padding = torch.zeros(x.size(0), self.in_channels - x.size(1), x.size(2), x.size(3),
                                      dtype=x.dtype, device=x.device)
                x = torch.cat([x, padding], dim=1)

        # 主路径
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.shuffle(out)
        out = self.bn2(self.conv2(out))
        out = self.bn3(self.conv3(out))

        # 短路连接
        if self.stride == 2:
            # 如果stride=2，使用concat
            shortcut = self.shortcut(x)
            out = torch.cat([out, shortcut], dim=1)
            return F.relu(out)
        else:
            # 如果stride=1，使用add
            return F.relu(out + self.shortcut(x))


class ShuffleNet(nn.Module):
    def __init__(self, num_classes=10, net_width=128, net_depth=3, net_act='relu',
                 net_norm='instancenorm', net_pooling='avgpooling', im_size=(64, 64),
                 dataset='eurosat', groups=2):
        super(ShuffleNet, self).__init__()

        # 获取输入通道数
        self.input_channels = channel_dict.get(dataset, 3)
        self.groups = groups
        self.net_depth = max(net_depth, 3)  # 至少3层

        # 调整网络宽度确保是groups的整数倍
        net_width = max((net_width // groups) * groups, groups * 4)

        # 定义每个阶段的通道数
        if net_width <= 144:
            stage_out_channels = [24, net_width, net_width * 2, net_width * 4]
        else:
            stage_out_channels = [24, net_width, net_width * 2, net_width]

        # 确保所有通道都是groups的整数倍
        for i in range(len(stage_out_channels)):
            if i > 0:  # 第一个阶段的通道数可以不是groups的整数倍
                stage_out_channels[i] = (stage_out_channels[i] // groups) * groups

        # 第一层卷积
        self.conv1 = nn.Conv2d(self.input_channels, stage_out_channels[0],
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = self._get_normlayer(net_norm, stage_out_channels[0])
        self.relu1 = self._get_activation(net_act)

        # 如果设置了池化层
        self.pool = None
        if net_pooling != 'none':
            self.pool = self._get_pooling(net_pooling)

        # 构建ShuffleNet阶段
        self.stage2 = self._make_stage(stage_out_channels[0], stage_out_channels[1],
                                       repeats=net_depth // 3, stride=2,
                                       groups=groups, net_norm=net_norm, net_act=net_act)

        self.stage3 = self._make_stage(stage_out_channels[1], stage_out_channels[2],
                                       repeats=net_depth // 3, stride=2,
                                       groups=groups, net_norm=net_norm, net_act=net_act)

        self.stage4 = self._make_stage(stage_out_channels[2], stage_out_channels[3],
                                       repeats=net_depth // 3, stride=2,
                                       groups=groups, net_norm=net_norm, net_act=net_act)

        # 计算特征维度
        final_channels = stage_out_channels[3]
        h, w = im_size

        # 计算经过网络后的特征图尺寸
        if self.pool:
            h, w = h // 2, w // 2  # 第一个池化层

        h, w = h // 8, w // 8  # 三个stride=2的阶段

        # 打印特征数量
        num_features = final_channels * h * w
        print(f"num feat {num_features}")

        # 分类器
        self.classifier = nn.Linear(num_features, num_classes)

        # 初始化权重
        self._initialize_weights()

    def _make_stage(self, in_channels, out_channels, repeats, stride, groups, net_norm, net_act):
        """构建一个ShuffleNet阶段"""
        layers = []

        # 阶段的第一个单元处理stride和通道变化
        layers.append(ShuffleUnit(in_channels, out_channels, stride=stride,
                                  groups=groups, net_norm=net_norm, net_act=net_act))

        # 阶段的剩余单元
        for _ in range(repeats - 1):
            layers.append(ShuffleUnit(out_channels, out_channels, stride=1,
                                      groups=groups, net_norm=net_norm, net_act=net_act))

        return nn.Sequential(*layers)

    def _initialize_weights(self):
        """初始化模型权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x, return_features=False):
        """前向传播"""
        out = self.get_feature(x)
        features = out
        out = self.classifier(out)
        if return_features:
            return out, features
        else:
            return out

    def get_feature(self, x):
        """获取特征向量"""
        out = self.relu1(self.bn1(self.conv1(x)))

        if self.pool:
            out = self.pool(out)

        out = self.stage2(out)
        out = self.stage3(out)
        out = self.stage4(out)

        out = out.view(out.size(0), -1)
        return out

    def _get_activation(self, net_act):
        """返回激活函数"""
        if net_act == 'relu':
            return nn.ReLU(inplace=True)
        elif net_act == 'leakyrelu':
            return nn.LeakyReLU(negative_slope=0.01, inplace=True)
        elif net_act == 'sigmoid':
            return nn.Sigmoid()
        else:
            return nn.ReLU(inplace=True)  # 默认使用ReLU

    def _get_pooling(self, net_pooling):
        """返回池化层"""
        if net_pooling == 'maxpooling':
            return nn.MaxPool2d(kernel_size=2, stride=2)
        elif net_pooling == 'avgpooling':
            return nn.AvgPool2d(kernel_size=2, stride=2)
        else:
            return None

    def _get_normlayer(self, net_norm, num_channels):
        """返回归一化层"""
        if net_norm == 'batchnorm':
            return nn.BatchNorm2d(num_channels, affine=True)
        elif net_norm == 'instancenorm':
            return nn.GroupNorm(num_channels, num_channels, affine=True)
        elif net_norm == 'groupnorm':
            return nn.GroupNorm(4, num_channels, affine=True)
        elif net_norm == 'layernorm':
            return nn.GroupNorm(1, num_channels, affine=True)  # 使用GroupNorm(1,...)替代LayerNorm
        else:
            return nn.Identity()


''' ConvNet '''
class ConvNet(nn.Module):
    def __init__(self, num_classes=10, net_width=128, net_depth=3, net_act='relu', net_norm='instancenorm', net_pooling='avgpooling', im_size = (32,32), dataset = 'cifar10'):
        super(ConvNet, self).__init__()
        channel =  channel_dict.get(dataset)
        self.features, shape_feat = self._make_layers(channel, net_width, net_depth, net_norm, net_act, net_pooling, im_size)
        num_feat = shape_feat[0]*shape_feat[1]*shape_feat[2]
        print(f"num feat {num_feat}")
        self.classifier = nn.Linear(num_feat, num_classes)

    def forward(self, x, return_features=False):
        # print("MODEL DATA ON: ", x.get_device(), "MODEL PARAMS ON: ", self.classifier.weight.data.get_device())
        out = self.get_feature(x)
        features = out
        out = self.classifier(out)
        if return_features:
            return out, features
        else:
            return out

    def get_feature(self,x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        return out

    def _get_activation(self, net_act):
        if net_act == 'sigmoid':
            return nn.Sigmoid()
        elif net_act == 'relu':
            return nn.ReLU(inplace=True)
        elif net_act == 'leakyrelu':
            return nn.LeakyReLU(negative_slope=0.01)
        else:
            exit('unknown activation function: %s'%net_act)

    def _get_pooling(self, net_pooling):
        if net_pooling == 'maxpooling':
            return nn.MaxPool2d(kernel_size=2, stride=2)
        elif net_pooling == 'avgpooling':
            return nn.AvgPool2d(kernel_size=2, stride=2)
        elif net_pooling == 'none':
            return None
        else:
            exit('unknown net_pooling: %s'%net_pooling)

    def _get_normlayer(self, net_norm, shape_feat):
        # shape_feat = (c*h*w)
        if net_norm == 'batchnorm':
            return nn.BatchNorm2d(shape_feat[0], affine=True)
        elif net_norm == 'layernorm':
            return nn.LayerNorm(shape_feat, elementwise_affine=True)
        elif net_norm == 'instancenorm':
            return nn.GroupNorm(shape_feat[0], shape_feat[0], affine=True)
        elif net_norm == 'groupnorm':
            return nn.GroupNorm(4, shape_feat[0], affine=True)
        elif net_norm == 'none':
            return None
        else:
            exit('unknown net_norm: %s'%net_norm)

    def _make_layers(self, channel, net_width, net_depth, net_norm, net_act, net_pooling, im_size):
        layers = []
        in_channels = channel
        if im_size[0] == 28:
            im_size = (32, 32)
        shape_feat = [in_channels, im_size[0], im_size[1]]
        for d in range(net_depth):
            layers += [nn.Conv2d(in_channels, net_width, kernel_size=3, padding=3 if channel == 1 and d == 0 else 1)]
            shape_feat[0] = net_width
            if net_norm != 'none':
                layers += [self._get_normlayer(net_norm, shape_feat)]
            layers += [self._get_activation(net_act)]
            in_channels = net_width
            if net_pooling != 'none':
                layers += [self._get_pooling(net_pooling)]
                shape_feat[1] //= 2
                shape_feat[2] //= 2


        return nn.Sequential(*layers), shape_feat



class TextModel(nn.Module):

    def __init__(self, vocab_size=95811, embed_dim=64, num_classes=4):
        super(TextModel, self).__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim)
        self.fc = nn.Linear(embed_dim, num_classes)
        self.init_weights()

    def init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()

    def forward(self, text, offsets):
        embedded = self.embedding(text, offsets)
        return self.fc(embedded)

class LogisticRegression(nn.Module):
    def __init__(self, input_dim=130107, num_classes=20):
        super(LogisticRegression, self).__init__()
        self.fc = torch.nn.Parameter(torch.zeros(input_dim, num_classes))


    def forward(self, x):
        out = x @ self.fc
        return out

def get_model(model):

  return  {   "mobilenetv1" : (MobileNetV1, optim.Adam, {"lr" : 0.001}),
              "shufflenet" : (ShuffleNet, optim.Adam, {"lr" : 0.001}),
              "GhostNet": (GhostNet, optim.Adam, {"lr": 0.001}),
                "resnet8" : (ResNet8, optim.Adam, {"lr" : 0.001}),
                # "resnet8_noskip" : (resnet8_noskip, optim.Adam, {"lr" : 0.001}),
                "ConvNet" : (ConvNet, optim.Adam, {"lr" : 0.001}),
                "MLP" : (MLP, optim.Adam, {"lr" : 0.001}),
                "TextModel" : (TextModel, optim.Adam, {"lr" : 1}),
                "LogisticRegression" : (LogisticRegression, optim.Adam, {"lr" : 0.001}),
          }[model]


def print_model(model):
  n = 0
  print("Model:")
  for key, value in model.named_parameters():
    print(' -', '{:30}'.format(key), list(value.shape), "Requires Grad:", value.requires_grad)
    n += value.numel()
  print("Total number of Parameters: ", n) 
  print()






