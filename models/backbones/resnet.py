#------------------------------------------------------------------------------
#  Libraries
#------------------------------------------------------------------------------
from base import BaseBackboneWrapper
from timm.models.resnet import ResNet as BaseResNet
from timm.models.resnet import default_cfgs, load_pretrained, BasicBlock, Bottleneck

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


#------------------------------------------------------------------------------
#   ResNetBlock
#------------------------------------------------------------------------------
class ResNetBasicBlock(nn.Module):
	expansion = 1

	def __init__(self, in_channels, out_channels):
		super(ResNetBasicBlock, self).__init__()
		downsample = nn.Sequential(OrderedDict([
			("conv", nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)),
			("bn", nn.BatchNorm2d(out_channels))
		]))
		self.block = BasicBlock(
			in_channels,
			int(out_channels/BasicBlock.expansion),
			downsample=downsample,
		)

	def forward(self, x):
		x = self.block(x)
		return x


class ResNetBottleneckBlock(nn.Module):
	expansion = 4
	
	def __init__(self, in_channels, out_channels):
		super(ResNetBottleneckBlock, self).__init__()
		downsample = nn.Sequential(OrderedDict([
			("conv", nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)),
			("bn", nn.BatchNorm2d(out_channels))
		]))
		self.block = Bottleneck(
			in_channels,
			int(out_channels/Bottleneck.expansion),
			downsample=downsample,
		)

	def forward(self, x):
		x = self.block(x)
		return x


#------------------------------------------------------------------------------
#  ResNet
#------------------------------------------------------------------------------
class ResNet(BaseResNet, BaseBackboneWrapper):
	def __init__(self, block, layers, frozen_stages=-1, norm_eval=False, **kargs):
		super(ResNet, self).__init__(block=block, layers=layers, **kargs)
		self.frozen_stages = frozen_stages
		self.norm_eval = norm_eval

	def forward(self, input):
		# Stem
		x1 = self.conv1(input)
		x1 = self.bn1(x1)
		x1 = self.relu(x1)
		# Stage1
		x2 = self.maxpool(x1)
		x2 = self.layer1(x2)
		# Stage2
		x3 = self.layer2(x2)
		# Stage3
		x4 = self.layer3(x3)
		# Stage4
		x5 = self.layer4(x4)
		# Output
		return x1, x2, x3, x4, x5

	def init_from_imagenet(self, archname):
		load_pretrained(self, default_cfgs[archname], self.num_classes)

	def _freeze_stages(self):
		# Freeze stem
		if self.frozen_stages>=0:
			self.bn1.eval()
			for module in [self.conv1, self.bn1]:
				for param in module.parameters():
					param.requires_grad = False

		# Chosen subsequent blocks are also frozen
		for stage_idx in range(1, self.frozen_stages+1):
			for module in getattr(self, "layer%d"%(stage_idx)):
				module.eval()
				for param in module.parameters():
					param.requires_grad = False


#------------------------------------------------------------------------------
#  Versions of ResNet
#------------------------------------------------------------------------------
def resnet18(pretrained=False, num_classes=1000, in_chans=3, **kwargs):
	"""Constructs a ResNet-18 model.
	"""
	default_cfg = default_cfgs['resnet18']
	model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, in_chans=in_chans, **kwargs)
	model.default_cfg = default_cfg
	if pretrained:
		load_pretrained(model, default_cfg, num_classes, in_chans)
	return model

def resnet34(pretrained=False, num_classes=1000, in_chans=3, **kwargs):
	"""Constructs a ResNet-34 model.
	"""
	default_cfg = default_cfgs['resnet34']
	model = ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, in_chans=in_chans, **kwargs)
	model.default_cfg = default_cfg
	if pretrained:
		load_pretrained(model, default_cfg, num_classes, in_chans)
	return model

def resnet26(pretrained=False, num_classes=1000, in_chans=3, **kwargs):
	"""Constructs a ResNet-26 model.
	"""
	default_cfg = default_cfgs['resnet26']
	model = ResNet(Bottleneck, [2, 2, 2, 2], num_classes=num_classes, in_chans=in_chans, **kwargs)
	model.default_cfg = default_cfg
	if pretrained:
		load_pretrained(model, default_cfg, num_classes, in_chans)
	return model

def resnet26d(pretrained=False, num_classes=1000, in_chans=3, **kwargs):
	"""Constructs a ResNet-26 v1d model.
	This is technically a 28 layer ResNet, sticking with 'd' modifier from Gluon for now.
	"""
	default_cfg = default_cfgs['resnet26d']
	model = ResNet(
		Bottleneck, [2, 2, 2, 2], stem_width=32, deep_stem=True, avg_down=True,
		num_classes=num_classes, in_chans=in_chans, **kwargs)
	model.default_cfg = default_cfg
	if pretrained:
		load_pretrained(model, default_cfg, num_classes, in_chans)
	return model

def resnet50(pretrained=False, num_classes=1000, in_chans=3, **kwargs):
	"""Constructs a ResNet-50 model.
	"""
	default_cfg = default_cfgs['resnet50']
	model = ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, in_chans=in_chans, **kwargs)
	model.default_cfg = default_cfg
	if pretrained:
		load_pretrained(model, default_cfg, num_classes, in_chans)
	return model

def resnet101(pretrained=False, num_classes=1000, in_chans=3, **kwargs):
	"""Constructs a ResNet-101 model.
	"""
	default_cfg = default_cfgs['resnet101']
	model = ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes, in_chans=in_chans, **kwargs)
	model.default_cfg = default_cfg
	if pretrained:
		load_pretrained(model, default_cfg, num_classes, in_chans)
	return model

def resnet152(pretrained=False, num_classes=1000, in_chans=3, **kwargs):
	"""Constructs a ResNet-152 model.
	"""
	default_cfg = default_cfgs['resnet152']
	model = ResNet(Bottleneck, [3, 8, 36, 3], num_classes=num_classes, in_chans=in_chans, **kwargs)
	model.default_cfg = default_cfg
	if pretrained:
		load_pretrained(model, default_cfg, num_classes, in_chans)
	return model

def tv_resnet34(pretrained=False, num_classes=1000, in_chans=3, **kwargs):
	"""Constructs a ResNet-34 model with original Torchvision weights.
	"""
	model = ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, in_chans=in_chans, **kwargs)
	model.default_cfg = default_cfgs['tv_resnet34']
	if pretrained:
		load_pretrained(model, model.default_cfg, num_classes, in_chans)
	return model

def tv_resnet50(pretrained=False, num_classes=1000, in_chans=3, **kwargs):
	"""Constructs a ResNet-50 model with original Torchvision weights.
	"""
	model = ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, in_chans=in_chans, **kwargs)
	model.default_cfg = default_cfgs['tv_resnet50']
	if pretrained:
		load_pretrained(model, model.default_cfg, num_classes, in_chans)
	return model

def wide_resnet50_2(pretrained=False, num_classes=1000, in_chans=3, **kwargs):
	"""Constructs a Wide ResNet-50-2 model.
	The model is the same as ResNet except for the bottleneck number of channels
	which is twice larger in every block. The number of channels in outer 1x1
	convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
	channels, and in Wide ResNet-50-2 has 2048-1024-2048.
	"""
	model = ResNet(
		Bottleneck, [3, 4, 6, 3], base_width=128,
		num_classes=num_classes, in_chans=in_chans, **kwargs)
	model.default_cfg = default_cfgs['wide_resnet50_2']
	if pretrained:
		load_pretrained(model, model.default_cfg, num_classes, in_chans)
	return model

def wide_resnet101_2(pretrained=False, num_classes=1000, in_chans=3, **kwargs):
	"""Constructs a Wide ResNet-101-2 model.
	The model is the same as ResNet except for the bottleneck number of channels
	which is twice larger in every block. The number of channels in outer 1x1
	convolutions is the same.
	"""
	model = ResNet(
		Bottleneck, [3, 4, 23, 3], base_width=128,
		num_classes=num_classes, in_chans=in_chans, **kwargs)
	model.default_cfg = default_cfgs['wide_resnet101_2']
	if pretrained:
		load_pretrained(model, model.default_cfg, num_classes, in_chans)
	return model

def resnext50_32x4d(pretrained=False, num_classes=1000, in_chans=3, **kwargs):
	"""Constructs a ResNeXt50-32x4d model.
	"""
	default_cfg = default_cfgs['resnext50_32x4d']
	model = ResNet(
		Bottleneck, [3, 4, 6, 3], cardinality=32, base_width=4,
		num_classes=num_classes, in_chans=in_chans, **kwargs)
	model.default_cfg = default_cfg
	if pretrained:
		load_pretrained(model, default_cfg, num_classes, in_chans)
	return model

def resnext50d_32x4d(pretrained=False, num_classes=1000, in_chans=3, **kwargs):
	"""Constructs a ResNeXt50d-32x4d model. ResNext50 w/ deep stem & avg pool downsample
	"""
	default_cfg = default_cfgs['resnext50d_32x4d']
	model = ResNet(
		Bottleneck, [3, 4, 6, 3], cardinality=32, base_width=4,
		stem_width=32, deep_stem=True, avg_down=True,
		num_classes=num_classes, in_chans=in_chans, **kwargs)
	model.default_cfg = default_cfg
	if pretrained:
		load_pretrained(model, default_cfg, num_classes, in_chans)
	return model

def resnext101_32x4d(pretrained=False, num_classes=1000, in_chans=3, **kwargs):
	"""Constructs a ResNeXt-101 32x4d model.
	"""
	default_cfg = default_cfgs['resnext101_32x4d']
	model = ResNet(
		Bottleneck, [3, 4, 23, 3], cardinality=32, base_width=4,
		num_classes=num_classes, in_chans=in_chans, **kwargs)
	model.default_cfg = default_cfg
	if pretrained:
		load_pretrained(model, default_cfg, num_classes, in_chans)
	return model

def resnext101_32x8d(pretrained=False, num_classes=1000, in_chans=3, **kwargs):
	"""Constructs a ResNeXt-101 32x8d model.
	"""
	default_cfg = default_cfgs['resnext101_32x8d']
	model = ResNet(
		Bottleneck, [3, 4, 23, 3], cardinality=32, base_width=8,
		num_classes=num_classes, in_chans=in_chans, **kwargs)
	model.default_cfg = default_cfg
	if pretrained:
		load_pretrained(model, default_cfg, num_classes, in_chans)
	return model

def resnext101_64x4d(pretrained=False, num_classes=1000, in_chans=3, **kwargs):
	"""Constructs a ResNeXt101-64x4d model.
	"""
	default_cfg = default_cfgs['resnext101_32x4d']
	model = ResNet(
		Bottleneck, [3, 4, 23, 3], cardinality=64, base_width=4,
		num_classes=num_classes, in_chans=in_chans, **kwargs)
	model.default_cfg = default_cfg
	if pretrained:
		load_pretrained(model, default_cfg, num_classes, in_chans)
	return model

def tv_resnext50_32x4d(pretrained=False, num_classes=1000, in_chans=3, **kwargs):
	"""Constructs a ResNeXt50-32x4d model with original Torchvision weights.
	"""
	default_cfg = default_cfgs['tv_resnext50_32x4d']
	model = ResNet(
		Bottleneck, [3, 4, 6, 3], cardinality=32, base_width=4,
		num_classes=num_classes, in_chans=in_chans, **kwargs)
	model.default_cfg = default_cfg
	if pretrained:
		load_pretrained(model, default_cfg, num_classes, in_chans)
	return model

def ig_resnext101_32x8d(pretrained=False, num_classes=1000, in_chans=3, **kwargs):
	"""Constructs a ResNeXt-101 32x8 model pre-trained on weakly-supervised data
	and finetuned on ImageNet from Figure 5 in
	`"Exploring the Limits of Weakly Supervised Pretraining" <https://arxiv.org/abs/1805.00932>`_
	Weights from https://pytorch.org/hub/facebookresearch_WSL-Images_resnext/
	Args:
		pretrained (bool): load pretrained weights
		num_classes (int): number of classes for classifier (default: 1000 for pretrained)
		in_chans (int): number of input planes (default: 3 for pretrained / color)
	"""
	default_cfg = default_cfgs['ig_resnext101_32x8d']
	model = ResNet(Bottleneck, [3, 4, 23, 3], cardinality=32, base_width=8, **kwargs)
	model.default_cfg = default_cfg
	if pretrained:
		load_pretrained(model, default_cfg, num_classes, in_chans)
	return model

def ig_resnext101_32x16d(pretrained=False, num_classes=1000, in_chans=3, **kwargs):
	"""Constructs a ResNeXt-101 32x16 model pre-trained on weakly-supervised data
	and finetuned on ImageNet from Figure 5 in
	`"Exploring the Limits of Weakly Supervised Pretraining" <https://arxiv.org/abs/1805.00932>`_
	Weights from https://pytorch.org/hub/facebookresearch_WSL-Images_resnext/
	Args:
		pretrained (bool): load pretrained weights
		num_classes (int): number of classes for classifier (default: 1000 for pretrained)
		in_chans (int): number of input planes (default: 3 for pretrained / color)
	"""
	default_cfg = default_cfgs['ig_resnext101_32x16d']
	model = ResNet(Bottleneck, [3, 4, 23, 3], cardinality=32, base_width=16, **kwargs)
	model.default_cfg = default_cfg
	if pretrained:
		load_pretrained(model, default_cfg, num_classes, in_chans)
	return model

def ig_resnext101_32x32d(pretrained=False, num_classes=1000, in_chans=3, **kwargs):
	"""Constructs a ResNeXt-101 32x32 model pre-trained on weakly-supervised data
	and finetuned on ImageNet from Figure 5 in
	`"Exploring the Limits of Weakly Supervised Pretraining" <https://arxiv.org/abs/1805.00932>`_
	Weights from https://pytorch.org/hub/facebookresearch_WSL-Images_resnext/
	Args:
		pretrained (bool): load pretrained weights
		num_classes (int): number of classes for classifier (default: 1000 for pretrained)
		in_chans (int): number of input planes (default: 3 for pretrained / color)
	"""
	default_cfg = default_cfgs['ig_resnext101_32x32d']
	model = ResNet(Bottleneck, [3, 4, 23, 3], cardinality=32, base_width=32, **kwargs)
	model.default_cfg = default_cfg
	if pretrained:
		load_pretrained(model, default_cfg, num_classes, in_chans)
	return model

def ig_resnext101_32x48d(pretrained=False, num_classes=1000, in_chans=3, **kwargs):
	"""Constructs a ResNeXt-101 32x48 model pre-trained on weakly-supervised data
	and finetuned on ImageNet from Figure 5 in
	`"Exploring the Limits of Weakly Supervised Pretraining" <https://arxiv.org/abs/1805.00932>`_
	Weights from https://pytorch.org/hub/facebookresearch_WSL-Images_resnext/
	Args:
		pretrained (bool): load pretrained weights
		num_classes (int): number of classes for classifier (default: 1000 for pretrained)
		in_chans (int): number of input planes (default: 3 for pretrained / color)
	"""
	default_cfg = default_cfgs['ig_resnext101_32x48d']
	model = ResNet(Bottleneck, [3, 4, 23, 3], cardinality=32, base_width=48, **kwargs)
	model.default_cfg = default_cfg
	if pretrained:
		load_pretrained(model, default_cfg, num_classes, in_chans)
	return model
