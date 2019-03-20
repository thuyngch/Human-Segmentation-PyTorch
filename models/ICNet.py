#------------------------------------------------------------------------------
#  Libraries
#------------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

from base.base_model import BaseModel
from models.backbonds import ResNet


#------------------------------------------------------------------------------
#  Convolutional block
#------------------------------------------------------------------------------
class ConvBlock(nn.Module):
	def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
		super(ConvBlock, self).__init__()
		self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)
		self.bn = nn.BatchNorm2d(out_channels)

	def forward(self, input):
		x = self.conv(input)
		x = self.bn(x)
		x = F.relu(x, inplace=True)
		return x


#------------------------------------------------------------------------------
#  Pyramid Pooling Module
#------------------------------------------------------------------------------
class PyramidPoolingModule(nn.Module):
	def __init__(self, pyramids=[1,2,3,6]):
		super(PyramidPoolingModule, self).__init__()
		self.pyramids = pyramids

	def forward(self, input):
		feat = input
		height, width = input.shape[2:]
		for bin_size in self.pyramids:
			x = F.adaptive_avg_pool2d(input, output_size=bin_size)
			x = F.interpolate(x, size=(height, width), mode='bilinear', align_corners=True)
			feat  = feat + x
		return feat


#------------------------------------------------------------------------------
#  Cascade Feature Fusion
#------------------------------------------------------------------------------
class CascadeFeatFusion(nn.Module):
	def __init__(self, low_channels, high_channels, out_channels, num_classes):
		super(CascadeFeatFusion, self).__init__()
		self.conv_low = nn.Sequential(OrderedDict([
			('conv', nn.Conv2d(low_channels, out_channels, kernel_size=3, dilation=2, padding=2, bias=False)),
			('bn', nn.BatchNorm2d(out_channels))
		]))
		self.conv_high = nn.Sequential(OrderedDict([
			('conv', nn.Conv2d(high_channels, out_channels, kernel_size=1, bias=False)),
			('bn', nn.BatchNorm2d(out_channels))
		]))
		self.conv_low_cls = nn.Conv2d(out_channels, num_classes, kernel_size=1, bias=False)

	def forward(self, input_low, input_high):
		input_low = F.interpolate(input_low, size=input_high.shape[2:], mode='bilinear', align_corners=True)
		x_low = self.conv_low(input_low)
		x_high = self.conv_high(input_high)
		x = x_low + x_high
		x = F.relu(x, inplace=True)

		if self.training:
			x_low_cls = self.conv_low_cls(input_low)
			return x, x_low_cls
		else:
			return x


#------------------------------------------------------------------------------
#  ICNet
#------------------------------------------------------------------------------
class ICNet(BaseModel):
	pyramids = [1, 2, 3, 6]
	backbone_os = 8

	def __init__(self, backbone='resnet18', num_classes=2, pretrained_backbone=None):
		super(ICNet, self).__init__()
		if 'resnet' in backbone:
			if backbone=='resnet18':
				n_layers = 18
				stage5_channels = 512
			elif backbone=='resnet34':
				n_layers = 34
				stage5_channels = 512
			elif backbone=='resnet50':
				n_layers = 50
				stage5_channels = 2048
			elif backbone=='resnet101':
				n_layers = 101
				stage5_channels = 2048
			else:
				raise NotImplementedError

			# Sub1
			self.conv_sub1 = nn.Sequential(OrderedDict([
				('conv1', ConvBlock(in_channels=3, out_channels=32, kernel_size=3, stride=2, padding=1, bias=False)),
				('conv2', ConvBlock(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1, bias=False)),
				('conv3', ConvBlock(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1, bias=False))
			]))

			# Sub2 and Sub4
			self.backbone = ResNet.get_resnet(n_layers, output_stride=self.backbone_os, num_classes=None)
			self.ppm = PyramidPoolingModule(pyramids=self.pyramids)
			self.conv_sub4_reduce = ConvBlock(stage5_channels, stage5_channels//4, kernel_size=1, bias=False)

			# Cascade Feature Fusion
			self.cff_24 = CascadeFeatFusion(low_channels=stage5_channels//4, high_channels=128, out_channels=128, num_classes=num_classes)
			self.cff_12 = CascadeFeatFusion(low_channels=128, high_channels=64, out_channels=128, num_classes=num_classes)

			# Classification
			self.conv_cls = nn.Conv2d(in_channels=128, out_channels=num_classes, kernel_size=1, bias=False)

		else:
			raise NotImplementedError

		self._init_weights()
		if pretrained_backbone is not None:
			self.backbone._load_pretrained_model(pretrained_backbone)


	def forward(self, input):
		# Sub1
		x_sub1 = self.conv_sub1(input)

		# Sub2
		x_sub2 = F.interpolate(input, scale_factor=0.5, mode='bilinear', align_corners=True)
		x_sub2 = self._run_backbone_sub2(x_sub2)

		# Sub4
		x_sub4 = F.interpolate(x_sub2, scale_factor=0.5, mode='bilinear', align_corners=True)
		x_sub4 = self._run_backbone_sub4(x_sub4)
		x_sub4 = self.ppm(x_sub4)
		x_sub4 = self.conv_sub4_reduce(x_sub4)

		# Output
		if self.training:
			# Cascade Feature Fusion
			x_cff_24, x_24_cls = self.cff_24(x_sub4, x_sub2)
			x_cff_12, x_12_cls = self.cff_12(x_cff_24, x_sub1)

			# Classification
			x_cff_12 = F.interpolate(x_cff_12, scale_factor=2, mode='bilinear', align_corners=True)
			x_124_cls = self.conv_cls(x_cff_12)
			return x_124_cls, x_12_cls, x_24_cls

		else:
			# Cascade Feature Fusion
			x_cff_24 = self.cff_24(x_sub4, x_sub2)
			x_cff_12 = self.cff_12(x_cff_24, x_sub1)

			# Classification
			x_cff_12 = F.interpolate(x_cff_12, scale_factor=2, mode='bilinear', align_corners=True)
			x_124_cls = self.conv_cls(x_cff_12)
			x_124_cls = F.interpolate(x_124_cls, scale_factor=4, mode='bilinear', align_corners=True)
			return x_124_cls


	def _run_backbone_sub2(self, input):
		# Stage1
		x = self.backbone.conv1(input)
		x = self.backbone.bn1(x)
		x = self.backbone.relu(x)
		# Stage2
		x = self.backbone.maxpool(x)
		x = self.backbone.layer1(x)
		# Stage3
		x = self.backbone.layer2(x)
		return x


	def _run_backbone_sub4(self, input):
		# Stage4
		x = self.backbone.layer3(input)
		# Stage5
		x = self.backbone.layer4(x)
		return x


	def _init_weights(self):
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
				if m.bias is not None:
					m.bias.data.zero_()
			elif isinstance(m, nn.BatchNorm2d):
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 0)