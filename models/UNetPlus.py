#------------------------------------------------------------------------------
#   Libraries
#------------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

from base import BaseModel
from models import backbones


#------------------------------------------------------------------------------
#   DecoderBlock
#------------------------------------------------------------------------------
class DecoderBlock(nn.Module):
	def __init__(self, in_channels, out_channels, block, use_deconv=True, squeeze=1, dropout=0.2):
		super(DecoderBlock, self).__init__()
		self.use_deconv = use_deconv

		# Deconvolution
		if self.use_deconv:
			if squeeze==1:
				self.upsampler = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
			else:
				hid_channels = int(in_channels/squeeze)
				self.upsampler = nn.Sequential(OrderedDict([
					('conv1', nn.Conv2d(in_channels, hid_channels, kernel_size=1, bias=False)),
					('bn1', nn.BatchNorm2d(hid_channels)),
					('relu1', nn.ReLU(inplace=True)),
					('dropout1', nn.Dropout2d(p=dropout)),

					('conv2', nn.ConvTranspose2d(hid_channels, hid_channels, kernel_size=4, stride=2, padding=1, bias=False)),
					('bn2', nn.BatchNorm2d(hid_channels)),
					('relu2', nn.ReLU(inplace=True)),
					('dropout2', nn.Dropout2d(p=dropout)),

					('conv3', nn.Conv2d(hid_channels, out_channels, kernel_size=1, bias=False)),
					('bn3', nn.BatchNorm2d(out_channels)),
					('relu3', nn.ReLU(inplace=True)),
					('dropout3', nn.Dropout2d(p=dropout)),
				]))
		else:
			self.upsampler = nn.Sequential(OrderedDict([
				('conv', nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)),
				('bn', nn.BatchNorm2d(out_channels)),
				('relu', nn.ReLU(inplace=True)),
			]))

		# Block
		self.block = nn.Sequential(OrderedDict([
			('bottleneck', block(2*out_channels, out_channels)),
			('dropout', nn.Dropout(p=dropout))
		]))


	def forward(self, x, shortcut):
		if not self.use_deconv:
			x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
		x = self.upsampler(x)
		x = torch.cat([x, shortcut], dim=1)
		x = self.block(x)
		return x


#------------------------------------------------------------------------------
#   UNetPlus
#------------------------------------------------------------------------------
class UNetPlus(BaseModel):
	def __init__(self, backbone='resnet18', num_classes=2, in_channels=3,
				use_deconv=False, squeeze=4, dropout=0.2, frozen_stages=-1,
				norm_eval=True, init_backbone_from_imagenet=False):

		# Instantiate
		super(UNetPlus, self).__init__()
		self.in_channels = in_channels

		# Build Backbone and Decoder
		if ('resnet' in backbone) or ('resnext' in backbone) or ('wide_resnet' in backbone):
			self.backbone = getattr(backbones, backbone)(in_chans=in_channels, frozen_stages=frozen_stages, norm_eval=norm_eval)
			inplanes = 64
			block = backbones.ResNetBasicBlock if '18' in backbone or '34' in backbone else backbones.ResNetBottleneckBlock
			expansion = block.expansion

			self.decoder = nn.Module()
			self.decoder.layer1 = DecoderBlock(8*inplanes*expansion, 4*inplanes*expansion, block, squeeze=squeeze, dropout=dropout, use_deconv=use_deconv)
			self.decoder.layer2 = DecoderBlock(4*inplanes*expansion, 2*inplanes*expansion, block, squeeze=squeeze, dropout=dropout, use_deconv=use_deconv)
			self.decoder.layer3 = DecoderBlock(2*inplanes*expansion, inplanes*expansion, block, squeeze=squeeze, dropout=dropout, use_deconv=use_deconv)
			self.decoder.layer4 = DecoderBlock(inplanes*expansion, inplanes, block, squeeze=squeeze, dropout=dropout, use_deconv=use_deconv)
			out_channels = inplanes

		elif 'efficientnet' in backbone:
			self.backbone = getattr(backbones, backbone)(in_chans=in_channels, frozen_stages=frozen_stages, norm_eval=norm_eval)
			block = backbones.EfficientNetBlock
			num_channels = self.backbone.stage_features[self.backbone.model_name]

			self.decoder = nn.Module()
			self.decoder.layer1 = DecoderBlock(num_channels[4], num_channels[3], block, squeeze=squeeze, dropout=dropout, use_deconv=use_deconv)
			self.decoder.layer2 = DecoderBlock(num_channels[3], num_channels[2], block, squeeze=squeeze, dropout=dropout, use_deconv=use_deconv)
			self.decoder.layer3 = DecoderBlock(num_channels[2], num_channels[1], block, squeeze=squeeze, dropout=dropout, use_deconv=use_deconv)
			self.decoder.layer4 = DecoderBlock(num_channels[1], num_channels[0], block, squeeze=squeeze, dropout=dropout, use_deconv=use_deconv)
			out_channels = num_channels[0]

		else:
			raise NotImplementedError

		# Build Head
		self.mask = nn.Sequential(OrderedDict([
			('conv1', nn.Conv2d(out_channels, 32, kernel_size=3, padding=1, bias=False)),
			('bn1', nn.BatchNorm2d(num_features=32)),
			('relu1', nn.ReLU(inplace=True)),
			('dropout1', nn.Dropout2d(p=dropout)),
			('conv2', nn.Conv2d(32, 16, kernel_size=3, padding=1, bias=False)),
			('bn2', nn.BatchNorm2d(num_features=16)),
			('relu2', nn.ReLU(inplace=True)),
			('dropout2', nn.Dropout2d(p=dropout)),
			('conv3', nn.Conv2d(16, num_classes, kernel_size=3, padding=1)),
		]))

		# Initialize weights
		self.init_weights()
		if init_backbone_from_imagenet:
			self.backbone.init_from_imagenet(archname=backbone)

	def forward(self, images, **kargs):
		# Encoder
		x1, x2, x3, x4, x5 = self.backbone(images)

		# Decoder
		y = self.decoder.layer1(x5, x4)
		y = self.decoder.layer2(y, x3)
		y = self.decoder.layer3(y, x2)
		y = self.decoder.layer4(y, x1)
		y = F.interpolate(y, scale_factor=2, mode='bilinear', align_corners=True)

		# Output
		return self.mask(y)
