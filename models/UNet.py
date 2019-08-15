#------------------------------------------------------------------------------
#   Libraries
#------------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import reduce

from base import BaseModel
from models.backbonds import MobileNetV2, ResNet


#------------------------------------------------------------------------------
#   Decoder block
#------------------------------------------------------------------------------
class DecoderBlock(nn.Module):
	def __init__(self, in_channels, out_channels, block_unit):
		super(DecoderBlock, self).__init__()
		self.deconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, padding=1, stride=2)
		self.block_unit = block_unit

	def forward(self, input, shortcut):
		x = self.deconv(input)
		x = torch.cat([x, shortcut], dim=1)
		x = self.block_unit(x)
		return x


#------------------------------------------------------------------------------
#   Class of UNet
#------------------------------------------------------------------------------
class UNet(BaseModel):
	def __init__(self, backbone="mobilenetv2", num_classes=2, pretrained_backbone=None):
		super(UNet, self).__init__()
		if backbone=='mobilenetv2':
			alpha = 1.0
			expansion = 6
			self.backbone = MobileNetV2.MobileNetV2(alpha=alpha, expansion=expansion, num_classes=None)
			self._run_backbone = self._run_backbone_mobilenetv2
			# Stage 1
			channel1 = MobileNetV2._make_divisible(int(96*alpha), 8)
			block_unit = MobileNetV2.InvertedResidual(2*channel1, channel1, 1, expansion)
			self.decoder1 = DecoderBlock(self.backbone.last_channel, channel1, block_unit)
			# Stage 2
			channel2 = MobileNetV2._make_divisible(int(32*alpha), 8)
			block_unit = MobileNetV2.InvertedResidual(2*channel2, channel2, 1, expansion)
			self.decoder2 = DecoderBlock(channel1, channel2, block_unit)
			# Stage 3
			channel3 = MobileNetV2._make_divisible(int(24*alpha), 8)
			block_unit = MobileNetV2.InvertedResidual(2*channel3, channel3, 1, expansion)
			self.decoder3 = DecoderBlock(channel2, channel3, block_unit)
			# Stage 4
			channel4 = MobileNetV2._make_divisible(int(16*alpha), 8)
			block_unit = MobileNetV2.InvertedResidual(2*channel4, channel4, 1, expansion)
			self.decoder4 = DecoderBlock(channel3, channel4, block_unit)

		elif 'resnet' in backbone:
			if backbone=='resnet18':
				n_layers = 18
			elif backbone=='resnet34':
				n_layers = 34
			elif backbone=='resnet50':
				n_layers = 50
			elif backbone=='resnet101':
				n_layers = 101
			else:
				raise NotImplementedError
			filters = 64
			self.backbone = ResNet.get_resnet(n_layers, num_classes=None)
			self._run_backbone = self._run_backbone_resnet
			block = ResNet.BasicBlock if (n_layers==18 or n_layers==34) else ResNet.Bottleneck
			# Stage 1
			last_channel = 8*filters if (n_layers==18 or n_layers==34) else 32*filters
			channel1 = 4*filters if (n_layers==18 or n_layers==34) else 16*filters
			downsample = nn.Sequential(ResNet.conv1x1(2*channel1, channel1), nn.BatchNorm2d(channel1))
			block_unit = block(2*channel1, int(channel1/block.expansion), 1, downsample)
			self.decoder1 = DecoderBlock(last_channel, channel1, block_unit)
			# Stage 2
			channel2 = 2*filters if (n_layers==18 or n_layers==34) else 8*filters
			downsample = nn.Sequential(ResNet.conv1x1(2*channel2, channel2), nn.BatchNorm2d(channel2))
			block_unit = block(2*channel2, int(channel2/block.expansion), 1, downsample)
			self.decoder2 = DecoderBlock(channel1, channel2, block_unit)
			# Stage 3
			channel3 = filters if (n_layers==18 or n_layers==34) else 4*filters
			downsample = nn.Sequential(ResNet.conv1x1(2*channel3, channel3), nn.BatchNorm2d(channel3))
			block_unit = block(2*channel3, int(channel3/block.expansion), 1, downsample)
			self.decoder3 = DecoderBlock(channel2, channel3, block_unit)
			# Stage 4
			channel4 = filters
			downsample = nn.Sequential(ResNet.conv1x1(2*channel4, channel4), nn.BatchNorm2d(channel4))
			block_unit = block(2*channel4, int(channel4/block.expansion), 1, downsample)
			self.decoder4 = DecoderBlock(channel3, channel4, block_unit)

		else:
			raise NotImplementedError

		self.conv_last = nn.Sequential(
			nn.Conv2d(channel4, 3, kernel_size=3, padding=1),
			nn.Conv2d(3, num_classes, kernel_size=3, padding=1),
		)

		# Initialize
		self._init_weights()
		if pretrained_backbone is not None:
			self.backbone._load_pretrained_model(pretrained_backbone)


	def forward(self, input):
		x1, x2, x3, x4, x5 = self._run_backbone(input)
		x = self.decoder1(x5, x4)
		x = self.decoder2(x, x3)
		x = self.decoder3(x, x2)
		x = self.decoder4(x, x1)
		x = self.conv_last(x)
		x = F.interpolate(x, size=input.shape[-2:], mode='bilinear', align_corners=True)
		return x


	def _run_backbone_mobilenetv2(self, input):
		x = input
		# Stage1
		x = reduce(lambda x, n: self.backbone.features[n](x), list(range(0,2)), x)
		x1 = x
		# Stage2
		x = reduce(lambda x, n: self.backbone.features[n](x), list(range(2,4)), x)
		x2 = x
		# Stage3
		x = reduce(lambda x, n: self.backbone.features[n](x), list(range(4,7)), x)
		x3 = x
		# Stage4
		x = reduce(lambda x, n: self.backbone.features[n](x), list(range(7,14)), x)
		x4 = x
		# Stage5
		x5 = reduce(lambda x, n: self.backbone.features[n](x), list(range(14,19)), x)
		return x1, x2, x3, x4, x5


	def _run_backbone_resnet(self, input):
		# Stage1
		x1 = self.backbone.conv1(input)
		x1 = self.backbone.bn1(x1)
		x1 = self.backbone.relu(x1)
		# Stage2
		x2 = self.backbone.maxpool(x1)
		x2 = self.backbone.layer1(x2)
		# Stage3
		x3 = self.backbone.layer2(x2)
		# Stage4
		x4 = self.backbone.layer3(x3)
		# Stage5
		x5 = self.backbone.layer4(x4)
		return x1, x2, x3, x4, x5


	def _init_weights(self):
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
				if m.bias is not None:
					m.bias.data.zero_()
			elif isinstance(m, nn.BatchNorm2d):
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 0)
