#------------------------------------------------------------------------------
#   Libraries
#------------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import reduce
from base import BaseModel

# Backbones
from models.backbonds.MobileNetV2 import (
	MobileNetV2,
	InvertedResidual, _make_divisible,
)
from models.backbonds.ResNet import (
	get_resnet, conv1x1,
	BasicBlock, Bottleneck,
)


#------------------------------------------------------------------------------
#   Class of UNet with different backbones
#------------------------------------------------------------------------------
class UNet(BaseModel):
	def __init__(self, img_layers=3, num_classes=2, backbone="MobileNetV2", backbone_args=None):
		super(UNet, self).__init__()
		self.img_layers = img_layers
		self.backbone_type = backbone

		# Setup backbone for both encoder and decoder
		if self.backbone_type=="MobileNetV2":
			output_channel4 = self.setup_MobileNetV2(backbone_args)
		elif self.backbone_type=="ResNet":
			output_channel4 = self.setup_ResNet(backbone_args)

		# Layers for outputing a segmentation map
		self.conv_last = nn.Sequential(
			nn.Conv2d(output_channel4, 3, kernel_size=1),
			nn.Conv2d(3, num_classes, kernel_size=1),
		)

		# Initialize weights
		self._init_weights()
		pretrained = backbone_args["pretrained"]
		if pretrained is None:
			print("Initialize network from scratch")
		else:
			self.backbone.load_state_dict(torch.load(pretrained, map_location='cpu'), strict=False)
			print("Backbone loaded from %s" % (pretrained))


	def forward(self, input):
		# Encoder
		x = input
		if self.backbone_type=="MobileNetV2":
			x1 = reduce(lambda x, n: self.backbone.features[n](x), list(range(0,2)), x)
			x2 = reduce(lambda x, n: self.backbone.features[n](x), list(range(2,4)), x1)
			x3 = reduce(lambda x, n: self.backbone.features[n](x), list(range(4,7)), x2)
			x4 = reduce(lambda x, n: self.backbone.features[n](x), list(range(7,14)), x3)
			x5 = reduce(lambda x, n: self.backbone.features[n](x), list(range(14,19)), x4)

		elif self.backbone_type=="ResNet":
			x1 = self.backbone.relu(self.backbone.bn1(self.backbone.conv1(x)))
			x2 = self.backbone.layer1(self.backbone.maxpool(x1))
			x3 = self.backbone.layer2(x2)
			x4 = self.backbone.layer3(x3)
			x5 = self.backbone.layer4(x4)

		# Decoder
		up1 = self.block1(torch.cat([x4, self.convtrans1(x5)], dim=1))
		up2 = self.block2(torch.cat([x3, self.convtrans2(up1)], dim=1))
		up3 = self.block3(torch.cat([x2, self.convtrans3(up2)], dim=1))
		up4 = self.block4(torch.cat([x1, self.convtrans4(up3)], dim=1))

		# Last layers for normal
		y = self.conv_last(up4)
		y = F.interpolate(y, scale_factor=2, mode='bilinear', align_corners=True)
		return y


	def setup_MobileNetV2(self, backbone_args):
		input_size = backbone_args["input_sz"]
		alpha = backbone_args["alpha"]
		expansion = backbone_args["expansion"]
		self.backbone = MobileNetV2(img_layers=self.img_layers, input_size=input_size, alpha=alpha, expansion=expansion)
		del self.backbone.classifier
		# Stage 1
		output_channel1 = _make_divisible(int(96*alpha), 8)
		self.convtrans1 = nn.ConvTranspose2d(self.backbone.last_channel, output_channel1, kernel_size=4, padding=1, stride=2)
		self.block1 = InvertedResidual(2*output_channel1, output_channel1, 1, expansion, dilation=1)
		# Stage 2
		output_channel2 = _make_divisible(int(32*alpha), 8)
		self.convtrans2 = nn.ConvTranspose2d(output_channel1, output_channel2, kernel_size=4, padding=1, stride=2)
		self.block2 = InvertedResidual(2*output_channel2, output_channel2, 1, expansion, dilation=1)
		# Stage 3
		output_channel3 = _make_divisible(int(24*alpha), 8)
		self.convtrans3 = nn.ConvTranspose2d(output_channel2, output_channel3, kernel_size=4, padding=1, stride=2)
		self.block3 = InvertedResidual(2*output_channel3, output_channel3, 1, expansion, dilation=1)
		# Stage 4
		output_channel4 = _make_divisible(int(16*alpha), 8)
		self.convtrans4 = nn.ConvTranspose2d(output_channel3, output_channel4, kernel_size=4, padding=1, stride=2)
		self.block4 = InvertedResidual(2*output_channel4, output_channel4, 1, expansion, dilation=1)
		return output_channel4


	def setup_ResNet(self, backbone_args):
		filters = 64
		n_layers = backbone_args["n_layers"]
		block = BasicBlock if (n_layers==18 or n_layers==34) else Bottleneck
		self.backbone = get_resnet(n_layers, img_layers=self.img_layers)
		del self.backbone.avgpool, self.backbone.fc
		# Stage 1
		last_channel = 8*filters if (n_layers==18 or n_layers==34) else 32*filters
		output_channel1 = 4*filters if (n_layers==18 or n_layers==34) else 16*filters
		self.convtrans1 = nn.ConvTranspose2d(last_channel, output_channel1, 4, padding=1, stride=2)
		downsample1 = nn.Sequential(conv1x1(2*output_channel1, output_channel1), nn.BatchNorm2d(output_channel1))
		self.block1 = block(2*output_channel1, int(output_channel1/block.expansion), 1, downsample1)
		# Stage 2
		output_channel2 = 2*filters if (n_layers==18 or n_layers==34) else 8*filters
		self.convtrans2 = nn.ConvTranspose2d(output_channel1, output_channel2, 4, padding=1, stride=2)
		downsample2 = nn.Sequential(conv1x1(2*output_channel2, output_channel2), nn.BatchNorm2d(output_channel2))
		self.block2 = block(2*output_channel2, int(output_channel2/block.expansion), 1, downsample2)
		# Stage 3
		output_channel3 = filters if (n_layers==18 or n_layers==34) else 4*filters
		self.convtrans3 = nn.ConvTranspose2d(output_channel2, output_channel3, 4, padding=1, stride=2)
		downsample3 = nn.Sequential(conv1x1(2*output_channel3, output_channel3), nn.BatchNorm2d(output_channel3))
		self.block3 = block(2*output_channel3, int(output_channel3/block.expansion), 1, downsample3)
		# Stage 4
		output_channel4 = filters
		self.convtrans4 = nn.ConvTranspose2d(output_channel3, output_channel4, 4, padding=1, stride=2)
		downsample4 = nn.Sequential(conv1x1(2*output_channel4, output_channel4), nn.BatchNorm2d(output_channel4))
		self.block4 = block(2*output_channel4, int(output_channel4/block.expansion), 1, downsample4)
		return output_channel4


	def _init_weights(self):
		for m in self.modules():
			if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
				nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
				if m.bias is not None:
					m.bias.data.zero_()
			elif isinstance(m, nn.BatchNorm2d):
				m.weight.data.fill_(1)
				m.bias.data.zero_()