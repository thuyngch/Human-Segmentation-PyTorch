#------------------------------------------------------------------------------
#  Libraries
#------------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F

from base.base_model import BaseModel
from models.backbonds import ResNet, VGG


#------------------------------------------------------------------------------
#  ASSP
#------------------------------------------------------------------------------
class _ASPPModule(nn.Module):
	def __init__(self, inplanes, planes, kernel_size, padding, dilation):
		super(_ASPPModule, self).__init__()
		self.atrous_conv = nn.Conv2d(inplanes, planes, kernel_size=kernel_size, stride=1, padding=padding, dilation=dilation, bias=False)
		self.bn = nn.BatchNorm2d(planes)
		self.relu = nn.ReLU(inplace=True)

	def forward(self, x):
		x = self.atrous_conv(x)
		x = self.bn(x)
		return self.relu(x)


class ASPP(nn.Module):
	def __init__(self, output_stride, inplanes):
		super(ASPP, self).__init__()

		if output_stride == 16:
			dilations = [1, 6, 12, 18]
		elif output_stride == 8:
			dilations = [1, 12, 24, 36]

		self.aspp1 = _ASPPModule(inplanes, 256, 1, padding=0, dilation=dilations[0])
		self.aspp2 = _ASPPModule(inplanes, 256, 3, padding=dilations[1], dilation=dilations[1])
		self.aspp3 = _ASPPModule(inplanes, 256, 3, padding=dilations[2], dilation=dilations[2])
		self.aspp4 = _ASPPModule(inplanes, 256, 3, padding=dilations[3], dilation=dilations[3])

		self.global_avg_pool = nn.Sequential(
			nn.AdaptiveAvgPool2d((1, 1)),
			nn.Conv2d(inplanes, 256, 1, stride=1, bias=False),
			nn.BatchNorm2d(256),
			nn.ReLU(inplace=True),
		)
		self.conv1 = nn.Conv2d(1280, 256, 1, bias=False)
		self.bn1 = nn.BatchNorm2d(256)
		self.relu = nn.ReLU(inplace=True)
		self.dropout = nn.Dropout(0.5)

	def forward(self, x):
		x1 = self.aspp1(x)
		x2 = self.aspp2(x)
		x3 = self.aspp3(x)
		x4 = self.aspp4(x)
		x5 = self.global_avg_pool(x)
		x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)
		x = torch.cat((x1, x2, x3, x4, x5), dim=1)

		x = self.conv1(x)
		x = self.bn1(x)
		x = self.relu(x)
		return self.dropout(x)


#------------------------------------------------------------------------------
#  Decoder
#------------------------------------------------------------------------------
class Decoder(nn.Module):
	def __init__(self, num_classes, low_level_inplanes):
		super(Decoder, self).__init__()

		self.conv1 = nn.Conv2d(low_level_inplanes, 48, 1, bias=False)
		self.bn1 = nn.BatchNorm2d(48)
		self.relu = nn.ReLU(inplace=True)
		self.last_conv = nn.Sequential(
			nn.Conv2d(304, 256, kernel_size=3, stride=1, padding=1, bias=False),
			nn.BatchNorm2d(256),
			nn.ReLU(inplace=True),
			nn.Dropout(0.5),
			nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
			nn.BatchNorm2d(256),
			nn.ReLU(inplace=True),
			nn.Dropout(0.1),
			nn.Conv2d(256, num_classes, kernel_size=1, stride=1),
		)

	def forward(self, x, low_level_feat):
		low_level_feat = self.conv1(low_level_feat)
		low_level_feat = self.bn1(low_level_feat)
		low_level_feat = self.relu(low_level_feat)

		x = F.interpolate(x, size=low_level_feat.size()[2:], mode='bilinear', align_corners=True)
		x = torch.cat((x, low_level_feat), dim=1)
		x = self.last_conv(x)
		return x


#------------------------------------------------------------------------------
#  DeepLabV3Plus
#------------------------------------------------------------------------------
class DeepLab(BaseModel):
	def __init__(self, backbone='resnet50', output_stride=16, num_classes=2, freeze_bn=False, pretrained_backbone=None):
		super(DeepLab, self).__init__()
		if backbone=='resnet18':
			self.backbone = ResNet.resnet18(output_stride=output_stride, num_classes=None)
			self.aspp = ASPP(output_stride, inplanes=512)
			self.decoder = Decoder(num_classes, low_level_inplanes=64)
			self.low_feat_names = 'layer1'
		elif backbone=='resnet34':
			self.backbone = ResNet.resnet34(output_stride=output_stride, num_classes=None)
			self.aspp = ASPP(output_stride, inplanes=512)
			self.decoder = Decoder(num_classes, low_level_inplanes=64)
			self.low_feat_names = 'layer1'
		elif backbone=='resnet50':
			self.backbone = ResNet.resnet50(output_stride=output_stride, num_classes=None)
			self.aspp = ASPP(output_stride, inplanes=2048)
			self.decoder = Decoder(num_classes, low_level_inplanes=256)
			self.low_feat_names = 'layer1'
		elif backbone=='resnet101':
			self.backbone = ResNet.resnet101(output_stride=output_stride, num_classes=None)
			self.aspp = ASPP(output_stride, inplanes=2048)
			self.decoder = Decoder(num_classes, low_level_inplanes=256)
			self.low_feat_names = 'layer1'
		elif backbone=='vgg16':
			self.backbone = VGG.vgg16_bn(output_stride=output_stride)
			self.aspp = ASPP(output_stride, inplanes=512)
			self.decoder = Decoder(num_classes, low_level_inplanes=256)
			self.low_feat_names = 'layer3'
		else:
			raise NotImplementedError

		self._init_weights()
		if pretrained_backbone is not None:
			self.backbone._load_pretrained_model(pretrained_backbone)
		if freeze_bn:
			self._freeze_bn()


	def forward(self, input):
		x, low_feat = self.backbone(input, feature_names=self.low_feat_names)
		x = self.aspp(x)
		x = self.decoder(x, low_feat)
		x = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)
		return x


	def _freeze_bn(self):
		for m in self.modules():
			if isinstance(m, nn.BatchNorm2d):
				m.eval()


	def _init_weights(self):
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
			elif isinstance(m, nn.BatchNorm2d):
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 0)