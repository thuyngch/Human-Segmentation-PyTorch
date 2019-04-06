#------------------------------------------------------------------------------
#   Libraries
#------------------------------------------------------------------------------
import torch
import torch.nn as nn
from base import BaseBackbone


#------------------------------------------------------------------------------
#   Util functions
#------------------------------------------------------------------------------
def conv3x3(in_planes, out_planes, stride=1, dilation=1):
	return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=dilation, dilation=dilation, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
	return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


#------------------------------------------------------------------------------
#   Class of Basic block
#------------------------------------------------------------------------------
class BasicBlock(nn.Module):
	expansion = 1

	def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1):
		super(BasicBlock, self).__init__()
		self.conv1 = conv3x3(inplanes, planes, stride, dilation=dilation)
		self.bn1 = nn.BatchNorm2d(planes)
		self.relu = nn.ReLU(inplace=True)

		self.conv2 = conv3x3(planes, planes, dilation=dilation)
		self.bn2 = nn.BatchNorm2d(planes)

		self.downsample = downsample
		self.stride = stride


	def forward(self, x):
		residual = x

		out = self.conv1(x)
		out = self.bn1(out)
		out = self.relu(out)

		out = self.conv2(out)
		out = self.bn2(out)

		if self.downsample is not None:
			residual = self.downsample(x)

		out += residual
		out = self.relu(out)

		return out


#------------------------------------------------------------------------------
#   Class of Residual bottleneck
#------------------------------------------------------------------------------
class Bottleneck(nn.Module):
	expansion = 4

	def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1):
		super(Bottleneck, self).__init__()
		self.conv1 = conv1x1(inplanes, planes)
		self.bn1 = nn.BatchNorm2d(planes)

		self.conv2 = conv3x3(planes, planes, stride, dilation=dilation)
		self.bn2 = nn.BatchNorm2d(planes)

		self.conv3 = conv1x1(planes, planes * self.expansion)
		self.bn3 = nn.BatchNorm2d(planes * self.expansion)
		
		self.relu = nn.ReLU(inplace=True)
		self.downsample = downsample
		self.stride = stride


	def forward(self, x):
		residual = x

		out = self.conv1(x)
		out = self.bn1(out)
		out = self.relu(out)

		out = self.conv2(out)
		out = self.bn2(out)
		out = self.relu(out)

		out = self.conv3(out)
		out = self.bn3(out)

		if self.downsample is not None:
			residual = self.downsample(x)

		out += residual
		out = self.relu(out)

		return out


#------------------------------------------------------------------------------
#   Class of ResNet
#------------------------------------------------------------------------------
class ResNet(BaseBackbone):
	basic_inplanes = 64

	def __init__(self, block, layers, output_stride=32, num_classes=1000):
		super(ResNet, self).__init__()
		self.inplanes = self.basic_inplanes
		self.output_stride = output_stride
		self.num_classes = num_classes

		if output_stride==8:
			strides   = [1, 2, 1, 1]
			dilations = [1, 1, 2, 4]
		elif output_stride==16:
			strides   = [1, 2, 2, 1]
			dilations = [1, 1, 1, 2]
		elif output_stride==32:
			strides   = [1, 2, 2, 2]
			dilations = [1, 1, 1, 1]
		else:
			raise NotImplementedError

		self.conv1 = nn.Conv2d(3, self.basic_inplanes, kernel_size=7, stride=2, padding=3, bias=False)
		self.bn1 = nn.BatchNorm2d(self.basic_inplanes)
		self.relu = nn.ReLU(inplace=True)
		self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

		self.layer1 = self._make_layer(block, 1*self.basic_inplanes, num_layers=layers[0], stride=strides[0], dilation=dilations[0])
		self.layer2 = self._make_layer(block, 2*self.basic_inplanes, num_layers=layers[1], stride=strides[1], dilation=dilations[1])
		self.layer3 = self._make_layer(block, 4*self.basic_inplanes, num_layers=layers[2], stride=strides[2], dilation=dilations[2])
		self.layer4 = self._make_layer(block, 8*self.basic_inplanes, num_layers=layers[3], stride=strides[3], dilation=dilations[3])

		if self.num_classes is not None:
			self.fc = nn.Linear(8*self.basic_inplanes * block.expansion, num_classes)

		self.init_weights()


	def forward(self, x):
		# Stage1
		x = self.conv1(x)
		x = self.bn1(x)
		x = self.relu(x)
		# Stage2
		x = self.maxpool(x)
		x = self.layer1(x)
		# Stage3
		x = self.layer2(x)
		# Stage4
		x = self.layer3(x)
		# Stage5
		x = self.layer4(x)
		# Classification
		if self.num_classes is not None:
			x = x.mean(dim=(2,3))
			x = self.fc(x)
		# Output
		return x


	def _make_layer(self, block, planes, num_layers, stride=1, dilation=1, grids=None):
		# Downsampler
		downsample = None
		if (stride != 1) or (self.inplanes != planes * block.expansion):
			downsample = nn.Sequential(
				conv1x1(self.inplanes, planes * block.expansion, stride),
				nn.BatchNorm2d(planes * block.expansion))
		# Multi-grids
		if dilation!=1:
			dilations = [dilation*(2**layer_idx) for layer_idx in range(num_layers)]
		else:
			dilations = num_layers*[dilation]
		# Construct layers
		layers = []
		layers.append(block(self.inplanes, planes, stride, downsample, dilations[0]))
		self.inplanes = planes * block.expansion
		for i in range(1, num_layers):
			layers.append(block(self.inplanes, planes, dilation=dilations[i]))
		return nn.Sequential(*layers)


#------------------------------------------------------------------------------
#   Instances of ResNet
#------------------------------------------------------------------------------
def resnet18(pretrained=None, **kwargs):
	model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
	if pretrained is not None:
		model._load_pretrained_model(pretrained)
	return model


def resnet34(pretrained=None, **kwargs):
	model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
	if pretrained is not None:
		model._load_pretrained_model(pretrained)
	return model


def resnet50(pretrained=None, **kwargs):
	model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
	if pretrained is not None:
		model._load_pretrained_model(pretrained)
	return model


def resnet101(pretrained=None, **kwargs):
	model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
	if pretrained is not None:
		model._load_pretrained_model(pretrained)
	return model


def resnet152(pretrained=None, **kwargs):
	model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
	if pretrained is not None:
		model._load_pretrained_model(pretrained)
	return model


def get_resnet(num_layers, **kwargs):
	if num_layers==18:
		return resnet18(**kwargs)
	elif num_layers==34:
		return resnet34(**kwargs)
	elif num_layers==50:
		return resnet50(**kwargs)
	elif num_layers==101:
		return resnet101(**kwargs)
	elif num_layers==152:
		return resnet152(**kwargs)
	else:
		raise NotImplementedError