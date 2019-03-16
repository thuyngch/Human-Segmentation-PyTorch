#------------------------------------------------------------------------------
#   Libraries
#------------------------------------------------------------------------------
import torch
import torch.nn as nn


#------------------------------------------------------------------------------
#   Class of VGG
#------------------------------------------------------------------------------
class VGG(nn.Module):
	def __init__(self, blocks, input_sz=224, num_classes=1000, output_stride=32):
		super(VGG, self).__init__()
		self.output_stride = output_stride
		if output_stride==8:
			strides   = [2, 2, 2, 1, 1]
			dilations = [1, 1, 1, 2, 4]
		elif output_stride==16:
			strides   = [2, 2, 2, 2, 1]
			dilations = [1, 1, 1, 1, 2]
		elif output_stride==32:
			strides   = [2, 2, 2, 2, 2]
			dilations = [1, 1, 1, 1, 1]
		else:
			raise NotImplementedError

		self.layer1 = self._build_block(in_channels=3, block=blocks[0], dilation=dilations[0])
		self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2) if strides[0]==2 else nn.Sequential()

		self.layer2 = self._build_block(in_channels=blocks[0][-1], block=blocks[1], dilation=dilations[1])
		self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2) if strides[1]==2 else nn.Sequential()

		self.layer3 = self._build_block(in_channels=blocks[1][-1], block=blocks[2], dilation=dilations[2])
		self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2) if strides[2]==2 else nn.Sequential()

		self.layer4 = self._build_block(in_channels=blocks[2][-1], block=blocks[3], dilation=dilations[3])
		self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2) if strides[3]==2 else nn.Sequential()

		self.layer5 = self._build_block(in_channels=blocks[3][-1], block=blocks[4], dilation=dilations[4])
		self.maxpool5 = nn.MaxPool2d(kernel_size=2, stride=2) if strides[4]==2 else nn.Sequential()

		if output_stride==32:
			linear_out_channels = 512 * int(input_sz/32)**2
			self.classifier = nn.Sequential(
				nn.Linear(linear_out_channels, 4096),
				nn.ReLU(True),
				nn.Dropout(),
				nn.Linear(4096, 4096),
				nn.ReLU(True),
				nn.Dropout(),
				nn.Linear(4096, num_classes),
			)

		self._init_weights()


	def forward(self, x, feature_names=None):
		low_features = {}

		x = self.layer1(x)
		x = self.maxpool1(x)

		x = self.layer2(x)
		x = self.maxpool2(x)

		x = self.layer3(x)
		low_features['layer3'] = x
		x = self.maxpool3(x)

		x = self.layer4(x)
		x = self.maxpool4(x)

		x = self.layer5(x)
		x = self.maxpool5(x)

		if self.output_stride==32:
			x = x.view(x.size(0), -1)
			x = self.classifier(x)

		if feature_names is not None:
			if type(feature_names)==str:
				return x, low_features[feature_names]
			elif type(feature_names)==list:
				return tuple([x] + [low_features[name] for name in feature_names])
		else:
			return x


	def _build_block(self, in_channels, block, dilation):
		layers = []
		for layer_idx, out_channels in enumerate(block):
			if dilation!=1:
				grid = 2**layer_idx
				dilation *= grid
			conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=dilation, dilation=dilation)
			layers += [conv2d, nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)]
			in_channels = out_channels
		return nn.Sequential(*layers)


	def _load_pretrained_model(self, pretrained_file):
		pretrain_dict = torch.load(pretrained_file, map_location='cpu')
		model_dict = {}
		state_dict = self.state_dict()
		print("[VGG] Loading pretrained model...")
		for k, v in pretrain_dict.items():
			if k in state_dict:
				model_dict[k] = v
			else:
				print(k, "is ignored")
		state_dict.update(model_dict)
		self.load_state_dict(state_dict)
	

	def _init_weights(self):
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
			elif isinstance(m, nn.BatchNorm2d):
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 0)
			elif isinstance(m, nn.Linear):
				m.weight.data.normal_(0, 0.01)
				m.bias.data.zero_()


#------------------------------------------------------------------------------
#   Instances of VGG
#------------------------------------------------------------------------------
blocks = {
	'A': [1*[64], 1*[128], 2*[256], 2*[512], 2*[512]],
	'B': [2*[64], 2*[128], 2*[256], 2*[512], 2*[512]],
	'D': [2*[64], 2*[128], 3*[256], 3*[512], 3*[512]],
	'E': [2*[64], 2*[128], 4*[256], 4*[512], 4*[512]],
}

def vgg11_bn(pretrained=None, **kwargs):
	model = VGG(blocks['A'], **kwargs)
	if pretrained:
		model._load_pretrained_model(pretrained)
	return model

def vgg13_bn(pretrained=None, **kwargs):
	model = VGG(blocks['B'], **kwargs)
	if pretrained:
		model._load_pretrained_model(pretrained)
	return model

def vgg16_bn(pretrained=None, **kwargs):
	model = VGG(blocks['D'], **kwargs)
	if pretrained:
		model._load_pretrained_model(pretrained)
	return model

def vgg19_bn(pretrained=None, **kwargs):
	model = VGG(blocks['E'], **kwargs)
	if pretrained:
		model._load_pretrained_model(pretrained)
	return model

def get_vgg(n_layers, **kwargs):
	if n_layers==11:
		return vgg11_bn(**kwargs)
	elif n_layers==13:
		return vgg13_bn(**kwargs)
	elif n_layers==16:
		return vgg16_bn(**kwargs)
	elif n_layers==19:
		return vgg19_bn(**kwargs)
	else:
		raise NotImplementedError