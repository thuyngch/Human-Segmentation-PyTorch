#------------------------------------------------------------------------------
#  Libraries
#------------------------------------------------------------------------------
import math, torch, json
import torch.nn as nn


#------------------------------------------------------------------------------
#  Useful functions
#------------------------------------------------------------------------------
def _make_divisible(v, divisor, min_value=None):
	if min_value is None:
		min_value = divisor
	new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
	# Make sure that round down does not go down by more than 10%.
	if new_v < 0.9 * v:
		new_v += divisor
	return new_v


def conv_bn(inp, oup, stride):
	return nn.Sequential(
		nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
		nn.BatchNorm2d(oup),
		nn.ReLU6(inplace=True)
	)


def conv_1x1_bn(inp, oup):
	return nn.Sequential(
		nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
		nn.BatchNorm2d(oup),
		nn.ReLU6(inplace=True)
	)


#------------------------------------------------------------------------------
#  Class of Inverted Residual block
#------------------------------------------------------------------------------
class InvertedResidual(nn.Module):
	def __init__(self, inp, oup, stride, expansion, dilation=1):
		super(InvertedResidual, self).__init__()
		self.stride = stride
		assert stride in [1, 2]

		hidden_dim = round(inp * expansion)
		self.use_res_connect = self.stride == 1 and inp == oup

		if expansion == 1:
			self.conv = nn.Sequential(
				# dw
				nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, dilation=dilation, bias=False),
				nn.BatchNorm2d(hidden_dim),
				nn.ReLU6(inplace=True),
				# pw-linear
				nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
				nn.BatchNorm2d(oup),
			)
		else:
			self.conv = nn.Sequential(
				# pw
				nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
				nn.BatchNorm2d(hidden_dim),
				nn.ReLU6(inplace=True),
				# dw
				nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, dilation=dilation, bias=False),
				nn.BatchNorm2d(hidden_dim),
				nn.ReLU6(inplace=True),
				# pw-linear
				nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
				nn.BatchNorm2d(oup),
			)

	def forward(self, x):
		if self.use_res_connect:
			return x + self.conv(x)
		else:
			return self.conv(x)


#------------------------------------------------------------------------------
#  Class of MobileNetV2
#------------------------------------------------------------------------------
class MobileNetV2(nn.Module):
	def __init__(self, n_classes=1000, img_layers=3, input_size=224, alpha=1., expansion=6):
		super(MobileNetV2, self).__init__()
		block = InvertedResidual
		input_channel = 32
		last_channel = 1280
		interverted_residual_setting = [
			# t, c, n, s
			[1        , 16, 1, 1],
			[expansion, 24, 2, 2],
			[expansion, 32, 3, 2],
			[expansion, 64, 4, 2],
			[expansion, 96, 3, 1],
			[expansion, 160, 3, 2],
			[expansion, 320, 1, 1],
		]

		# building first layer
		assert input_size % 32 == 0
		input_channel = _make_divisible(input_channel*alpha, 8)
		self.last_channel = _make_divisible(last_channel*alpha, 8) if alpha > 1.0 else last_channel
		self.features = [conv_bn(img_layers, input_channel, 2)]

		# building inverted residual blocks
		for t, c, n, s in interverted_residual_setting:
			output_channel = _make_divisible(int(c*alpha), 8)
			for i in range(n):
				if i == 0:
					self.features.append(block(input_channel, output_channel, s, expansion=t))
				else:
					self.features.append(block(input_channel, output_channel, 1, expansion=t))
				input_channel = output_channel

		# building last several layers
		self.features.append(conv_1x1_bn(input_channel, self.last_channel))

		# make it nn.Sequential
		self.features = nn.Sequential(*self.features)

		# building classifier
		self.classifier = nn.Sequential(
			nn.Dropout(0.2),
			nn.Linear(self.last_channel, n_classes),
		)

		self._initialize_weights()


	def forward(self, x):
		x = self.features(x)
		x = x.mean(3).mean(2)
		x = self.classifier(x)
		return x


	def _initialize_weights(self):
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
				m.weight.data.normal_(0, math.sqrt(2. / n))
				if m.bias is not None:
					m.bias.data.zero_()
			elif isinstance(m, nn.BatchNorm2d):
				m.weight.data.fill_(1)
				m.bias.data.zero_()
			elif isinstance(m, nn.Linear):
				n = m.weight.size(1)
				m.weight.data.normal_(0, 0.01)
				m.bias.data.zero_()