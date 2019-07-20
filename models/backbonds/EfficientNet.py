#------------------------------------------------------------------------------
#  Libraries
#------------------------------------------------------------------------------
import torch
import torch.nn as nn

from .EfficientNet.efficientnet import EfficientNet as BaseEfficientNet
from .EfficientNet.efficientnet import MBConvBlock, load_pretrained_weights, get_model_params, relu_fn


#------------------------------------------------------------------------------
#  EfficientNet
#------------------------------------------------------------------------------
class EfficientNet(BaseEfficientNet):
	stages = {
		"efficientnet-b0": [2, 3, 10, 15],
		"efficientnet-b1": [4, 7, 15, 22],
		"efficientnet-b2": [4, 7, 15, 22],
		"efficientnet-b3": [4, 7, 17, 25],
		"efficientnet-b4": [5, 9, 21, 31],
		"efficientnet-b5": [7, 12, 26, 38],
		"efficientnet-b6": [8, 14, 30, 44],
		"efficientnet-b7": [10, 17, 37, 54],
	}
	in_channels_dict = {
		'efficientnet-b0': [24, 40, 112, 320],
		'efficientnet-b1': [24, 40, 112, 320],
		'efficientnet-b2': [24, 48, 120, 352],
		'efficientnet-b3': [32, 48, 136, 384],
		'efficientnet-b4': [32, 56, 160, 448],
		'efficientnet-b5': [40, 64, 176, 512],
		'efficientnet-b6': [40, 72, 200, 576],
		'efficientnet-b7': [48, 80, 224, 2560],
	}
	def __init__(self, model_name, use_se=True, init_from_pretrain=True):
		# Parameters
		self.model_name = model_name
		self.use_se = use_se
		self.init_from_pretrain = init_from_pretrain

		# Initialize model
		blocks_args, global_params = get_model_params(model_name, use_se, override_params=None)
		super(EfficientNet, self).__init__(blocks_args, global_params)
		self.init_weights()
		del self._dropout, self._fc

	def forward(self, inputs):
		outs = []
		x = relu_fn(self._bn0(self._conv_stem(inputs)))
		for idx, block in enumerate(self._blocks):
			x = block(x)
			if idx in self.stages[self.model_name]:
				outs.append(x)
		return outs

	def init_weights(self):
		if self.init_from_pretrain:
			print("[%s] Load pretrained weights" % (self.__class__.__name__))
			load_pretrained_weights(self, self.model_name)

		else:
			print("[%s] Initialize weights from scratch" % (self.__class__.__name__))
			for m in self.modules():
				if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
					nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
					if m.bias is not None:
						m.bias.data.zero_()
				elif isinstance(m, nn.BatchNorm2d):
					nn.init.constant_(m.weight, 1)
					nn.init.constant_(m.bias, 0)
				elif isinstance(m, nn.Linear):
					m.weight.data.normal_(0, 0.01)
					m.bias.data.zero_()
