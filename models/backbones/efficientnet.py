#------------------------------------------------------------------------------
#  Libraries
#------------------------------------------------------------------------------
from base import BaseBackbone
from timm.models.gen_efficientnet import (
	InvertedResidual, default_cfgs, load_pretrained, _round_channels,
	_decode_arch_def, _resolve_bn_args, swish,
)
from timm.models.gen_efficientnet import GenEfficientNet as BaseEfficientNet

import torch
import torch.nn as nn
from torch.nn import functional as F
from collections import OrderedDict


#------------------------------------------------------------------------------
#   EfficientNetBlock
#------------------------------------------------------------------------------
class EfficientNetBlock(nn.Module):
	def __init__(self, in_channels, out_channels, num_blocks=1, kernel_size=3):
		super(EfficientNetBlock, self).__init__()
		if num_blocks==2:
			self.block = nn.Sequential(
				InvertedResidual(
					in_channels, in_channels, dw_kernel_size=kernel_size,
					act_fn=swish, exp_ratio=6.0, se_ratio=0.25),
				InvertedResidual(
					in_channels, out_channels, dw_kernel_size=kernel_size,
					act_fn=swish, exp_ratio=6.0, se_ratio=0.25))
		else:
			self.block = nn.Sequential(
				InvertedResidual(
					in_channels, out_channels, dw_kernel_size=kernel_size,
					act_fn=swish, exp_ratio=6.0, se_ratio=0.25))

	def forward(self, x):
		x = self.block(x)
		return x


#------------------------------------------------------------------------------
#  EfficientNet
#------------------------------------------------------------------------------
class EfficientNet(BaseEfficientNet, BaseBackbone):
	stage_indices = {
		"tf_efficientnet_b0": [0, 1, 2, 4, 6],
		"tf_efficientnet_b1": [0, 1, 2, 4, 6],
		"tf_efficientnet_b2": [0, 1, 2, 4, 6],
		"tf_efficientnet_b3": [0, 1, 2, 4, 6],
		"tf_efficientnet_b4": [0, 1, 2, 4, 6],
		"tf_efficientnet_b5": [0, 1, 2, 4, 6],
		"tf_efficientnet_b6": [0, 1, 2, 4, 6],
		"tf_efficientnet_b7": [0, 1, 2, 4, 6],
	}
	stage_features = {
		"tf_efficientnet_b0": [16, 24, 40, 112, 320],
		"tf_efficientnet_b1": [16, 24, 40, 112, 320],
		"tf_efficientnet_b2": [16, 24, 48, 120, 352],
		"tf_efficientnet_b3": [24, 32, 48, 136, 384],
		"tf_efficientnet_b4": [24, 32, 56, 160, 448],
		"tf_efficientnet_b5": [24, 40, 64, 176, 512],
		"tf_efficientnet_b6": [32, 40, 72, 200, 576],
		"tf_efficientnet_b7": [32, 48, 80, 224, 640],
	}
	def __init__(self, block_args, model_name, frozen_stages=-1, norm_eval=False, **kargs):
		super(EfficientNet, self).__init__(block_args=block_args, **kargs)
		self.frozen_stages = frozen_stages
		self.norm_eval = norm_eval
		self.model_name = model_name

	def forward(self, input):
		# Stem
		x = self.conv_stem(input)
		x = self.bn1(x)
		x = self.act_fn(x, inplace=True)

		# Blocks
		outs = []
		for idx, block in enumerate(self.blocks):
			x = block(x)
			if idx in self.stage_indices[self.model_name]:
				outs.append(x)
		return tuple(outs)

	def init_from_imagenet(self, archname):
		print("[%s] Load pretrained weights from ImageNet" % (self.__class__.__name__))
		load_pretrained(self, default_cfgs["tf_%s"%(archname)], self.num_classes)

	def _freeze_stages(self):
		# Freeze stem
		if self.frozen_stages>=0:
			self.bn1.eval()
			for module in [self.conv_stem, self.bn1]:
				for param in module.parameters():
					param.requires_grad = False

		# Chosen subsequent blocks are also frozen gradient
		frozen_stages = list(range(1, self.frozen_stages+1))
		for idx, block in enumerate(self.blocks):
			if idx <= self.stage_indices[self.model_name][0]:
				stage = 1
			elif self.stage_indices[self.model_name][0] < idx <= self.stage_indices[self.model_name][1]:
				stage = 2
			elif self.stage_indices[self.model_name][1] < idx <= self.stage_indices[self.model_name][2]:
				stage = 3
			elif self.stage_indices[self.model_name][2] < idx <= self.stage_indices[self.model_name][3]:
				stage = 4
			if stage in frozen_stages:
				block.eval()
				for param in block.parameters():
					param.requires_grad = False
			else:
				break


#------------------------------------------------------------------------------
#  Versions of EfficientNet
#------------------------------------------------------------------------------
def _gen_efficientnet(model_name, channel_multiplier=1.0, depth_multiplier=1.0, num_classes=1000, **kwargs):
	"""
	EfficientNet params
	name: (channel_multiplier, depth_multiplier, resolution, dropout_rate)
	'efficientnet-b0': (1.0, 1.0, 224, 0.2),
	'efficientnet-b1': (1.0, 1.1, 240, 0.2),
	'efficientnet-b2': (1.1, 1.2, 260, 0.3),
	'efficientnet-b3': (1.2, 1.4, 300, 0.3),
	'efficientnet-b4': (1.4, 1.8, 380, 0.4),
	'efficientnet-b5': (1.6, 2.2, 456, 0.4),
	'efficientnet-b6': (1.8, 2.6, 528, 0.5),
	'efficientnet-b7': (2.0, 3.1, 600, 0.5),
	Args:
	  channel_multiplier: multiplier to number of channels per layer
	  depth_multiplier: multiplier to number of repeats per stage
	"""
	arch_def = [
		['ds_r1_k3_s1_e1_c16_se0.25'],
		['ir_r2_k3_s2_e6_c24_se0.25'],
		['ir_r2_k5_s2_e6_c40_se0.25'],
		['ir_r3_k3_s2_e6_c80_se0.25'],
		['ir_r3_k5_s1_e6_c112_se0.25'],
		['ir_r4_k5_s2_e6_c192_se0.25'],
		['ir_r1_k3_s1_e6_c320_se0.25'],
	]
	# NOTE: other models in the family didn't scale the feature count
	num_features = _round_channels(1280, channel_multiplier, 8, None)
	model = EfficientNet(
		_decode_arch_def(arch_def, depth_multiplier),
		model_name=model_name,
		num_classes=num_classes,
		stem_size=32,
		channel_multiplier=channel_multiplier,
		num_features=num_features,
		bn_args=_resolve_bn_args(kwargs),
		act_fn=swish,
		**kwargs
	)
	return model

def efficientnet_b0(pretrained=False, num_classes=1000, in_chans=3, **kwargs):
	""" EfficientNet-B0 """
	model_name = "tf_efficientnet_b0"
	default_cfg = default_cfgs[model_name]
	# NOTE for train, drop_rate should be 0.2
	#kwargs['drop_connect_rate'] = 0.2  # set when training, TODO add as cmd arg
	model = _gen_efficientnet(
		model_name=model_name, channel_multiplier=1.0, depth_multiplier=1.0,
		num_classes=num_classes, in_chans=in_chans, **kwargs)
	model.default_cfg = default_cfg
	if pretrained:
		load_pretrained(model, default_cfgs[model_name], num_classes)
	return model

def efficientnet_b1(pretrained=False, num_classes=1000, in_chans=3, **kwargs):
	""" EfficientNet-B1 """
	model_name = "tf_efficientnet_b1"
	default_cfg = default_cfgs[model_name]
	# NOTE for train, drop_rate should be 0.2
	#kwargs['drop_connect_rate'] = 0.2  # set when training, TODO add as cmd arg
	model = _gen_efficientnet(
		model_name=model_name, channel_multiplier=1.0, depth_multiplier=1.1,
		num_classes=num_classes, in_chans=in_chans, **kwargs)
	model.default_cfg = default_cfg
	if pretrained:
		load_pretrained(model, default_cfgs[model_name], num_classes)
	return model

def efficientnet_b2(pretrained=False, num_classes=1000, in_chans=3, **kwargs):
	""" EfficientNet-B2 """
	model_name = "tf_efficientnet_b2"
	default_cfg = default_cfgs[model_name]
	# NOTE for train, drop_rate should be 0.3
	#kwargs['drop_connect_rate'] = 0.2  # set when training, TODO add as cmd arg
	model = _gen_efficientnet(
		model_name=model_name, channel_multiplier=1.1, depth_multiplier=1.2,
		num_classes=num_classes, in_chans=in_chans, **kwargs)
	model.default_cfg = default_cfg
	if pretrained:
		load_pretrained(model, default_cfgs[model_name], num_classes)
	return model

def efficientnet_b3(pretrained=False, num_classes=1000, in_chans=3, **kwargs):
	""" EfficientNet-B3 """
	model_name = "tf_efficientnet_b3"
	default_cfg = default_cfgs[model_name]
	# NOTE for train, drop_rate should be 0.3
	#kwargs['drop_connect_rate'] = 0.2  # set when training, TODO add as cmd arg
	model = _gen_efficientnet(
		model_name=model_name, channel_multiplier=1.2, depth_multiplier=1.4,
		num_classes=num_classes, in_chans=in_chans, **kwargs)
	model.default_cfg = default_cfg
	if pretrained:
		load_pretrained(model, default_cfgs[model_name], num_classes)
	return model

def efficientnet_b4(pretrained=False, num_classes=1000, in_chans=3, **kwargs):
	""" EfficientNet-B4 """
	model_name = "tf_efficientnet_b4"
	default_cfg = default_cfgs[model_name]
	# NOTE for train, drop_rate should be 0.4
	#kwargs['drop_connect_rate'] = 0.2  #  set when training, TODO add as cmd arg
	model = _gen_efficientnet(
		model_name=model_name, channel_multiplier=1.4, depth_multiplier=1.8,
		num_classes=num_classes, in_chans=in_chans, **kwargs)
	model.default_cfg = default_cfg
	if pretrained:
		load_pretrained(model, default_cfgs[model_name], num_classes)
	return model

def efficientnet_b5(pretrained=False, num_classes=1000, in_chans=3, **kwargs):
	""" EfficientNet-B5 """
	# NOTE for train, drop_rate should be 0.4
	#kwargs['drop_connect_rate'] = 0.2  # set when training, TODO add as cmd arg
	model_name = "tf_efficientnet_b5"
	default_cfg = default_cfgs[model_name]
	model = _gen_efficientnet(
		model_name=model_name, channel_multiplier=1.6, depth_multiplier=2.2,
		num_classes=num_classes, in_chans=in_chans, **kwargs)
	model.default_cfg = default_cfg
	if pretrained:
		load_pretrained(model, default_cfgs[model_name], num_classes)
	return model

def efficientnet_b6(pretrained=False, num_classes=1000, in_chans=3, **kwargs):
	""" EfficientNet-B6 """
	# NOTE for train, drop_rate should be 0.5
	#kwargs['drop_connect_rate'] = 0.2  # set when training, TODO add as cmd arg
	model_name = "tf_efficientnet_b6"
	default_cfg = default_cfgs[model_name]
	model = _gen_efficientnet(
		model_name=model_name, channel_multiplier=1.8, depth_multiplier=2.6,
		num_classes=num_classes, in_chans=in_chans, **kwargs)
	model.default_cfg = default_cfg
	if pretrained:
		load_pretrained(model, default_cfgs[model_name], num_classes)
	return model

def efficientnet_b7(pretrained=False, num_classes=1000, in_chans=3, **kwargs):
	""" EfficientNet-B7 """
	# NOTE for train, drop_rate should be 0.5
	#kwargs['drop_connect_rate'] = 0.2  # set when training, TODO add as cmd arg
	model_name = "tf_efficientnet_b7"
	default_cfg = default_cfgs[model_name]
	model = _gen_efficientnet(
		model_name=model_name, channel_multiplier=2.0, depth_multiplier=3.1,
		num_classes=num_classes, in_chans=in_chans, **kwargs)
	model.default_cfg = default_cfg
	if pretrained:
		load_pretrained(model, default_cfgs[model_name], num_classes)
	return model
