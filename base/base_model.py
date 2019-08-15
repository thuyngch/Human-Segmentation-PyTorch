#------------------------------------------------------------------------------
#   Libraries
#------------------------------------------------------------------------------
import torch
import torch.nn as nn

import torchsummary
import os, warnings, sys
from utils import add_flops_counting_methods, flops_to_string


#------------------------------------------------------------------------------
#   BaseModel
#------------------------------------------------------------------------------
class BaseModel(nn.Module):
	def __init__(self):
		super(BaseModel, self).__init__()

	def summary(self, input_shape, batch_size=1, device='cpu', print_flops=False):
		print("[%s] Network summary..." % (self.__class__.__name__))
		torchsummary.summary(self, input_size=input_shape, batch_size=batch_size, device=device)
		if print_flops:
			input = torch.randn([1, *input_shape], dtype=torch.float)
			counter = add_flops_counting_methods(self)
			counter.eval().start_flops_count()
			counter(input)
			print('Flops:  {}'.format(flops_to_string(counter.compute_average_flops_cost())))
			print('----------------------------------------------------------------')

	def init_weights(self):
		print("[%s] Initialize weights..." % (self.__class__.__name__))
		for m in self.modules():
			if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
				nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
				if m.bias is not None:
					m.bias.data.zero_()
			elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 0)
			elif isinstance(m, nn.Linear):
				m.weight.data.normal_(0, 0.01)
				m.bias.data.zero_()

	def load_pretrained_model(self, pretrained):
		if isinstance(pretrained, str):
			print("[%s] Load pretrained model from %s" % (self.__class__.__name__, pretrained))
			pretrain_dict = torch.load(pretrained, map_location='cpu')
			if 'state_dict' in pretrain_dict:
				pretrain_dict = pretrain_dict['state_dict']
		elif isinstance(pretrained, dict):
			print("[%s] Load pretrained model" % (self.__class__.__name__))
			pretrain_dict = pretrained

		model_dict = {}
		state_dict = self.state_dict()
		for k, v in pretrain_dict.items():
			if k in state_dict:
				if state_dict[k].shape==v.shape:
					model_dict[k] = v
				else:
					print("[%s]"%(self.__class__.__name__), k, "is ignored due to not matching shape")
			else:
				print("[%s]"%(self.__class__.__name__), k, "is ignored due to not matching key")
		state_dict.update(model_dict)
		self.load_state_dict(state_dict)


#------------------------------------------------------------------------------
#   BaseBackbone
#------------------------------------------------------------------------------
class BaseBackbone(BaseModel):
	def __init__(self):
		super(BaseBackbone, self).__init__()

	def load_pretrained_model_extended(self, pretrained):
		"""
		This function is specifically designed for loading pretrain with different in_channels
		"""
		if isinstance(pretrained, str):
			print("[%s] Load pretrained model from %s" % (self.__class__.__name__, pretrained))
			pretrain_dict = torch.load(pretrained, map_location='cpu')
			if 'state_dict' in pretrain_dict:
				pretrain_dict = pretrain_dict['state_dict']
		elif isinstance(pretrained, dict):
			print("[%s] Load pretrained model" % (self.__class__.__name__))
			pretrain_dict = pretrained

		model_dict = {}
		state_dict = self.state_dict()
		for k, v in pretrain_dict.items():
			if k in state_dict:
				if state_dict[k].shape!=v.shape:
					model_dict[k] = state_dict[k]
					model_dict[k][:,:3,...] = v
				else:
					model_dict[k] = v
			else:
				print("[%s]"%(self.__class__.__name__), k, "is ignored")
		state_dict.update(model_dict)
		self.load_state_dict(state_dict)


#------------------------------------------------------------------------------
#  BaseBackboneWrapper
#------------------------------------------------------------------------------
class BaseBackboneWrapper(BaseBackbone):
	def __init__(self):
		super(BaseBackboneWrapper, self).__init__()

	def train(self, mode=True):
		if mode:
			print("[%s] Switch to train mode" % (self.__class__.__name__))
		else:
			print("[%s] Switch to eval mode" % (self.__class__.__name__))

		super(BaseBackboneWrapper, self).train(mode)
		self._freeze_stages()
		if mode and self.norm_eval:
			for module in self.modules():
				# trick: eval have effect on BatchNorm only
				if isinstance(module, nn.BatchNorm2d):
					module.eval()
				elif isinstance(module, nn.Sequential):
					for m in module:
						if isinstance(module, (nn.BatchNorm2d, nn.GroupNorm)):
							m.eval()

	def init_from_imagenet(self, archname):
		pass

	def _freeze_stages(self):
		pass
