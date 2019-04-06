#------------------------------------------------------------------------------
#   Libraries
#------------------------------------------------------------------------------
import torch
import torch.nn as nn
import torchsummary
from utils.flops_counter import add_flops_counting_methods, flops_to_string


#------------------------------------------------------------------------------
#   BaseModel
#------------------------------------------------------------------------------
class BaseModel(nn.Module):
	def __init__(self):
		super(BaseModel, self).__init__()

	def summary(self, input_shape, batch_size=1, device='cpu'):
		print("[%s] Network summary..." % (self.__class__.__name__))
		torchsummary.summary(self, input_size=input_shape, batch_size=batch_size, device=device)
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
			elif isinstance(m, nn.BatchNorm2d):
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 0)
			elif isinstance(m, nn.Linear):
				m.weight.data.normal_(0, 0.01)
				m.bias.data.zero_()

	def preprocess_input(self, input):
		return (input/127.5 - 1.0)


#------------------------------------------------------------------------------
#   BaseBackbone
#------------------------------------------------------------------------------
class BaseBackbone(BaseModel):
	def __init__(self):
		super(BaseBackbone, self).__init__()

	def load_pretrained_model(self, pretrained_file):
		pretrain_dict = torch.load(pretrained_file, map_location='cpu')
		model_dict = {}
		state_dict = self.state_dict()
		print("[%s] Loading pretrained model..." % (self.__class__.__name__))
		for k, v in pretrain_dict.items():
			if k in state_dict:
				model_dict[k] = v
			else:
				print(k, "is ignored")
		state_dict.update(model_dict)
		self.load_state_dict(state_dict)