#------------------------------------------------------------------------------
#   Libraries
#------------------------------------------------------------------------------
import torch
import torch.nn.functional as F


#------------------------------------------------------------------------------
#   Dice loss
#------------------------------------------------------------------------------
def dice_loss(logits, targets, smooth=1.0):
	"""
	logits: (torch.float32)  shape (N, C, H, W)
	targets: (torch.float32) shape (N, H, W), value {0,1,...,C-1}
	"""
	outputs = F.softmax(logits, dim=1)
	targets = torch.unsqueeze(targets, dim=1)
	targets = torch.zeros_like(logits).scatter_(dim=1, index=targets.type(torch.int64), src=torch.tensor(1.0))

	inter = outputs * targets
	dice = 1 - ((2*inter.sum(dim=(2,3)) + smooth) / (outputs.sum(dim=(2,3))+targets.sum(dim=(2,3)) + smooth))
	return dice.mean()


def dice_loss_with_sigmoid(sigmoid, targets, smooth=1.0):
	"""
	sigmoid: (torch.float32)  shape (N, 1, H, W)
	targets: (torch.float32) shape (N, H, W), value {0,1}
	"""
	outputs = torch.squeeze(sigmoid, dim=1)

	inter = outputs * targets
	dice = 1 - ((2*inter.sum(dim=(1,2)) + smooth) / (outputs.sum(dim=(1,2))+targets.sum(dim=(1,2)) + smooth))
	dice = dice.mean()
	return dice


#------------------------------------------------------------------------------
#   Cross Entropy loss
#------------------------------------------------------------------------------
def ce_loss(logits, targets):
	"""
	logits: (torch.float32)  shape (N, C, H, W)
	targets: (torch.float32) shape (N, H, W), value {0,1,...,C-1}
	"""
	targets = targets.type(torch.int64)
	ce_loss = F.cross_entropy(logits, targets)
	return ce_loss


#------------------------------------------------------------------------------
#   Custom loss for BiSeNet
#------------------------------------------------------------------------------
def custom_bisenet_loss(logits, targets):
	"""
	logits: (torch.float32) (main_out, feat_os16_sup, feat_os32_sup) of shape (N, C, H, W)
	targets: (torch.float32) shape (N, H, W), value {0,1,...,C-1}
	"""
	if type(logits)==tuple:
		main_loss = ce_loss(logits[0], targets)
		os16_loss = ce_loss(logits[1], targets)
		os32_loss = ce_loss(logits[2], targets)
		return main_loss + os16_loss + os32_loss
	else:
		return ce_loss(logits, targets)