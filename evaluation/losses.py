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


#------------------------------------------------------------------------------
#   Dice loss with sigmoid
#------------------------------------------------------------------------------
def dice_loss_with_sigmoid(sigmoid, targets, smooth=1.0):
	"""
	sigmoid: (torch.float32)  shape (N, 1, H, W)
	targets: (torch.float32) shape (N, H, W), value {0,1}
	"""
	outputs = torch.squeeze(sigmoid, dim=1)

	inter = outputs * targets
	dice = 1 - ((2*inter.sum(dim=(1,2)) + smooth) / (outputs.sum(dim=(1,2))+targets.sum(dim=(1,2)) + smooth))
	return dice.mean()


#------------------------------------------------------------------------------
#   Cross Entropy loss
#------------------------------------------------------------------------------
def cross_entropy_loss(logits, targets):
	"""
	logits: (torch.float32)  shape (N, C, H, W)
	targets: (torch.float32) shape (N, H, W), value {0,1,...,C-1}
	"""
	targets = targets.type(torch.int64)
	ce_loss = F.cross_entropy(logits, targets)
	return ce_loss