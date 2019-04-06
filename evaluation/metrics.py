#------------------------------------------------------------------------------
#   Libraries
#------------------------------------------------------------------------------
import torch
from torch.nn import functional as F


#------------------------------------------------------------------------------
#   Fundamental metrics
#------------------------------------------------------------------------------
def miou(logits, targets, eps=1e-6):
	"""
	logits: (torch.float32)  shape (N, C, H, W)
	targets: (torch.float32) shape (N, H, W), value {0,1,...,C-1}
	"""
	outputs = torch.argmax(logits, dim=1, keepdim=True).type(torch.int64)
	targets = torch.unsqueeze(targets, dim=1).type(torch.int64)
	outputs = torch.zeros_like(logits).scatter_(dim=1, index=outputs, src=torch.tensor(1.0)).type(torch.int8)
	targets = torch.zeros_like(logits).scatter_(dim=1, index=targets, src=torch.tensor(1.0)).type(torch.int8)

	inter = (outputs & targets).type(torch.float32).sum(dim=(2,3))
	union = (outputs | targets).type(torch.float32).sum(dim=(2,3))
	iou = inter / (union + eps)
	return iou.mean()


def iou_with_sigmoid(sigmoid, targets, eps=1e-6):
	"""
	sigmoid: (torch.float32) shape (N, 1, H, W)
	targets: (torch.float32) shape (N, H, W), value {0,1}
	"""
	outputs = torch.squeeze(sigmoid, dim=1).type(torch.int8)
	targets = targets.type(torch.int8)

	inter = (outputs & targets).type(torch.float32).sum(dim=(1,2))
	union = (outputs | targets).type(torch.float32).sum(dim=(1,2))
	iou = inter / (union + eps)
	return iou.mean()


#------------------------------------------------------------------------------
#   Custom IoU for BiSeNet
#------------------------------------------------------------------------------
def custom_bisenet_miou(logits, targets):
	"""
	logits: (torch.float32) (main_out, feat_os16_sup, feat_os32_sup) of shape (N, C, H, W)
	targets: (torch.float32) shape (N, H, W), value {0,1,...,C-1}
	"""
	if type(logits)==tuple:
		return miou(logits[0], targets)
	else:
		return miou(logits, targets)


#------------------------------------------------------------------------------
#   Custom IoU for PSPNet
#------------------------------------------------------------------------------
def custom_pspnet_miou(logits, targets):
	"""
	logits: (torch.float32) (main_out, aux_out) of shape (N, C, H, W), (N, C, H/8, W/8)
	targets: (torch.float32) shape (N, H, W), value {0,1,...,C-1}
	"""
	if type(logits)==tuple:
		return miou(logits[0], targets)
	else:
		return miou(logits, targets)


#------------------------------------------------------------------------------
#   Custom IoU for BiSeNet
#------------------------------------------------------------------------------
def custom_icnet_miou(logits, targets):
	"""
	logits: (torch.float32)
		[train_mode] (x_124_cls, x_12_cls, x_24_cls) of shape
						(N, C, H/4, W/4), (N, C, H/8, W/8), (N, C, H/16, W/16)

		[valid_mode] x_124_cls of shape (N, C, H, W)

	targets: (torch.float32) shape (N, H, W), value {0,1,...,C-1}
	"""
	if type(logits)==tuple:
		targets = torch.unsqueeze(targets, dim=1)
		targets = F.interpolate(targets, size=logits[0].shape[-2:], mode='bilinear', align_corners=True)[:,0,...]
		return miou(logits[0], targets)
	else:
		return miou(logits, targets)