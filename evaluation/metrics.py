#------------------------------------------------------------------------------
#   Libraries
#------------------------------------------------------------------------------
import torch


#------------------------------------------------------------------------------
#   IoU with sigmoid
#------------------------------------------------------------------------------
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
#   mIoU
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