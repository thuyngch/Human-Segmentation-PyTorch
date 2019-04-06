#------------------------------------------------------------------------------
#	Libraries
#------------------------------------------------------------------------------
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import torch, argparse
from time import time
from torchsummary import summary

from models import UNet, DeepLabV3Plus, BiSeNet, PSPNet, ICNet


#------------------------------------------------------------------------------
#   Argument parsing
#------------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Arguments for the script")

parser.add_argument('--use_cuda', action='store_true', default=False,
					help='Use GPU acceleration')

parser.add_argument('--input_sz', type=int, default=320,
					help='Size of the input')

parser.add_argument('--n_measures', type=int, default=10,
					help='Number of time measurements')

args = parser.parse_args()


#------------------------------------------------------------------------------
#	Create model
#------------------------------------------------------------------------------
# # UNet
# model = UNet(
# 	backbone="mobilenetv2",
# 	num_classes=2,
#     pretrained_backbone=None,
# )

# # DeepLabV3+
# model = DeepLabV3Plus(
#     backbone='resnet18',
#     output_stride=16,
#     num_classes=2,
#     pretrained_backbone=None,
# )

# # BiSeNet
# model = BiSeNet(
#     backbone='resnet18',
#     num_classes=2,
#     pretrained_backbone=None,
# )

# PSPNet
model = PSPNet(
	backbone='resnet18',
	num_classes=2,
	pretrained_backbone=None,
)

# # ICNet
# model = ICNet(
#     backbone='resnet18',
#     num_classes=2,
#     pretrained_backbone=None,
# )


#------------------------------------------------------------------------------
#   Summary network
#------------------------------------------------------------------------------
model.train()
model.summary(input_shape=(3, args.input_sz, args.input_sz), device='cpu')


#------------------------------------------------------------------------------
#   Measure time
#------------------------------------------------------------------------------
input = torch.randn([1, 3, args.input_sz, args.input_sz], dtype=torch.float)
if args.use_cuda:
	model.cuda()
	input = input.cuda()

for _ in range(10):
	model(input)

start_time = time()
for _ in range(args.n_measures):
	model(input)
finish_time = time()

if args.use_cuda:
	print("Inference time on cuda: %.2f [ms]" % ((finish_time-start_time)*1000/args.n_measures))
	print("Inference fps on cuda: %.2f [fps]" % (1 / ((finish_time-start_time)/args.n_measures)))
else:
	print("Inference time on cpu: %.2f [ms]" % ((finish_time-start_time)*1000/args.n_measures))
	print("Inference fps on cpu: %.2f [fps]" % (1 / ((finish_time-start_time)/args.n_measures)))