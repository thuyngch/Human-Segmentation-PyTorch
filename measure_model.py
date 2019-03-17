#------------------------------------------------------------------------------
#	Libraries
#------------------------------------------------------------------------------
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import torch, argparse
from time import time
from torchsummary import summary

from models import UNet, DeepLab, BiSeNet, PSPNet
from utils.flops_counter import add_flops_counting_methods, flops_to_string, get_model_parameters_number


#------------------------------------------------------------------------------
#   Argument parsing
#------------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Arguments for the script")

parser.add_argument('--use_cuda', action='store_true', default=True,
                    help='Use GPU acceleration')

parser.add_argument('--img_layers', type=int, default=3,
                    help='Number of image layers')

parser.add_argument('--input_sz', type=int, default=224,
                    help='Size of the input')

parser.add_argument('--n_measures', type=int, default=10,
                    help='Number of time measurements')

args = parser.parse_args()


#------------------------------------------------------------------------------
#	Create model
#------------------------------------------------------------------------------
# # UNet + MobileNetV2
# model = UNet(
# 	backbone="mobilenetv2",
# 	num_classes=2,
# 	pretrained_backbone="/media/antiaegis/storing/PyTorch-pretrained/mobilenetv2.pth",
# )

# # UNet + ResNet
# model = UNet(
#     backbone="resnet18",
#     num_classes=2,
#     pretrained_backbone="/media/antiaegis/storing/PyTorch-pretrained/resnet18.pth",
# )

# # DeepLabV3+
# model = DeepLab(
#     backbone='resnet18',
#     output_stride=16,
#     num_classes=2,
#     freeze_bn=False,
#     pretrained_backbone="/media/antiaegis/storing/PyTorch-pretrained/resnet18.pth",
# )

# # BiSeNet
# model = BiSeNet(
#     backbone='resnet18',
#     num_classes=2,
#     pretrained_backbone="/media/antiaegis/storing/PyTorch-pretrained/resnet18.pth",
# )

# PSPNet
model = PSPNet(
    backbone='resnet18',
    num_classes=2,
    pretrained_backbone="/media/antiaegis/storing/PyTorch-pretrained/resnet18.pth",
)


#------------------------------------------------------------------------------
#   Summary network
#------------------------------------------------------------------------------
model.eval()
summary(model, input_size=(args.img_layers, args.input_sz, args.input_sz), device='cpu')


#------------------------------------------------------------------------------
#   Measure time
#------------------------------------------------------------------------------
input = torch.randn([1, args.img_layers, args.input_sz, args.input_sz], dtype=torch.float)
if args.use_cuda:
    model = model.cuda()
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


#------------------------------------------------------------------------------
#   Count FLOPs
#------------------------------------------------------------------------------
print('----------------------------------------------------------------')
counter = add_flops_counting_methods(model)
counter.eval().start_flops_count()
_ = counter(input)
print('Flops:  {}'.format(flops_to_string(counter.compute_average_flops_cost())))
print('Params: ' + get_model_parameters_number(counter))