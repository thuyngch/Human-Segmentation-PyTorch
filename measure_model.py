#------------------------------------------------------------------------------
#	Libraries
#------------------------------------------------------------------------------
import torch, argparse
from time import time
from torchsummary import summary

from models.UNet import UNet
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

parser.add_argument('--n_measures', type=int, default=100,
                    help='Number of time measurements')

args = parser.parse_args()


#------------------------------------------------------------------------------
#	Create model
#------------------------------------------------------------------------------
# # UNet + MobileNetV2
# model = UNet(
# 	n_classes=2,
# 	img_layers=args.img_layers,
# 	backbone="MobileNetV2",
# 	backbone_args={
# 		"input_sz": args.input_sz,
#         "alpha": 1.0,
#         "expansion": 6,
# 		"pretrained": "/media/antiaegis/storing/PyTorch-pretrained/mobilenet2.pth",
# 	}
# )

# UNet + ResNet
model = UNet(
    n_classes=2,
    img_layers=args.img_layers,
    backbone="ResNet",
    backbone_args={
        "n_layers": 18,
        "filters": 64,
        "input_sz": args.input_sz,
        "pretrained": "/media/antiaegis/storing/PyTorch-pretrained/resnet18.pth",
    }
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
    _ = model(input)

start_time = time()
for _ in range(args.n_measures):
    _ = model(input)
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