#------------------------------------------------------------------------------
#	Libraries
#------------------------------------------------------------------------------
import torch, argparse
from models import UNet
from base import VideoInference


#------------------------------------------------------------------------------
#   Argument parsing
#------------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Arguments for the script")

parser.add_argument('--use_cuda', action='store_true', default=False,
                    help='Use GPU acceleration')

parser.add_argument('--input_size', type=int, default=320,
                    help='Input size')

parser.add_argument('--checkpoint', type=str, default="model_best.pth",
                    help='Path to the trained model file')

args = parser.parse_args()


#------------------------------------------------------------------------------
#	Main execution
#------------------------------------------------------------------------------
# Build model
model = UNet(backbone="resnet18", num_classes=2)
trained_dict = torch.load(args.checkpoint, map_location="cpu")['state_dict']
model.load_state_dict(trained_dict, strict=False)
if args.use_cuda:
	model.cuda()
model.eval()


# Inference
inference = VideoInference(
    model=model,
    video_path=0,
    input_size=args.input_size,
    use_cuda=args.use_cuda,
    draw_mode='matting',
)
inference.run()