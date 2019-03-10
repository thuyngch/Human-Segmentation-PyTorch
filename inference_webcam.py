#------------------------------------------------------------------------------
#	Libraries
#------------------------------------------------------------------------------
import cv2, torch, argparse
from time import time
import numpy as np
from models import UNet
from dataloaders import transforms


#------------------------------------------------------------------------------
#  Preprocessing
#------------------------------------------------------------------------------
mean = np.array([0.485, 0.456, 0.406])[None,None,:]
std = np.array([0.229, 0.224, 0.225])[None,None,:]

def preprocessing(image, expected_size=224, pad_value=0):
	image, pad_up, pad_left, h_new, w_new = transforms.resize_image(image, expected_size, pad_value, ret_params=True)
	image = image.astype(np.float32) / 255.0
	image = (image - mean) / std
	X = np.transpose(image, axes=(2, 0, 1))
	X = np.expand_dims(X, axis=0)
	X = torch.tensor(X, dtype=torch.float32)
	return X, pad_up, pad_left, h_new, w_new


#------------------------------------------------------------------------------
#  Draw image with transperency
#------------------------------------------------------------------------------
def draw_transperency(image, mask, color_f, color_b):
	alpha = np.zeros_like(image, dtype=np.uint8)
	alpha[mask==1, :] = color_f
	alpha[mask==0, :] = color_b
	image_alpha = cv2.add(image, alpha)
	return image_alpha


#------------------------------------------------------------------------------
#  Draw foreground pasted into background
#------------------------------------------------------------------------------
def draw_fore_to_back(image, mask, background, kernel_sz=13, sigma=0):
	mask_filtered = cv2.GaussianBlur(mask, (kernel_sz, kernel_sz), sigma)
	mask_filtered = np.expand_dims(mask_filtered, axis=2)
	mask_filtered = np.tile(mask_filtered, (1,1,3))
	image_alpha = image*mask_filtered + background*(1-mask_filtered)
	return image_alpha.astype(np.uint8)


#------------------------------------------------------------------------------
#   Argument parsing
#------------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Arguments for the script")

parser.add_argument('--use_cuda', action='store_true', default=True,
                    help='Use GPU acceleration')

parser.add_argument('--bg', type=str, default=None,
                    help='Path to the background image file')

parser.add_argument('--img_layers', type=int, default=3,
                    help='Number of layers of the input image')

parser.add_argument('--input_sz', type=int, default=224,
                    help='Input size')

parser.add_argument('--checkpoint', type=str, default="/media/antiaegis/storing/FORGERY/segmentation/checkpoints/ZaloHumanSeg/PyTorch/HumanSeg/0310_155053/model_best.pth",
                    help='Path to the trained model file')

args = parser.parse_args()


#------------------------------------------------------------------------------
#	Parameters
#------------------------------------------------------------------------------
# Alpha transperency
COLOR1 = [255, 0, 0]
COLOR2 = [0, 0, 255]

# Background
if args.bg is not None:
	BACKGROUND = cv2.imread(args.bg)[...,::-1]
	BACKGROUND = cv2.resize(BACKGROUND, (720,1280), interpolation=cv2.INTER_LINEAR)
else:
	KERNEL_SZ = 25
	SIGMA = 0


#------------------------------------------------------------------------------
#	Main execution
#------------------------------------------------------------------------------
model = UNet(
    n_classes=1,
    img_layers=args.img_layers,
    backbone="ResNet",
    backbone_args={
        "n_layers": 18,
        "filters": 64,
        "input_sz": args.input_sz,
        "pretrained": None,
    }
)
if args.use_cuda:
	model = model.cuda()
trained_dict = torch.load(args.checkpoint, map_location="cpu")['state_dict']
model.load_state_dict(trained_dict, strict=True)
model.eval()


# Predict frames
i = 0
cap = cv2.VideoCapture(0)
while(cap.isOpened()):
	# Read frame from camera
	start_time = time()
	_, frame = cap.read()
	image = frame[...,::-1]
	h, w = image.shape[:2]
	read_cam_time = time()

	# Predict mask
	X, pad_up, pad_left, h_new, w_new = preprocessing(image, expected_size=args.input_sz, pad_value=0)
	preproc_time = time()
	with torch.no_grad():
		if args.use_cuda:
			mask = model(X.cuda())[0,0,...].cpu().numpy()
		else:
			mask = model(X)[0,0,...].numpy()
	predict_time = time()

	# Resize mask to the original size
	mask = mask[pad_up: pad_up+h_new, pad_left: pad_left+w_new]
	mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)
	mask = mask.round()

	# Draw result
	if args.bg is None:
		image_alpha = draw_transperency(image, mask, COLOR1, COLOR2)
	else:
		image_alpha = draw_fore_to_back(image, mask, BACKGROUND, kernel_sz=KERNEL_SZ, sigma=SIGMA)
	draw_time = time()

	# Wait for interupt
	cv2.imshow('webcam', image_alpha[..., ::-1])
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

	# Print runtime
	read = read_cam_time-start_time
	preproc = preproc_time-read_cam_time
	pred = predict_time-preproc_time
	draw = draw_time-predict_time
	total = read + preproc + pred + draw
	fps = 1 / total
	print("read: %.3f [s]; preproc: %.3f [s]; pred: %.3f [s]; draw: %.3f [s]; total: %.3f [s]; fps: %.2f [Hz]" % 
		(read, preproc, pred, draw, total, fps))

cap.release()
cv2.destroyAllWindows()